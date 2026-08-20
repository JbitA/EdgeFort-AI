from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .artifact import TrustedKeyring, manifest_from_dict, sha256_stream, verify_artifact
from .errors import ArtifactIntegrityError, RegistryError
from .util import InterProcessFileLock, atomic_write, canonical_json_bytes, safe_version, strict_json_loads

_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 4096
_MAX_KEY_ID_BYTES = 1024


class Registry:
    """Filesystem-backed immutable registry with verification on every read."""

    def __init__(self, root: str | Path, keyring: TrustedKeyring):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise RegistryError("registry root must be a regular directory")
        self.keyring = keyring
        self._ipc_lock = InterProcessFileLock(self.root / ".registry.lock")

    def _dir(self, version: str) -> Path:
        return self.root / safe_version(version)

    @staticmethod
    def _open_directory(directory: str | Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(Path(directory), flags)
        except OSError as e:
            raise RegistryError(f"registry entry cannot be opened safely: {e}") from e
        try:
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise RegistryError("registry entry is not a directory")
            return fd
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _open_regular_at(dir_fd: int, name: str, label: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(name, flags, dir_fd=dir_fd)
        except OSError as e:
            raise RegistryError(f"{label} missing or cannot be opened safely") from e
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RegistryError(f"{label} is not a regular file")
            return fd
        except Exception:
            os.close(fd)
            raise

    @classmethod
    def _read_small_file_at(cls, dir_fd: int, name: str, label: str, limit: int) -> bytes:
        fd = cls._open_regular_at(dir_fd, name, label)
        try:
            with os.fdopen(fd, "rb", closefd=True) as handle:
                data = handle.read(limit + 1)
        except Exception:
            # fd is owned by fdopen after construction; only explicit open failures reach here.
            raise
        if len(data) > limit:
            raise RegistryError(f"{label} exceeds size limit")
        return data

    @contextmanager
    def open_verified_model(self, directory: str | Path, expected_version: str | None = None):
        """Yield ``(model_handle, manifest)`` from one verified stable directory inode.

        Metadata and model bytes are opened relative to a no-follow directory descriptor.
        The model hash is checked on the same file handle later passed to the runtime loader,
        closing the path-reopen TOCTOU gap between verification and allocation/execution.
        """
        dir_fd = self._open_directory(directory)
        model_handle = None
        try:
            try:
                manifest_raw = self._read_small_file_at(
                    dir_fd, "manifest.json", "manifest", _MAX_MANIFEST_BYTES
                )
                signature_raw = self._read_small_file_at(
                    dir_fd, "signature.txt", "signature", _MAX_SIGNATURE_BYTES
                )
                key_id_raw = self._read_small_file_at(
                    dir_fd, "key_id.txt", "key id", _MAX_KEY_ID_BYTES
                )
                manifest = manifest_from_dict(strict_json_loads(manifest_raw.decode("utf-8")))
                signature = signature_raw.decode("utf-8").strip()
                key_id = key_id_raw.decode("utf-8").strip()
            except RegistryError:
                raise
            except Exception as e:
                raise RegistryError(f"corrupt registry entry: {e}") from e

            if expected_version is not None and manifest.version != expected_version:
                raise RegistryError("registry path/version does not match signed manifest")
            self.keyring.verify(key_id, manifest, signature)

            model_fd = self._open_regular_at(dir_fd, "model.npz", "model artifact")
            model_handle = os.fdopen(model_fd, "rb", closefd=True)
            st = os.fstat(model_handle.fileno())
            if st.st_size != manifest.model_bytes:
                raise ArtifactIntegrityError("model size mismatch")
            if sha256_stream(model_handle) != manifest.model_sha256:
                raise ArtifactIntegrityError("model checksum mismatch")
            yield model_handle, manifest
        finally:
            if model_handle is not None:
                model_handle.close()
            os.close(dir_fd)

    def verify_directory(self, directory: str | Path, expected_version: str | None = None):
        with self.open_verified_model(directory, expected_version=expected_version) as (_, manifest):
            return manifest

    def _assert_release_sequence_available(self, release_sequence: int, *, version: str) -> None:
        """Fail closed unless ``release_sequence`` is globally unique in this registry.

        The anti-rollback counter is an identity coordinate, not merely ordering metadata.
        Reusing it would make external-anchor crash recovery ambiguous. Hidden staging/lock
        entries are excluded; every published version directory must verify successfully.
        """
        for entry in self.root.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_symlink() or not entry.is_dir():
                raise RegistryError("unexpected non-directory entry in registry")
            try:
                existing = self.verify_directory(entry, expected_version=entry.name)
            except Exception as e:
                raise RegistryError(
                    "cannot establish release-sequence uniqueness because registry contains "
                    f"an invalid published entry: {entry.name}"
                ) from e
            if existing.release_sequence == release_sequence and existing.version != version:
                raise RegistryError(
                    f"release_sequence {release_sequence} is already assigned to {existing.version}"
                )

    def assert_release_sequence_identity(self, version: str, release_sequence: int) -> None:
        """Verify that one canonical published version uniquely owns a release sequence."""
        safe_version(version)
        with self._ipc_lock.shared():
            target = self.inspect(version)[1]
            if target.release_sequence != release_sequence:
                raise RegistryError("version does not match expected release sequence")
            self._assert_release_sequence_available(release_sequence, version=version)

    def add(self, model_path, manifest, signature: str, key_id: str) -> Path:
        safe_version(manifest.version)
        self.keyring.verify(key_id, manifest, signature)
        verify_artifact(model_path, manifest)
        final = self._dir(manifest.version)

        with self._ipc_lock.exclusive():
            if final.exists() or final.is_symlink():
                raise RegistryError("registry versions are immutable")
            self._assert_release_sequence_available(
                manifest.release_sequence, version=manifest.version
            )

            tmp = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root))
            try:
                shutil.copy2(model_path, tmp / "model.npz")
                atomic_write(tmp / "manifest.json", canonical_json_bytes(manifest.to_dict()) + b"\n")
                atomic_write(tmp / "signature.txt", (signature + "\n").encode())
                atomic_write(tmp / "key_id.txt", (key_id + "\n").encode())
                self.verify_directory(tmp, expected_version=manifest.version)
                os.replace(tmp, final)
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise
        return final

    def inspect(self, version: str):
        d = self._dir(version)
        manifest = self.verify_directory(d, expected_version=version)
        return d, manifest

    def versions(self):
        out = []
        for p in self.root.iterdir():
            if p.is_dir() and not p.name.startswith(".") and not p.is_symlink():
                try:
                    self.inspect(p.name)
                    out.append(p.name)
                except Exception:
                    continue
        return sorted(out)
