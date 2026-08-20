from __future__ import annotations
import json, os, re, stat, tempfile, threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")
MAX_RELEASE_SEQUENCE = (1 << 63) - 1

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def strict_json_loads(data: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate object keys and non-finite constants."""
    def object_no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    return json.loads(
        data, object_pairs_hook=object_no_duplicates, parse_constant=reject_constant
    )

def validate_semver(version: str) -> None:
    if not isinstance(version, str) or not _SEMVER.match(version):
        raise ValueError(f"invalid semantic version: {version!r}")

def safe_version(version: str) -> str:
    validate_semver(version)
    return version

def atomic_write(path: str | Path, data: bytes, mode: int = 0o600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(dfd)
            finally: os.close(dfd)
        except OSError:
            pass
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


class InterProcessFileLock:
    """POSIX advisory shared/exclusive file lock with thread serialization.

    Edge targets qualified by this repository are Linux/Unix. The lock file is kept
    separate from atomic state files so rename-based commits do not invalidate the lock.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    @contextmanager
    def acquire(self, *, exclusive: bool):
        try:
            import fcntl
        except ImportError as e:  # pragma: no cover - qualified targets are POSIX
            raise RuntimeError("inter-process locking requires POSIX fcntl") from e
        with self._thread_lock:
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.path, flags, 0o600)
            except OSError as e:
                raise RuntimeError(f"lock file cannot be opened safely: {e}") from e
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError("lock path is not a regular file")
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                yield
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def shared(self):
        return self.acquire(exclusive=False)

    def exclusive(self):
        return self.acquire(exclusive=True)
