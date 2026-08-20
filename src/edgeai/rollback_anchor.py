from __future__ import annotations

import os
import stat
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from .errors import DeploymentError
from .util import MAX_RELEASE_SEQUENCE, InterProcessFileLock, atomic_write, canonical_json_bytes, strict_json_loads

_MAX_ANCHOR_BYTES = 4096


@runtime_checkable
class RollbackAnchor(Protocol):
    """External monotonic anti-rollback trust anchor.

    Implementations must persist a value that cannot decrease across ordinary reboot and
    storage rollback in the threat model they claim to cover. ``advance_to`` must return
    only after the new value is durable. A production TPM/secure-element adapter should
    implement this protocol without storing the protected value on the deployment
    filesystem.
    """

    def read_floor(self) -> int:
        """Return the durable monotonic floor, or ``-1`` for a newly provisioned device."""

    def advance_to(self, release_sequence: int) -> int:
        """Durably advance to ``release_sequence`` and return the resulting floor."""


class MemoryRollbackAnchor:
    """Thread-safe software test double; not a production anti-rollback boundary."""

    def __init__(self, initial: int = -1):
        self._validate(initial)
        self._value = initial
        self._lock = threading.Lock()

    @staticmethod
    def _validate(value: int) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -1
            or value > MAX_RELEASE_SEQUENCE
        ):
            raise ValueError(
                f"rollback anchor value must be an integer in [-1, {MAX_RELEASE_SEQUENCE}]"
            )

    def read_floor(self) -> int:
        with self._lock:
            return self._value

    def advance_to(self, release_sequence: int) -> int:
        self._validate(release_sequence)
        with self._lock:
            if release_sequence < self._value:
                raise DeploymentError("rollback anchor cannot decrease")
            self._value = release_sequence
            return self._value


class FileRollbackAnchor:
    """Persistent *software* anchor for integration tests and controlled labs.

    This is intentionally not represented as protection against whole-filesystem rollback.
    It is useful for exercising crash/restart behavior and the adapter contract before a
    TPM, secure element, monotonic boot counter, or remote trusted monotonic service is
    integrated. Place it outside the deployment root when using it for fault tests.
    """

    def __init__(self, path: str | Path, *, initial: int = -1):
        MemoryRollbackAnchor._validate(initial)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise ValueError("rollback anchor parent must be a regular directory")
        self._thread_lock = threading.RLock()
        self._ipc_lock = InterProcessFileLock(self.path.with_name(self.path.name + ".lock"))
        with self._thread_lock, self._ipc_lock.exclusive():
            if not self.path.exists() and not self.path.is_symlink():
                self._write(initial)
            self._read()

    def _read(self) -> int:
        if self.path.is_symlink():
            raise DeploymentError("rollback anchor path must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags)
        except OSError as e:
            raise DeploymentError(f"rollback anchor cannot be opened safely: {e}") from e
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise DeploymentError("rollback anchor is not a regular file")
            with os.fdopen(fd, "rb", closefd=True) as handle:
                raw = handle.read(_MAX_ANCHOR_BYTES + 1)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        if len(raw) > _MAX_ANCHOR_BYTES:
            raise DeploymentError("rollback anchor exceeds size limit")
        try:
            data = strict_json_loads(raw.decode("utf-8"))
        except Exception as e:
            raise DeploymentError(f"invalid rollback anchor: {e}") from e
        if not isinstance(data, dict) or set(data) != {"floor", "schema_version"}:
            raise DeploymentError("invalid rollback anchor structure")
        if data["schema_version"] != 1:
            raise DeploymentError("unsupported rollback anchor schema")
        floor = data["floor"]
        try:
            MemoryRollbackAnchor._validate(floor)
        except ValueError as e:
            raise DeploymentError(str(e)) from e
        return floor

    def _write(self, floor: int) -> None:
        payload = {"schema_version": 1, "floor": floor}
        atomic_write(self.path, canonical_json_bytes(payload) + b"\n")

    def read_floor(self) -> int:
        with self._thread_lock, self._ipc_lock.shared():
            return self._read()

    def advance_to(self, release_sequence: int) -> int:
        MemoryRollbackAnchor._validate(release_sequence)
        with self._thread_lock, self._ipc_lock.exclusive():
            current = self._read()
            if release_sequence < current:
                raise DeploymentError("rollback anchor cannot decrease")
            if release_sequence > current:
                self._write(release_sequence)
            return release_sequence
