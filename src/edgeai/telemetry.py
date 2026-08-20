from __future__ import annotations

import os
import re
import secrets
import stat
import threading
from pathlib import Path

from .util import InterProcessFileLock, atomic_write, canonical_json_bytes, strict_json_loads

_EVENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_RECORD_MARKER = "edgeai-spool-v1"


class OfflineSpool:
    """Bounded durable JSONL telemetry spool with retry-stable event IDs.

    ``flush`` preserves the historical API and sends only the original event payload.
    ``flush_records`` sends ``{"event_id": ..., "event": ...}``, allowing a remote sink
    to use the stable ID as an idempotency key when a crash occurs after the sink side
    effect but before the local spool rewrite. Legacy raw JSONL records remain readable
    but naturally have no persistent event ID.
    """

    def __init__(self, path, max_bytes=4 * 1024 * 1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self._ipc_lock = InterProcessFileLock(self.path.with_name(self.path.name + ".lock"))
        self.corrupt = 0
        self.dropped = 0
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1024:
            raise ValueError("max_bytes too small")
        self._reject_unsafe_existing_path()

    def _reject_unsafe_existing_path(self) -> None:
        try:
            st = os.lstat(self.path)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("telemetry spool path must be a regular file")

    def _read_bytes(self) -> bytes:
        try:
            fd = os.open(
                self.path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        except FileNotFoundError:
            return b""
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("telemetry spool path must be a regular file")
            with os.fdopen(fd, "rb", closefd=True) as handle:
                return handle.read(self.max_bytes + 1)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _append_line(self, line: bytes) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("telemetry spool path must be a regular file")
            with os.fdopen(fd, "ab", closefd=True) as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    @staticmethod
    def _validate_event_id(event_id: str) -> str:
        if not isinstance(event_id, str) or not _EVENT_ID_RE.fullmatch(event_id):
            raise ValueError("event_id must be 32 lowercase hexadecimal characters")
        return event_id

    @classmethod
    def _encode_record(cls, event, event_id: str) -> bytes:
        cls._validate_event_id(event_id)
        record = {"_record": _RECORD_MARKER, "event_id": event_id, "event": event}
        return canonical_json_bytes(record) + b"\n"

    @classmethod
    def _decode_record(cls, raw: bytes):
        value = strict_json_loads(raw)
        if (
            isinstance(value, dict)
            and set(value) == {"_record", "event_id", "event"}
            and value.get("_record") == _RECORD_MARKER
        ):
            return cls._validate_event_id(value["event_id"]), value["event"], True
        return None, value, False

    def append(self, event, *, event_id: str | None = None):
        supplied_event_id = event_id is not None
        event_id = self._validate_event_id(event_id) if supplied_event_id else secrets.token_hex(16)
        line = self._encode_record(event, event_id)
        if len(line) > self.max_bytes:
            with self.lock:
                self.dropped += 1
            return False

        with self.lock, self._ipc_lock.exclusive():
            raw = self._read_bytes()
            if len(raw) > self.max_bytes:
                raise ValueError("telemetry spool exceeds configured size")

            # Caller-supplied IDs support local idempotent enqueue as well as remote
            # idempotent delivery. Reusing an ID for different content is rejected.
            if supplied_event_id:
                for existing_raw in raw.splitlines():
                    try:
                        existing_id, existing_event, _ = self._decode_record(existing_raw)
                    except Exception:
                        continue
                    if existing_id == event_id:
                        if canonical_json_bytes(existing_event) != canonical_json_bytes(event):
                            raise ValueError("event_id is already associated with different content")
                        return True

            existing = len(raw)
            if existing + len(line) > self.max_bytes:
                lines = raw.splitlines(keepends=True)
                total = sum(map(len, lines))
                drop_count = 0
                while lines and total + len(line) > self.max_bytes:
                    removed = lines.pop(0)
                    total -= len(removed)
                    drop_count += 1
                self.dropped += drop_count
                atomic_write(self.path, b"".join(lines) + line)
            else:
                self._append_line(line)
        return True

    def _flush(self, sink, max_events=None, *, include_record: bool):
        if max_events is not None and (
            isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 0
        ):
            raise ValueError("max_events must be a non-negative integer")
        with self.lock, self._ipc_lock.exclusive():
            raw_spool = self._read_bytes()
            if not raw_spool:
                return 0
            if len(raw_spool) > self.max_bytes:
                raise ValueError("telemetry spool exceeds configured size")
            records = raw_spool.splitlines()
            remaining = []
            sent = 0
            for index, raw in enumerate(records):
                if max_events is not None and sent >= max_events:
                    remaining.extend(records[index:])
                    break
                try:
                    event_id, event, enveloped = self._decode_record(raw)
                except Exception:
                    self.corrupt += 1
                    continue
                try:
                    if include_record:
                        sink({"event_id": event_id, "event": event, "persistent_id": enveloped})
                    else:
                        sink(event)
                    sent += 1
                except Exception:
                    remaining.extend(records[index:])
                    break
            atomic_write(self.path, b"".join(x + b"\n" for x in remaining))
            return sent

    def flush(self, sink, max_events=None):
        return self._flush(sink, max_events=max_events, include_record=False)

    def flush_records(self, sink, max_events=None):
        """Flush delivery records containing stable event IDs for idempotent sinks."""
        return self._flush(sink, max_events=max_events, include_record=True)

    def stats(self):
        with self.lock, self._ipc_lock.shared():
            raw = self._read_bytes()
            if len(raw) > self.max_bytes:
                raise ValueError("telemetry spool exceeds configured size")
            persistent = 0
            legacy = 0
            for line in raw.splitlines():
                try:
                    _, _, enveloped = self._decode_record(line)
                except Exception:
                    continue
                if enveloped:
                    persistent += 1
                else:
                    legacy += 1
            return {
                "bytes": len(raw),
                "dropped": self.dropped,
                "corrupt": self.corrupt,
                "persistent_id_records": persistent,
                "legacy_records": legacy,
            }
