from __future__ import annotations

"""Linux process hardening for disposable model-execution workers.

This module deliberately uses only stdlib primitives.  It is *not* a complete sandbox:
the child still runs under the service UID and therefore needs an external service
manager/container policy for filesystem and network isolation.  The controls here reduce
ambient authority before untrusted model/backend code is loaded.
"""

import ctypes
import os
import resource
import sys


PR_SET_NO_NEW_PRIVS = 38


def _positive_int(name: str, value: int | None, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _set_no_new_privs() -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("no_new_privs hardening requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def apply_process_sandbox(
    *,
    memory_limit_bytes: int | None,
    nofile_limit: int = 64,
    clear_environment: bool = True,
    no_new_privs: bool = True,
) -> None:
    """Reduce ambient authority before importing/loading a model backend.

    The model worker gets no inherited environment by default, core dumps are disabled,
    the file-descriptor ceiling is reduced, and Linux privilege escalation through
    ``execve`` is blocked with ``PR_SET_NO_NEW_PRIVS``.  Address-space limiting remains
    optional because realistic limits are target/backend specific.
    """

    memory_limit_bytes = _positive_int("memory_limit_bytes", memory_limit_bytes)
    nofile_limit = _positive_int("nofile_limit", nofile_limit, minimum=16)
    if not isinstance(clear_environment, bool):
        raise ValueError("clear_environment must be boolean")
    if not isinstance(no_new_privs, bool):
        raise ValueError("no_new_privs must be boolean")

    if clear_environment:
        os.environ.clear()

    # Core files may contain model inputs, outputs, keys from process memory, or backend
    # internals.  A disposable worker should never create them.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    if nofile_limit is not None:
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        hard = nofile_limit if current_hard == resource.RLIM_INFINITY else min(nofile_limit, current_hard)
        soft = min(nofile_limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))

    if memory_limit_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))

    if no_new_privs:
        _set_no_new_privs()
