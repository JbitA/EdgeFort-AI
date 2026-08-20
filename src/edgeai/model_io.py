from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping
import math
import os
import stat
import zipfile

import numpy as np

from .errors import ValidationError

DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 10_000
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}


@dataclass(frozen=True)
class ArrayHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    data_bytes: int


def open_regular_binary(path: str | Path) -> BinaryIO:
    """Open one stable regular-file inode without following a final symlink."""
    p = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(p, flags)
    except OSError as e:
        raise ValidationError(f"model artifact cannot be opened safely: {e}") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValidationError("model artifact is not a regular file")
        return os.fdopen(fd, "rb", closefd=True)
    except Exception:
        os.close(fd)
        raise



@contextmanager
def model_source(source):
    """Yield a seekable binary model handle, preserving caller-owned handles."""
    if hasattr(source, "read") and hasattr(source, "seek"):
        source.seek(0)
        try:
            yield source
        finally:
            source.seek(0)
        return
    with open_regular_binary(source) as handle:
        yield handle


def _read_npy_header(member) -> ArrayHeader:
    try:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            reader = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"unsupported NPY version {version!r}")
        shape, fortran_order, dtype = reader(member, max_header_size=_MAX_NPY_HEADER_BYTES)
    except Exception as e:
        raise ValidationError(f"invalid NPY member header: {e}") from e
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValidationError("object dtypes are forbidden in model artifacts")
    if not isinstance(shape, tuple) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValidationError("invalid array shape in model artifact")
    count = math.prod(shape) if shape else 1
    data_bytes = count * dtype.itemsize
    return ArrayHeader(tuple(shape), dtype, bool(fortran_order), data_bytes)


def preflight_npz(
    fileobj: BinaryIO,
    expected_arrays: set[str],
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES,
) -> Mapping[str, ArrayHeader]:
    """Validate NPZ structure and allocation bounds before NumPy allocates arrays.

    The caller must keep ``fileobj`` open and should load from this same handle after
    preflight. That preserves inode identity across the preflight/load boundary.
    """
    if isinstance(max_uncompressed_bytes, bool) or not isinstance(max_uncompressed_bytes, int):
        raise ValueError("max_uncompressed_bytes must be an integer")
    if max_uncompressed_bytes < 1024:
        raise ValueError("max_uncompressed_bytes must be >= 1024")
    if not expected_arrays:
        raise ValueError("expected_arrays must not be empty")

    try:
        fileobj.seek(0)
        with zipfile.ZipFile(fileobj, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            expected_members = {f"{name}.npy" for name in expected_arrays}
            if len(names) != len(set(names)):
                raise ValidationError("duplicate NPZ members are forbidden")
            if set(names) != expected_members:
                raise ValidationError("unexpected or missing arrays in model artifact")
            if len(infos) > 16:
                raise ValidationError("too many arrays in model artifact")

            declared_total = 0
            headers: dict[str, ArrayHeader] = {}
            for info in infos:
                if info.is_dir() or "/" in info.filename or "\\" in info.filename:
                    raise ValidationError("nested NPZ members are forbidden")
                if info.flag_bits & 0x1:
                    raise ValidationError("encrypted NPZ members are forbidden")
                if info.compress_type not in _ALLOWED_COMPRESSION:
                    raise ValidationError("unsupported NPZ compression method")
                if info.file_size < 1:
                    raise ValidationError("empty NPZ member")
                declared_total += info.file_size
                if declared_total > max_uncompressed_bytes:
                    raise ValidationError("model expanded size exceeds configured limit")

                with archive.open(info, mode="r") as member:
                    header = _read_npy_header(member)
                # The NPY member must contain at least the bytes its own header declares.
                # This catches forged huge shapes before NumPy can allocate them.
                if header.data_bytes > info.file_size:
                    raise ValidationError("array shape exceeds NPZ member size")
                headers[info.filename[:-4]] = header
    except ValidationError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        raise ValidationError(f"invalid NPZ model artifact: {e}") from e
    finally:
        fileobj.seek(0)

    return headers


def validate_linear_headers(headers: Mapping[str, ArrayHeader], *, quantized: bool) -> None:
    weight_name = "qW" if quantized else "W"
    weights = headers[weight_name]
    bias = headers["b"]
    abi = headers["abi"]
    if weights.shape == () or len(weights.shape) != 2 or min(weights.shape) < 1:
        raise ValidationError("model weights must be a non-empty rank-2 array")
    if len(bias.shape) != 1 or bias.shape != (weights.shape[0],):
        raise ValidationError("model bias shape does not match output dimension")
    if abi.shape != ():
        raise ValidationError("runtime ABI marker must be scalar")

    if quantized:
        scales = headers["scales"]
        if weights.dtype != np.dtype(np.int8):
            raise ValidationError("quantized weights must be int8")
        if len(scales.shape) != 1 or scales.shape != (weights.shape[0],):
            raise ValidationError("quantization scale shape does not match output dimension")
        if scales.dtype != np.dtype(np.float32) or bias.dtype != np.dtype(np.float32):
            raise ValidationError("quantization scales and bias must be float32")
    else:
        if weights.dtype != np.dtype(np.float32) or bias.dtype != np.dtype(np.float32):
            raise ValidationError("float model weights and bias must be float32")

    if abi.dtype.kind not in {"U", "S"}:
        raise ValidationError("runtime ABI marker must be a string")
