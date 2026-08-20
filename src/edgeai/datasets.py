from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from .errors import ValidationError
from .util import canonical_json_bytes


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    title: str
    source_url: str
    doi: str
    license_name: str
    license_url: str
    source_page: str
    max_download_bytes: int
    max_uncompressed_bytes: int


UCI_HAR = DatasetSpec(
    slug="uci-har",
    title="Human Activity Recognition Using Smartphones",
    source_url=(
        "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip"
    ),
    doi="10.24432/C54S4K",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    source_page=(
        "https://archive.ics.uci.edu/dataset/240/"
        "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones"
    ),
    max_download_bytes=80 * 1024 * 1024,
    max_uncompressed_bytes=300 * 1024 * 1024,
)

_UCI_HAR_REQUIRED = {
    "UCI HAR Dataset/train/X_train.txt": "train/X.txt",
    "UCI HAR Dataset/train/y_train.txt": "train/y.txt",
    "UCI HAR Dataset/train/subject_train.txt": "train/subject.txt",
    "UCI HAR Dataset/test/X_test.txt": "test/X.txt",
    "UCI HAR Dataset/test/y_test.txt": "test/y.txt",
    "UCI HAR Dataset/test/subject_test.txt": "test/subject.txt",
    "UCI HAR Dataset/activity_labels.txt": "activity_labels.txt",
    "UCI HAR Dataset/features.txt": "features.txt",
    "UCI HAR Dataset/README.txt": "UPSTREAM_README.txt",
}


@dataclass(frozen=True)
class HARSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_subjects: np.ndarray
    test_subjects: np.ndarray
    activity_names: tuple[str, ...]
    feature_names: tuple[str, ...]


def _safe_zip_members(archive: zipfile.ZipFile, spec: DatasetSpec) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValidationError("dataset archive contains an unsafe path")
        # UNIX symlink bit, when present in the external attributes.
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValidationError("dataset archive contains a symlink")
        if info.file_size < 0:
            raise ValidationError("dataset archive contains an invalid member size")
        total += info.file_size
        if total > spec.max_uncompressed_bytes:
            raise ValidationError("dataset archive exceeds uncompressed size limit")
        if name in infos:
            raise ValidationError("dataset archive contains duplicate members")
        infos[name] = info
    return infos


def _copy_limited(source, destination, *, max_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        count += len(chunk)
        if count > max_bytes:
            raise ValidationError("dataset download exceeds configured size limit")
        digest.update(chunk)
        destination.write(chunk)
    return count, digest.hexdigest()


def _write_provenance(root: Path, spec: DatasetSpec, *, source_bytes: int, source_sha256: str) -> None:
    provenance = {
        "schema_version": 1,
        "dataset": spec.slug,
        "title": spec.title,
        "source_url": spec.source_url,
        "source_page": spec.source_page,
        "doi": spec.doi,
        "license": spec.license_name,
        "license_url": spec.license_url,
        "retrieved_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_bytes": source_bytes,
        "source_sha256": source_sha256,
        "note": "The checksum records the exact upstream archive retrieved for this run; it is not asserted as an upstream-published checksum.",
    }
    (root / "provenance.json").write_bytes(canonical_json_bytes(provenance) + b"\n")


def fetch_uci_har(destination: str | Path, *, force: bool = False, timeout_s: float = 60.0) -> Path:
    """Download the CC BY 4.0 UCI HAR dataset and extract only required files.

    Network access is explicit. The archive is streamed through a hard byte limit, then
    structurally inspected before any member is written. A provenance record captures the
    exact archive SHA-256 observed during acquisition.
    """
    dest = Path(destination)
    if dest.exists() and not force:
        required = [dest / rel for rel in _UCI_HAR_REQUIRED.values()]
        if (dest / "provenance.json").is_file() and all(path.is_file() for path in required):
            return dest
        raise ValidationError("dataset destination exists but is incomplete; use force=True to replace it")

    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.fetch-", dir=parent))
    archive_path = staging / "source.zip"
    try:
        request = Request(UCI_HAR.source_url, headers={"User-Agent": "edge-ai-deployment-platform/1.0"})
        with urlopen(request, timeout=timeout_s) as response, archive_path.open("wb") as handle:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise ValidationError("dataset source returned invalid Content-Length") from exc
                if declared > UCI_HAR.max_download_bytes:
                    raise ValidationError("dataset source exceeds configured download size limit")
            source_bytes, source_sha256 = _copy_limited(
                response, handle, max_bytes=UCI_HAR.max_download_bytes
            )

        extracted = staging / "dataset"
        extracted.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = _safe_zip_members(archive, UCI_HAR)
                missing = sorted(set(_UCI_HAR_REQUIRED) - set(infos))
                if missing:
                    raise ValidationError(f"dataset archive missing required members: {missing}")
                for upstream, relative in _UCI_HAR_REQUIRED.items():
                    target = extracted / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(infos[upstream], "r") as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
        except zipfile.BadZipFile as exc:
            raise ValidationError("dataset source is not a valid ZIP archive") from exc

        _write_provenance(
            extracted,
            UCI_HAR,
            source_bytes=source_bytes,
            source_sha256=source_sha256,
        )
        if dest.exists():
            shutil.rmtree(dest)
        extracted.replace(dest)
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_vector(path: Path, *, dtype) -> np.ndarray:
    try:
        value = np.loadtxt(path, dtype=dtype)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"cannot parse dataset file {path}") from exc
    return np.asarray(value)


def load_uci_har(root: str | Path) -> HARSplit:
    root = Path(root)
    required = [root / rel for rel in _UCI_HAR_REQUIRED.values()] + [root / "provenance.json"]
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise ValidationError("UCI HAR dataset is missing or incomplete; run the dataset fetch command")

    x_train = _load_vector(root / "train/X.txt", dtype=np.float32)
    x_test = _load_vector(root / "test/X.txt", dtype=np.float32)
    y_train_raw = _load_vector(root / "train/y.txt", dtype=np.int64).reshape(-1)
    y_test_raw = _load_vector(root / "test/y.txt", dtype=np.int64).reshape(-1)
    train_subjects = _load_vector(root / "train/subject.txt", dtype=np.int64).reshape(-1)
    test_subjects = _load_vector(root / "test/subject.txt", dtype=np.int64).reshape(-1)

    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise ValidationError("UCI HAR feature matrices have inconsistent shapes")
    if x_train.shape[1] != 561:
        raise ValidationError("UCI HAR feature count differs from the qualified 561-feature schema")
    if not (np.isfinite(x_train).all() and np.isfinite(x_test).all()):
        raise ValidationError("UCI HAR features contain non-finite values")
    if len(x_train) != len(y_train_raw) or len(x_train) != len(train_subjects):
        raise ValidationError("UCI HAR training rows are inconsistent")
    if len(x_test) != len(y_test_raw) or len(x_test) != len(test_subjects):
        raise ValidationError("UCI HAR test rows are inconsistent")
    if set(train_subjects.tolist()) & set(test_subjects.tolist()):
        raise ValidationError("UCI HAR subject split is not disjoint")

    labels = np.unique(np.concatenate([y_train_raw, y_test_raw]))
    if not np.array_equal(labels, np.arange(1, 7, dtype=np.int64)):
        raise ValidationError("UCI HAR activity labels differ from expected 1..6 schema")
    y_train = (y_train_raw - 1).astype(np.int64)
    y_test = (y_test_raw - 1).astype(np.int64)

    try:
        activities = []
        for line in (root / "activity_labels.txt").read_text(encoding="utf-8").splitlines():
            idx, name = line.strip().split(maxsplit=1)
            activities.append((int(idx), name))
        activities.sort()
        activity_names = tuple(name for _, name in activities)
        if tuple(idx for idx, _ in activities) != tuple(range(1, 7)):
            raise ValueError

        features = []
        for line in (root / "features.txt").read_text(encoding="utf-8").splitlines():
            idx, name = line.strip().split(maxsplit=1)
            features.append((int(idx), name))
        features.sort()
        feature_names = tuple(name for _, name in features)
        if len(feature_names) != 561:
            raise ValueError
    except (OSError, ValueError) as exc:
        raise ValidationError("UCI HAR metadata files are malformed") from exc

    return HARSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        activity_names=activity_names,
        feature_names=feature_names,
    )


def read_dataset_provenance(root: str | Path) -> dict[str, object]:
    path = Path(root) / "provenance.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("dataset provenance is unavailable or malformed") from exc
    if not isinstance(value, dict):
        raise ValidationError("dataset provenance must be a JSON object")
    return value
