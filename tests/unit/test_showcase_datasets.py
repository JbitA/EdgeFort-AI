from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from edgeai.datasets import UCI_HAR, fetch_uci_har, load_uci_har, read_dataset_provenance
from edgeai.errors import ValidationError
from edgeai.showcase import feature_shift_proxy, run_showcase


def _har_files() -> dict[str, bytes]:
    labels = np.tile(np.arange(6), 4)
    x_train = np.zeros((24, 561), dtype=np.float32)
    x_test = np.zeros((24, 561), dtype=np.float32)
    for row, label in enumerate(labels):
        x_train[row, label] = 1.0
        x_train[row, 6] = row / 20.0
        x_test[row, label] = 0.9
        x_test[row, 6] = row / 22.0

    def matrix(value):
        buf = io.StringIO()
        np.savetxt(buf, value, fmt="%.7f")
        return buf.getvalue().encode()

    features = "\n".join(f"{i} feature-{i}" for i in range(1, 562)) + "\n"
    activities = "\n".join(
        [
            "1 WALKING",
            "2 WALKING_UPSTAIRS",
            "3 WALKING_DOWNSTAIRS",
            "4 SITTING",
            "5 STANDING",
            "6 LAYING",
        ]
    ) + "\n"
    prefix = "UCI HAR Dataset/"
    return {
        prefix + "train/X_train.txt": matrix(x_train),
        prefix + "train/y_train.txt": ("\n".join(str(v) for v in ([1,2,3,4,5,6] * 4)) + "\n").encode(),
        prefix + "train/subject_train.txt": ("\n".join(str(v) for v in range(1,25)) + "\n").encode(),
        prefix + "test/X_test.txt": matrix(x_test),
        prefix + "test/y_test.txt": ("\n".join(str(v) for v in ([1,2,3,4,5,6] * 4)) + "\n").encode(),
        prefix + "test/subject_test.txt": ("\n".join(str(v) for v in range(25,49)) + "\n").encode(),
        prefix + "activity_labels.txt": activities.encode(),
        prefix + "features.txt": features.encode(),
        prefix + "README.txt": b"fixture\n",
    }


def _zip_bytes(extra: dict[str, bytes] | None = None) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in _har_files().items():
            archive.writestr(name, value)
        for name, value in (extra or {}).items():
            archive.writestr(name, value)
    return out.getvalue()


class _Response(io.BytesIO):
    def __init__(self, value: bytes, length: str | None = None):
        super().__init__(value)
        self.headers = {} if length is None else {"Content-Length": length}


def test_fetch_and_load_uci_har_records_provenance(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload, str(len(payload))))
    root = fetch_uci_har(tmp_path / "har")
    split = load_uci_har(root)
    assert split.x_train.shape == (24, 561)
    assert split.x_test.shape == (24, 561)
    assert split.y_train.tolist() == [0, 1, 2, 3, 4, 5] * 4
    assert set(split.train_subjects).isdisjoint(set(split.test_subjects))
    assert split.activity_names[0] == "WALKING"
    assert len(split.feature_names) == 561
    provenance = read_dataset_provenance(root)
    assert provenance["license"] == "CC BY 4.0"
    assert provenance["doi"] == UCI_HAR.doi
    assert len(provenance["source_sha256"]) == 64
    assert fetch_uci_har(root) == root  # complete destinations are idempotent


def test_fetch_rejects_incomplete_existing_destination(tmp_path):
    root = tmp_path / "har"
    root.mkdir()
    with pytest.raises(ValidationError, match="incomplete"):
        fetch_uci_har(root)


def test_fetch_force_replaces_existing_destination(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    root = tmp_path / "har"
    root.mkdir()
    (root / "junk").write_text("old")
    fetch_uci_har(root, force=True)
    assert not (root / "junk").exists()
    assert (root / "provenance.json").is_file()


def test_fetch_rejects_bad_content_length(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload, "not-an-int"))
    with pytest.raises(ValidationError, match="Content-Length"):
        fetch_uci_har(tmp_path / "har")


def test_fetch_rejects_declared_oversize(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr(
        "edgeai.datasets.urlopen",
        lambda *a, **k: _Response(payload, str(UCI_HAR.max_download_bytes + 1)),
    )
    with pytest.raises(ValidationError, match="download size"):
        fetch_uci_har(tmp_path / "har")


def test_fetch_rejects_missing_required_member(tmp_path, monkeypatch):
    files = _har_files()
    files.pop("UCI HAR Dataset/features.txt")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    payload = out.getvalue()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    with pytest.raises(ValidationError, match="missing required"):
        fetch_uci_har(tmp_path / "har")


def test_fetch_rejects_unsafe_archive_path(tmp_path, monkeypatch):
    payload = _zip_bytes({"../escape": b"bad"})
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    with pytest.raises(ValidationError, match="unsafe path"):
        fetch_uci_har(tmp_path / "har")


def test_fetch_rejects_bad_zip(tmp_path, monkeypatch):
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(b"not a zip"))
    with pytest.raises(ValidationError, match="valid ZIP"):
        fetch_uci_har(tmp_path / "har")


def test_load_rejects_missing_and_subject_overlap(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="missing or incomplete"):
        load_uci_har(tmp_path)

    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    root = fetch_uci_har(tmp_path / "har")
    (root / "test/subject.txt").write_text("1\n26\n27\n28\n29\n30\n31\n32\n33\n34\n35\n36\n37\n38\n39\n40\n41\n42\n43\n44\n45\n46\n47\n48\n")
    with pytest.raises(ValidationError, match="not disjoint"):
        load_uci_har(root)


def test_read_dataset_provenance_rejects_non_object(tmp_path):
    (tmp_path / "provenance.json").write_text("[]")
    with pytest.raises(ValidationError, match="JSON object"):
        read_dataset_provenance(tmp_path)


def test_feature_shift_proxy_is_deterministic_and_detectable():
    x = np.arange(400, dtype=np.float32).reshape(20, 20) / 400
    a = feature_shift_proxy(x, seed=5)
    b = feature_shift_proxy(x, seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, x)
    assert np.isfinite(a).all()
    with pytest.raises(ValidationError):
        feature_shift_proxy(np.array([1.0, 2.0], dtype=np.float32))


def test_showcase_offline_smoke_generates_evidence(tmp_path):
    result = run_showcase(
        dataset="digits",
        output_dir=tmp_path / "out",
        benchmark_iterations=3,
    )
    assert result["deployment"]["baseline_result"] == "accepted"
    assert "rejected by health gate" in result["deployment"]["bad_signed_release"]
    assert result["deployment"]["active_after_bad_candidate"] == "1.1.0"
    assert result["deployment"]["active_after_rollback"] == "1.0.0"
    assert "anti-rollback" in result["deployment"]["old_release_replay"]
    assert result["drift"]["shifted_mean_psi"] > result["drift"]["clean_mean_psi"]
    assert (tmp_path / "out/showcase.json").is_file()
    assert (tmp_path / "out/showcase-summary.svg").is_file()
    report = (tmp_path / "out/README.md").read_text()
    assert "signature validity is necessary but insufficient" in report


def test_showcase_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValidationError, match="requires --data-dir"):
        run_showcase(dataset="uci-har", output_dir=tmp_path, benchmark_iterations=3)
    with pytest.raises(ValidationError, match="unsupported showcase dataset"):
        run_showcase(dataset="other", output_dir=tmp_path, benchmark_iterations=3)
    with pytest.raises(ValidationError, match="benchmark_iterations"):
        run_showcase(dataset="digits", output_dir=tmp_path, benchmark_iterations=2)


def test_showcase_uci_har_fixture_exercises_real_data_path(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    root = fetch_uci_har(tmp_path / "har")
    result = run_showcase(
        dataset="uci-har",
        data_dir=root,
        output_dir=tmp_path / "showcase",
        benchmark_iterations=3,
    )
    assert result["dataset"]["features"] == 561
    assert result["dataset"]["classes"] == 6
    assert result["dataset"]["subject_split"]["train"] == list(range(1, 25))
    assert result["dataset"]["provenance"]["license"] == "CC BY 4.0"
    assert result["deployment"]["active_after_bad_candidate"] == "1.1.0"


def test_load_uci_har_rejects_wrong_schema_and_bad_provenance(tmp_path, monkeypatch):
    payload = _zip_bytes()
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(payload))
    root = fetch_uci_har(tmp_path / "har")

    original = (root / "test/y.txt").read_text()
    (root / "test/y.txt").write_text(original.replace("6\n", "7\n", 1))
    with pytest.raises(ValidationError, match="activity labels"):
        load_uci_har(root)
    (root / "test/y.txt").write_text(original)

    (root / "features.txt").write_text("1 only-one\n")
    with pytest.raises(ValidationError, match="metadata files"):
        load_uci_har(root)

    (root / "provenance.json").write_text("{bad json")
    with pytest.raises(ValidationError, match="provenance"):
        read_dataset_provenance(root)


def test_fetch_rejects_symlink_and_duplicate_members(tmp_path, monkeypatch):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in _har_files().items():
            archive.writestr(name, value)
        info = zipfile.ZipInfo("symlink-entry")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        archive.writestr(info, b"target")
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(out.getvalue()))
    with pytest.raises(ValidationError, match="symlink"):
        fetch_uci_har(tmp_path / "symlink")

    dup = io.BytesIO()
    with zipfile.ZipFile(dup, "w") as archive:
        for name, value in _har_files().items():
            archive.writestr(name, value)
        with pytest.warns(UserWarning):
            archive.writestr("UCI HAR Dataset/features.txt", b"duplicate")
    monkeypatch.setattr("edgeai.datasets.urlopen", lambda *a, **k: _Response(dup.getvalue()))
    with pytest.raises(ValidationError, match="duplicate"):
        fetch_uci_har(tmp_path / "duplicate")
