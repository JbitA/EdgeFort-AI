from __future__ import annotations

from dataclasses import asdict
import html
import json
from pathlib import Path
import tempfile

import numpy as np

from .artifact import TrustedKeyring, build_manifest, generate_keypair, public_key_id, sign_manifest
from .data import load as load_digits
from .datasets import HARSplit, load_uci_har, read_dataset_provenance
from .deploy import ABDeployer
from .drift import PSIProfile
from .errors import HealthGateError, RollbackRejected, ValidationError
from .health import HealthPolicy, benchmark
from .metrics import accuracy, expected_calibration_error
from .model import LinearModel, train
from .quantization import quantize_dynamic, quantize_weights
from .registry import Registry
from .types import ModelMetrics


_SHOWCASE_CREATED_UTC = "2026-08-20T00:00:00+00:00"


def _probabilities(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def feature_shift_proxy(x: np.ndarray, *, seed: int = 2026) -> np.ndarray:
    """Deterministic feature-space proxy for sensor/calibration shift.

    UCI HAR's public classification table contains engineered features rather than the raw
    sensor stream. This intentionally does not claim physical sensor simulation: it applies
    a bounded gain/bias change to 10% of feature columns plus 5% column dropout so drift
    detection and quality degradation can be demonstrated reproducibly.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2 or not np.isfinite(arr).all():
        raise ValidationError("invalid showcase feature matrix")
    rng = np.random.default_rng(seed)
    n_features = arr.shape[1]
    shifted = arr.copy()
    biased_n = max(1, int(round(n_features * 0.10)))
    dropped_n = max(1, int(round(n_features * 0.05)))
    order = rng.permutation(n_features)
    biased = order[:biased_n]
    dropped = order[biased_n : biased_n + dropped_n]
    shifted[:, biased] = shifted[:, biased] * np.float32(1.15) + np.float32(0.08)
    shifted[:, dropped] = 0.0
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return np.clip(shifted, lo, hi).astype(np.float32)


def _dataset_payload(dataset: str, data_dir: str | Path | None, seed: int):
    if dataset == "digits":
        x_train, x_test, y_train, y_test = load_digits(seed)
        return {
            "name": "scikit-learn digits (offline smoke track)",
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "subjects": None,
            "provenance": {
                "dataset": "sklearn-digits",
                "license": "offline smoke fixture; see scikit-learn dataset documentation",
                "note": "Offline smoke dataset; use uci-har for the public edge-sensor showcase.",
            },
        }
    if dataset == "uci-har":
        if data_dir is None:
            raise ValidationError("uci-har showcase requires --data-dir")
        split: HARSplit = load_uci_har(data_dir)
        return {
            "name": "UCI Human Activity Recognition Using Smartphones",
            "x_train": split.x_train,
            "x_test": split.x_test,
            "y_train": split.y_train,
            "y_test": split.y_test,
            "subjects": {
                "train": sorted({int(v) for v in split.train_subjects}),
                "test": sorted({int(v) for v in split.test_subjects}),
            },
            "provenance": read_dataset_provenance(data_dir),
        }
    raise ValidationError(f"unsupported showcase dataset {dataset!r}")


def _metrics(runtime, x_test, y_test, shifted, *, benchmark_iterations: int):
    bm = benchmark(runtime, x_test, y_test, iterations=benchmark_iterations)
    probs = _probabilities(runtime.logits(x_test))
    return {
        "clean_accuracy": accuracy(y_test, runtime.predict(x_test)),
        "shifted_accuracy": accuracy(y_test, runtime.predict(shifted)),
        "ece": expected_calibration_error(y_test, probs),
        "p95_latency_ms_batch32": bm.p95_latency_ms,
        "max_rss_mb": bm.max_rss_mb,
    }


def _write_showcase_svg(path: Path, results: dict[str, object]) -> None:
    models = results["model_tradeoffs"]
    names = ["float32", "weight int8", "dynamic int8"]
    keys = ["float32", "int8_weight", "int8_dynamic"]
    acc = [float(models[key]["clean_accuracy"]) for key in keys]
    shifted = [float(models[key]["shifted_accuracy"]) for key in keys]
    sizes = [int(models[key]["artifact_bytes"]) for key in keys]
    max_size = max(sizes)
    width, height = 920, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" rx="18" fill="#0b1020"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.h{font-weight:700;font-size:22px;fill:#f8fafc}.s{font-size:14px;fill:#cbd5e1}.n{font-size:13px;fill:#e2e8f0}.v{font-weight:700;font-size:13px;fill:#f8fafc}</style>',
        '<text class="h" x="32" y="42">Edge AI showcase — quality under shift + artifact footprint</text>',
        f'<text class="s" x="32" y="67">{html.escape(str(results["dataset"]["name"]))}</text>',
        '<text class="s" x="32" y="94">Accuracy bars: clean / shifted feature-space proxy. Footprint shown at right.</text>',
    ]
    y_positions = [150, 240, 330]
    for name, clean, shift, size, y in zip(names, acc, shifted, sizes, y_positions):
        lines.append(f'<text class="n" x="32" y="{y}">{name}</text>')
        base_x, bar_w = 175, 470
        lines.append(f'<rect x="{base_x}" y="{y-22}" width="{bar_w}" height="18" rx="4" fill="#1e293b"/>')
        lines.append(f'<rect x="{base_x}" y="{y-22}" width="{bar_w*clean:.1f}" height="8" rx="4" fill="#22c55e"/>')
        lines.append(f'<rect x="{base_x}" y="{y-12}" width="{bar_w*shift:.1f}" height="8" rx="4" fill="#f59e0b"/>')
        lines.append(f'<text class="v" x="{base_x+bar_w+14}" y="{y-10}">{clean*100:.1f}% / {shift*100:.1f}%</text>')
        size_w = 120 * (size / max_size)
        lines.append(f'<rect x="760" y="{y-22}" width="{size_w:.1f}" height="18" rx="4" fill="#60a5fa"/>')
        lines.append(f'<text class="v" x="760" y="{y+14}">{size:,} B</text>')
    drift = results["drift"]
    deployment = results["deployment"]
    lines.extend([
        f'<text class="s" x="32" y="392">Shift PSI mean {float(drift["shifted_mean_psi"]):.3f} · features over threshold {float(drift["shifted_fraction_over_threshold"])*100:.1f}% · bad signed release: {html.escape(str(deployment["bad_signed_release"]))}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, results: dict[str, object]) -> None:
    d = results["dataset"]
    drift = results["drift"]
    deployment = results["deployment"]
    model_rows = []
    for key, label in (("float32", "Float32"), ("int8_weight", "Weight int8"), ("int8_dynamic", "Dynamic int8")):
        row = results["model_tradeoffs"][key]
        model_rows.append(
            f'| {label} | {row["clean_accuracy"]*100:.2f}% | {row["shifted_accuracy"]*100:.2f}% | '
            f'{row["ece"]:.4f} | {row["artifact_bytes"]:,} B | {row["p95_latency_ms_batch32"]:.4f} ms |'
        )
    text = f"""# Reproducible showcase results

![Showcase result summary](showcase-summary.svg)

## Dataset

**{d['name']}**

- train rows: {d['train_rows']:,}
- test rows: {d['test_rows']:,}
- features: {d['features']:,}
- classes: {d['classes']}
- provenance: `showcase.json` → `dataset.provenance`

## Model trade-offs

| Runtime | Clean accuracy | Shifted accuracy | ECE | Artifact | p95 / batch 32* |
|---|---:|---:|---:|---:|---:|
{chr(10).join(model_rows)}

*Host/reference benchmark only; not target-device latency.*

## Drift signal

The controlled feature-space shift proxy moves mean PSI from **{drift['clean_mean_psi']:.4f}** on held-out clean data to **{drift['shifted_mean_psi']:.4f}**. **{drift['shifted_fraction_over_threshold']*100:.1f}%** of features cross the configured PSI threshold.

This is intentionally a feature-space proxy, not a claim that engineered UCI HAR features reproduce a particular physical sensor fault.

## Deployment experiment

- baseline signed release: **{deployment['baseline_release']}** → {deployment['baseline_result']}
- quantized signed release: **{deployment['quantized_release']}** → {deployment['quantized_result']}
- cryptographically valid but intentionally degraded release: **{deployment['bad_release']}** → **{deployment['bad_signed_release']}**
- active release after rejection: **{deployment['active_after_bad_candidate']}**
- operator rollback: **{deployment['rollback_result']}** → active **{deployment['active_after_rollback']}**
- ordinary redeploy of an older release: **{deployment['old_release_replay']}**
- trusted anti-rollback floor after rollback: **{deployment['highest_release_sequence']}**

The point of this experiment is that **signature validity is necessary but insufficient**. A release can be authentic and still be operationally unsafe; the health gate rejects it without replacing the active model.

## Reproduce

```bash
edgeai dataset fetch uci-har --destination data/uci-har
edgeai showcase --dataset uci-har --data-dir data/uci-har --output-dir results/showcase/uci-har
```

For an offline smoke run with no download:

```bash
edgeai showcase --dataset digits --output-dir results/showcase/digits
```
"""
    path.write_text(text, encoding="utf-8")


def run_showcase(
    *,
    dataset: str = "digits",
    data_dir: str | Path | None = None,
    output_dir: str | Path = "results/showcase",
    seed: int = 42,
    benchmark_iterations: int = 20,
) -> dict[str, object]:
    if benchmark_iterations < 3 or benchmark_iterations > 500:
        raise ValidationError("benchmark_iterations must be between 3 and 500")
    payload = _dataset_payload(dataset, data_dir, seed)
    x_train = np.asarray(payload["x_train"], dtype=np.float32)
    x_test = np.asarray(payload["x_test"], dtype=np.float32)
    y_train = np.asarray(payload["y_train"], dtype=np.int64)
    y_test = np.asarray(payload["y_test"], dtype=np.int64)

    model = train(x_train, y_train, seed)
    weight = quantize_weights(model)
    dynamic = quantize_dynamic(model)
    shifted = feature_shift_proxy(x_test, seed=seed + 1000)

    profile = PSIProfile.fit(x_train)
    clean_drift = profile.score(x_test)
    shifted_drift = profile.score(shifted)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="edgeai-showcase-"))
    try:
        artifacts = {
            "float32": ("float32-linear", model, "float32.npz"),
            "int8_weight": ("int8-weight-linear", weight, "int8-weight.npz"),
            "int8_dynamic": ("int8-dynamic-linear", dynamic, "int8-dynamic.npz"),
        }
        model_tradeoffs: dict[str, object] = {}
        saved: dict[str, Path] = {}
        for key, (_, runtime, filename) in artifacts.items():
            path = work / filename
            runtime.save(path)
            saved[key] = path
            row = _metrics(runtime, x_test, y_test, shifted, benchmark_iterations=benchmark_iterations)
            row["artifact_bytes"] = path.stat().st_size
            model_tradeoffs[key] = row

        priv, pub = generate_keypair()
        key_id = public_key_id(pub)
        registry = Registry(work / "registry", TrustedKeyring({key_id: pub}))
        baseline_accuracy = float(model_tradeoffs["float32"]["clean_accuracy"])
        policy = HealthPolicy(
            min_accuracy=max(0.50, baseline_accuracy - 0.05),
            max_accuracy_drop=0.025,
            max_ece=0.50,
            max_p95_latency_ms=100.0,
            max_rss_mb=4096.0,
        )
        deployer = ABDeployer(work / "deploy", registry, policy)

        def add_release(version, seq, kind, runtime, path, label):
            row = model_tradeoffs[label]
            metrics = ModelMetrics(
                accuracy=float(row["clean_accuracy"]),
                ece=float(row["ece"]),
                p95_latency_ms=float(row["p95_latency_ms_batch32"]),
                max_rss_mb=float(row["max_rss_mb"]),
            )
            manifest = build_manifest(
                path,
                version,
                seq,
                kind,
                runtime.ABI,
                metrics,
                {
                    "showcase_dataset": dataset,
                    "experiment": "github-value-showcase-v1",
                    "seed": seed,
                },
                created_utc=_SHOWCASE_CREATED_UTC,
            )
            registry.add(path, manifest, sign_manifest(manifest, priv), key_id)
            return manifest

        add_release("1.0.0", 1, "float32-linear", model, saved["float32"], "float32")
        deployer.deploy("1.0.0", x_test, y_test)
        add_release("1.1.0", 2, "int8-weight-linear", weight, saved["int8_weight"], "int8_weight")
        deployer.deploy("1.1.0", x_test, y_test)

        bad = LinearModel(np.zeros_like(model.W), np.zeros_like(model.b))
        bad_path = work / "bad-but-signed.npz"
        bad.save(bad_path)
        bad_bm = benchmark(bad, x_test, y_test, iterations=max(3, benchmark_iterations // 2))
        bad_probs = _probabilities(bad.logits(x_test))
        bad_metrics = ModelMetrics(
            accuracy=accuracy(y_test, bad.predict(x_test)),
            ece=expected_calibration_error(y_test, bad_probs),
            p95_latency_ms=bad_bm.p95_latency_ms,
            max_rss_mb=bad_bm.max_rss_mb,
        )
        bad_manifest = build_manifest(
            bad_path,
            "1.2.0",
            3,
            "float32-linear",
            bad.ABI,
            bad_metrics,
            {"showcase_dataset": dataset, "experiment": "signed-but-bad-release"},
            created_utc=_SHOWCASE_CREATED_UTC,
        )
        registry.add(bad_path, bad_manifest, sign_manifest(bad_manifest, priv), key_id)
        bad_result = "unexpectedly accepted"
        try:
            deployer.deploy("1.2.0", x_test, y_test)
        except HealthGateError as exc:
            bad_result = f"rejected by health gate ({exc})"

        active_after_bad = deployer.active_version()
        rollback_ok = deployer.rollback()
        active_after_rollback = deployer.active_version()
        replay_result = "unexpectedly accepted"
        try:
            deployer.deploy("1.0.0", x_test, y_test)
        except RollbackRejected as exc:
            replay_result = f"rejected by anti-rollback policy ({exc})"

        state = deployer.state()
        results = {
            "schema_version": 1,
            "experiment": "github-value-showcase-v1",
            "dataset": {
                "name": payload["name"],
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "features": int(x_train.shape[1]),
                "classes": int(len(np.unique(np.concatenate([y_train, y_test])))),
                "subject_split": payload["subjects"],
                "provenance": payload["provenance"],
            },
            "model_tradeoffs": model_tradeoffs,
            "drift": {
                "threshold": 0.2,
                "clean_mean_psi": clean_drift.mean_psi,
                "clean_max_psi": clean_drift.max_psi,
                "clean_fraction_over_threshold": clean_drift.fraction_over_threshold,
                "shifted_mean_psi": shifted_drift.mean_psi,
                "shifted_max_psi": shifted_drift.max_psi,
                "shifted_fraction_over_threshold": shifted_drift.fraction_over_threshold,
                "shift_proxy": "10% feature gain+bias plus 5% feature dropout; clipped to observed range",
            },
            "deployment": {
                "baseline_release": "1.0.0",
                "baseline_result": "accepted",
                "quantized_release": "1.1.0",
                "quantized_result": "accepted",
                "bad_release": "1.2.0",
                "bad_signed_release": bad_result,
                "active_after_bad_candidate": active_after_bad,
                "rollback_result": "succeeded" if rollback_ok else "failed",
                "active_after_rollback": active_after_rollback,
                "old_release_replay": replay_result,
                "highest_release_sequence": state.highest_release_sequence,
            },
            "limitations": [
                "Latency/RSS values are host-reference measurements, not target-device qualification.",
                "The drift scenario is a deterministic feature-space proxy, not a physical sensor simulator.",
                "The linear classifier is intentionally small; the experiment showcases lifecycle controls.",
            ],
        }

        json_path = output / "showcase.json"
        json_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_showcase_svg(output / "showcase-summary.svg", results)
        _write_markdown(output / "README.md", results)
        return results
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
