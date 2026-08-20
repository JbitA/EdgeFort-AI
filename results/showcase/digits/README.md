# Reproducible showcase results

![Showcase result summary](showcase-summary.svg)

## Dataset

**scikit-learn digits (offline smoke track)**

- train rows: 1,257
- test rows: 540
- features: 64
- classes: 10
- provenance: `showcase.json` → `dataset.provenance`

## Model trade-offs

| Runtime | Clean accuracy | Shifted accuracy | ECE | Artifact | p95 / batch 32* |
|---|---:|---:|---:|---:|---:|
| Float32 | 96.67% | 96.48% | 0.0702 | 3,013 B | 0.0108 ms |
| Weight int8 | 96.67% | 96.48% | 0.0703 | 1,480 B | 0.0113 ms |
| Dynamic int8 | 96.67% | 96.48% | 0.0704 | 1,481 B | 0.0517 ms |

*Host/reference benchmark only; not target-device latency.*

## Drift signal

The controlled feature-space shift proxy moves mean PSI from **0.0100** on held-out clean data to **0.3074**. **10.9%** of features cross the configured PSI threshold.

This is intentionally a feature-space proxy, not a claim that engineered UCI HAR features reproduce a particular physical sensor fault.

## Deployment experiment

- baseline signed release: **1.0.0** → accepted
- quantized signed release: **1.1.0** → accepted
- cryptographically valid but intentionally degraded release: **1.2.0** → **rejected by health gate (accuracy below minimum)**
- active release after rejection: **1.1.0**
- operator rollback: **succeeded** → active **1.0.0**
- ordinary redeploy of an older release: **rejected by anti-rollback policy (release sequence is not newer than trusted floor)**
- trusted anti-rollback floor after rollback: **2**

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
