# Showcase experiment design

## Question

The experiment is designed to answer a systems question, not a leaderboard question:

> **Can an edge device prove what model it is running, reject an authentic-but-unsafe release, preserve rollback guarantees, detect meaningful input shift, and expose the accuracy/footprint trade-off of quantization?**

The primary workload is UCI Human Activity Recognition Using Smartphones. The model is deliberately a compact linear classifier so the release-control and runtime behavior remain inspectable.

## Experiment A — model footprint vs. quality

Train one float32 linear classifier and derive two quantized artifacts:

1. float32;
2. per-output-channel weight-only int8;
3. dynamic int8 activation + int8 weights.

For each artifact, record:

- held-out accuracy;
- expected calibration error;
- p95 batch-32 host latency;
- peak host RSS during the reference benchmark;
- serialized artifact bytes;
- accuracy under the deterministic shift proxy.

Latency and RSS are explicitly **host-reference measurements**. They are not presented as Jetson/mobile/WCET evidence.

## Experiment B — distribution-shift evidence

Fit a PSI reference profile on the training split and compare:

- the untouched held-out split;
- a deterministic feature-space shift proxy.

Because the public UCI table contains engineered window features rather than the raw sensor stream, the proxy is intentionally modest in its claim: 10% of feature columns receive bounded gain+bias and 5% are dropped to zero. It demonstrates the drift monitor and quality response; it does not claim to emulate a specific sensor defect.

Record:

- mean PSI;
- maximum PSI;
- fraction of features above the configured PSI threshold;
- clean vs. shifted accuracy.

## Experiment C — authentic does not mean safe

Create three genuinely signed releases:

1. `1.0.0` — float baseline;
2. `1.1.0` — weight-int8 candidate;
3. `1.2.0` — intentionally degraded zero-weight model.

The third release has a **valid signature and valid manifest**. It must still fail the deployment health gate on semantic quality.

The experiment proves:

- baseline deployment succeeds;
- a healthy quantized release can promote;
- a cryptographically authentic but low-quality release is rejected;
- the active version is unchanged after rejection;
- operator rollback selects the previous slot;
- the trusted release-sequence floor does not decrease;
- an ordinary attempt to redeploy an older release is rejected by anti-rollback policy.

This is the core showcase claim: **supply-chain authenticity and runtime safety are separate controls.**

## Outputs

Each run writes:

```text
results/showcase/<dataset>/
  showcase.json          machine-readable evidence
  showcase-summary.svg   GitHub-renderable visual summary
  README.md              human-readable experiment report
```

The JSON file is the source of truth for the generated report and SVG.

## Reproduction

Primary public-data run:

```bash
make showcase-real
```

Offline smoke run:

```bash
make showcase
```

GitHub also contains a manually triggered `showcase-real-data` workflow that acquires UCI HAR from the official source, runs the experiment, and uploads the resulting evidence artifact.
