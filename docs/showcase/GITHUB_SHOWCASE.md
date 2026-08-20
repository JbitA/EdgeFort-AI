# GitHub showcase strategy

The repository should be read as an **edge AI deployment systems project**, not as a model-accuracy project.

## What a reviewer should understand in 60 seconds

1. The project solves deployment problems that `model.predict()` does not: authenticity, rollback, health gates, bounded admission, worker crashes, offline telemetry, and drift.
2. The primary experiment uses openly licensed smartphone IMU data rather than relying only on the bundled digits fixture.
3. A signed artifact can still be unsafe; the system demonstrates rejection of an authentic-but-degraded release.
4. Quantization is presented as a measured trade-off. Negative performance results are retained.
5. Claims are bounded: host latency is not target-device latency, process hardening is not a complete OS sandbox, and software rollback anchors are not TPM proof.

## Recommended repository front page

Keep the README's first screen focused on:

- one-sentence value proposition;
- the architecture diagram;
- the four headline experiment outcomes;
- a three-command reproduction path;
- links to threat model, adversarial review, and qualification evidence.

Detailed implementation notes belong in `docs/` so depth is available without burying the value proposition.
