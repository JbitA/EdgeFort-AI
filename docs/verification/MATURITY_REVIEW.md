# Software maturity review

## Positioning

The repository should be presented as an **edge AI deployment systems** project. The deliberately compact reference classifier makes lifecycle behavior inspectable; the engineering value is in signed release identity, health-gated A/B activation, anti-rollback, isolation, bounded service behavior, telemetry/drift and evidence generation.

## Current maturity

**Strong reference implementation / serious showcase; not yet a complete production device agent.**

The current tree has 301 passing Python tests, 90.50% combined statement+branch coverage, passing native Release + ASan/UBSan tests, an offline-built wheel, and a public-data showcase path using openly licensed smartphone sensor data.

## What is unusually strong for a showcase repository

- Authenticity and semantic release health are explicitly separate controls.
- Verification is repeated where artifacts are consumed, not assumed from registry insertion.
- A/B state changes and external anti-rollback transitions have explicit crash semantics.
- Backend output and child-process IPC are treated as untrusted boundaries.
- Hard inference deadlines kill a process rather than attempting unsafe Python thread cancellation.
- Crash-loop restarts are globally budgeted rather than multiplied by worker count.
- Public dataset acquisition captures provenance and validates archives rather than committing raw data.
- Experiments generate machine-readable evidence plus a reviewer-friendly SVG/Markdown summary.
- Negative quantization latency findings are retained.
- Qualification limits are stated next to results instead of hidden in a footnote.

## Principal remaining gaps

- real ONNX Runtime execution and model-graph policy qualification;
- same-UID backend process still has too much OS authority for a hostile-native-code threat model;
- no real TPM/secure-element rollback anchor;
- no device mTLS/identity lifecycle;
- no target web-server/kernel ingress profile;
- no named edge-device thermal/power/accelerator evidence;
- release provenance/attestation remains partial;
- UCI HAR real-data results need one retained network-enabled run.

These are now more valuable than adding another quantization mode, dashboard, accelerator wrapper, or programming language.
