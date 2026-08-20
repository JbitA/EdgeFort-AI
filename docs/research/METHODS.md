# Research Methods

## Question

What do per-output-channel weight quantization and per-request dynamic activation quantization change in a small edge workload when accuracy, corruption robustness, artifact size and CPU latency are measured together?

## Experimental controls

- same public digits dataset;
- stratified 70/30 split;
- fixed model family and preprocessing;
- deterministic seeds;
- ten repeated train/test splits for robustness evidence;
- Gaussian pixel corruption at two amplitudes;
- same runtime process for timing.

## Hypotheses

H1: per-channel weight int8 will reduce model storage materially with negligible accuracy loss.

H2: dynamic activation int8 will preserve accuracy but will not necessarily reduce CPU latency in a NumPy implementation because quantization overhead and lack of accelerator kernels can dominate.

H3: quantized variants should not show materially worse corruption robustness than the float reference on this workload.

## Interpretation rule

Artifact-size reduction may be claimed directly. Runtime acceleration may only be claimed on a backend where end-to-end latency is measured to improve. The NumPy dynamic-int8 path is a correctness/reference implementation, not an accelerator benchmark.
