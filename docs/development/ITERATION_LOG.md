# Engineering Iteration Record

## Native build definition

The first sanitizer qualification exposed invalid CMake command syntax. The build definition was corrected and both Release and sanitizer builds were rerun.

## Native test oracle

The first C++ test used an incorrectly calculated expected int8 output. The oracle was corrected after independently recomputing the integer dot product and scale application; the kernel itself was unchanged.

## Evaluation workspace ownership

Repeated evaluation initially collided with immutable versions in the previous evaluation registry. Evaluation workspaces now carry an ownership marker. A marked workspace can be reset for reproducible reruns; an unmarked nonempty directory is rejected rather than deleted. Regression tests cover both rerun and refusal behavior.

## Deployment transaction fault

A state-write failure is injected after candidate slot replacement. The deployer restores the inactive-slot backup and the active version remains unchanged. This guards the release transaction against partial state commit.

## Quantization interpretation

Dynamic int8 produced no CPU latency advantage in the validated NumPy runtime. Documentation and research output therefore distinguish storage/accuracy benefits from hardware acceleration claims.
