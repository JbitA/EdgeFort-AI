# Contributing

Changes to trust, deployment, quantization or persistence code require regression tests. Do not weaken security, health, performance or coverage gates to make a change pass.

Before proposing a change run:

```bash
make test
make coverage
make cpp
make cpp-sanitize
make evaluate
make research
```

Any benchmark claim must identify the runtime/backend and dataset used. Hardware acceleration claims require measurements on the named hardware/backend.
