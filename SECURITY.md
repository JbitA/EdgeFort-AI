# Security Policy

Report suspected vulnerabilities privately to the repository maintainer rather than opening a public exploit issue.

Never commit private signing keys, production API credentials, device credentials, or customer data. The repository's tests generate ephemeral signing keys at runtime.

Production deployment must treat this repository's API-key mechanism as a minimal service boundary. Distributed deployments should add transport security, device/service identity, certificate rotation and platform hardening appropriate to the threat model.

## Rollback-anchor boundary

The `RollbackAnchor` protocol is designed for TPM/secure-element/boot-chain integration. The included memory and file implementations are test doubles and do not establish whole-filesystem rollback resistance. Treat unexplained external-anchor/local-state disagreement as an integrity incident; do not reset the anchor to restore availability.
