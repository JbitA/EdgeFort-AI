# ADR 0001 — Asymmetric release signing

## Decision
Use Ed25519 detached signatures over canonical manifests instead of a shared HMAC secret.

## Rationale
Edge devices need only public verification material. Compromise of a device therefore does not automatically disclose the ability to sign arbitrary new releases. Ed25519 also provides compact keys/signatures and deterministic verification semantics.

## Consequence
Private-key custody becomes an external release-engineering responsibility and must not occur on ordinary edge devices.
