# ADR 0002 — A/B slots with independent anti-rollback floor

## Decision
Maintain active and previous model slots while persisting the highest trusted release sequence separately.

## Rationale
Operational rollback is necessary, but lowering the trusted sequence during rollback would permit replay of old signed releases. Slot selection and release trust are therefore separate state variables.
