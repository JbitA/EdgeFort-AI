# Requirements Traceability

| Requirement | Implementation | Principal tests/evidence |
|---|---|---|
| REQ-001 | `artifact.verify_artifact`, `registry.inspect` | `test_tampered_model_rejected`, `test_registry_tamper_detected` |
| REQ-002 | `TrustedKeyring` | wrong-key, revoked-key, key-rotation tests |
| REQ-003 | `Registry.add` | `test_registry_immutable`, symlink/version-mismatch tests |
| REQ-004 | `ABDeployer.deploy` | `test_anti_rollback_floor_survives_rollback` |
| REQ-005 | `health.benchmark/enforce` | health-policy and reference lifecycle tests |
| REQ-006 | `ABDeployer.deploy` | failed-health test |
| REQ-007 | `atomic_write`, A/B transaction | deployment-transaction fault injection |
| REQ-008 | `ABDeployer.rollback` | two-slot rollback test |
| REQ-009/010 | `InferenceEngine.predict` | engine bounds, ragged-input, and API tests |
| REQ-011 | `BoundedInferenceExecutor`, `/v1/predict` | overflow/deadline/drop-policy and async service-integration tests |
| REQ-012/013 | `OfflineSpool` | durability/corrupt-line/concurrency/order tests |
| REQ-014 | `Metrics`, `/metrics` | engine/API tests |
| REQ-015 | `create_app` | production-app-requires-key and production-docs tests |
| REQ-016 | quantization/evaluation | quantization unit and multi-seed system tests |
| REQ-017 | `cpp/` | CTest + ASan/UBSan qualification, overflow tests |
| REQ-018 | coverage configuration | qualification report |
| REQ-019 | `PSIProfile` | drift same/shifted/empty distribution tests |
| REQ-020 | docs + qualification JSON | production readiness + adversarial review |
| REQ-021 | `Registry.verify_directory`, `ABDeployer._verified_slot_runtime` | active-slot tamper test |
| REQ-022 | `health.expected_runtime_abi/load_runtime` | signed-runtime-ABI enforcement test |
| REQ-023 | `ABDeployer._parse_state` | state-corruption/path-traversal/missing-state tests |
| REQ-024 | `RequestBodyLimitMiddleware` | API pre-schema body-limit test |
| REQ-025 | `RollbackAnchor`, `ABDeployer._validate_state_anchor` | external-floor rewind/ahead/read-failure tests |
| REQ-026 | pending anchor journal + `ABDeployer._recover_anchor_commit` | simulated crash after anchor advance before state write; stale/pre-anchor journal tests |
| REQ-027 | `Registry._assert_release_sequence_available` | duplicate release-sequence and corrupt-registry fail-closed tests |
| REQ-028 | `manifest_from_dict`, `strict_json_loads` | coercion/unknown/duplicate manifest tests |
| REQ-029 | `RuntimeConfig`, `_UniqueKeySafeLoader` | boolean numeric, duplicate YAML, missing/unknown config tests |
| REQ-030 | `OfflineSpool.append/flush_records` | retry-stable ID, idempotent enqueue and legacy-record tests |
| REQ-031 | `BoundedInferenceExecutor.close`, FastAPI lifespan | hung-worker bounded-shutdown and timeout-validation tests |
| REQ-032 | `InferenceEngine._validated_logits`, `_worker_validate_logits` | malformed/rank/batch/class/element/non-finite backend-output tests |
| REQ-033 | `ExecutionMode.PROCESS`, `_ProcessInferenceRunner` | hung-child execution-timeout and child-crash recovery tests |
| REQ-034 | `ABDeployer.active_artifact_snapshot`, `InferenceEngine.prepare_isolated`, child `load_artifact` protocol | signed-registry process-mode test that forbids parent runtime load; IPC protocol tests |
| REQ-035 | `RequestBodyLimitMiddleware` | duplicate/conflicting content-length/transfer-encoding and declared-length mismatch tests |
| REQ-036 | `_apply_process_memory_limit`, process startup handshake | resource-limit helper/startup-error tests and process-mode readiness with configured ceiling |
| REQ-037 | `scripts/generate_sbom.py`, package CI reproducibility step | deterministic regeneration hash check and double wheel byte comparison |
