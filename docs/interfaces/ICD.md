# Interface Control Document

## Artifact manifest

Schema version: `1`.

Required fields:

| Field | Meaning |
|---|---|
| `release_sequence` | unique monotonic anti-rollback integer, exact JSON integer in range `0..2^63-1` |
| `version` | semantic version |
| `model_kind` | runtime loader selector |
| `runtime_abi` | artifact/runtime compatibility identifier |
| `model_sha256` | lower-case SHA-256 hexadecimal digest |
| `model_bytes` | exact byte count |
| `metrics` | accuracy, ECE, p95 latency and RSS measured during release creation |
| `created_utc` | ISO timestamp string |
| `metadata` | canonical-JSON operator/model metadata, maximum 16 KiB |

Manifest JSON is parsed strictly: duplicate keys, unknown fields, missing fields, boolean-as-number values, non-finite constants, and numeric/string coercion are rejected. Published registry releases may not reuse a `release_sequence` for another version.

## Model input

Validated reference runtimes consume an `N x D` finite `float32` array. `InferenceEngine` enforces maximum batch and feature counts before invoking the model.

## Prediction API

`POST /v1/predict`

```json
{"inputs": [[0.0, 1.0]]}
```

Response:

```json
{
  "predictions": [1],
  "logits": [[-0.2, 1.1]],
  "version": "1.0.2"
}
```

All output numbers must be finite JSON numbers. The HTTP boundary applies a configurable request-body byte limit before JSON/schema/model allocation; the reference default is 8 MiB.

## Operational readiness

`GET /readyz` returns service-unavailable while no model is active. Production calls require `X-API-Key`.

## Telemetry delivery record

New durable spool entries receive a persistent 128-bit lowercase hexadecimal `event_id`. The backward-compatible `flush()` interface passes only the original event. `flush_records()` exposes:

```json
{
  "event_id": "0123456789abcdef0123456789abcdef",
  "event": {"kind": "measurement", "value": 7},
  "persistent_id": true
}
```

A sink that can create externally visible side effects should deduplicate by `event_id`. Legacy spool records remain readable and are surfaced with `event_id: null` and `persistent_id: false`.
