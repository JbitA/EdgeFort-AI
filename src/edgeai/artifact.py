from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .errors import ArtifactIntegrityError, SignatureError, ValidationError
from .types import ArtifactManifest, ModelMetrics
from .util import MAX_RELEASE_SEQUENCE, canonical_json_bytes, validate_semver

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_METADATA_BYTES = 16 * 1024
_MANIFEST_FIELDS = {
    "schema_version", "release_sequence", "version", "model_kind", "runtime_abi",
    "model_sha256", "model_bytes", "metrics", "created_utc", "metadata",
}
_METRIC_FIELDS = {"accuracy", "ece", "p95_latency_ms", "max_rss_mb"}


def sha256_stream(fileobj, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    fileobj.seek(0)
    while True:
        block = fileobj.read(chunk)
        if not block:
            break
        h.update(block)
    fileobj.seek(0)
    return h.hexdigest()


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    with Path(path).open("rb") as f:
        return sha256_stream(f, chunk)


def generate_keypair() -> tuple[bytes, bytes]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    private = priv.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, public


def public_key_id(public_key: bytes) -> str:
    # Compatibility-stable lookup identifier. Authentication still uses the full
    # Ed25519 public key and signature, not this display/selector prefix.
    return hashlib.sha256(public_key).hexdigest()[:16]


def _validate_created_utc(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as e:
        raise ValidationError("created_utc must be an ISO-8601 timestamp") from e
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError("created_utc must include a timezone")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValidationError("created_utc must be normalized to UTC")


def _validate_metadata(metadata: dict) -> None:
    try:
        encoded = canonical_json_bytes(metadata)
    except (TypeError, ValueError) as e:
        raise ValidationError("metadata must be finite canonical-JSON data") from e
    if len(encoded) > _MAX_METADATA_BYTES:
        raise ValidationError(f"metadata exceeds {_MAX_METADATA_BYTES} bytes")


def _validate_manifest(manifest: ArtifactManifest) -> None:
    if not isinstance(manifest, ArtifactManifest):
        raise ValidationError("manifest must be an ArtifactManifest")
    if not isinstance(manifest.metrics, ModelMetrics):
        raise ValidationError("metrics must be a ModelMetrics")
    if isinstance(manifest.schema_version, bool) or not isinstance(manifest.schema_version, int):
        raise ValidationError("schema_version must be an integer")
    if (
        isinstance(manifest.release_sequence, bool)
        or not isinstance(manifest.release_sequence, int)
        or not 0 <= manifest.release_sequence <= MAX_RELEASE_SEQUENCE
    ):
        raise ValidationError("release_sequence must be an integer in the supported range")
    if (
        isinstance(manifest.model_bytes, bool)
        or not isinstance(manifest.model_bytes, int)
        or manifest.model_bytes < 1
    ):
        raise ValidationError("model_bytes must be a positive integer")
    for name, value in (
        ("version", manifest.version),
        ("model_kind", manifest.model_kind),
        ("runtime_abi", manifest.runtime_abi),
        ("model_sha256", manifest.model_sha256),
        ("created_utc", manifest.created_utc),
    ):
        if not isinstance(value, str):
            raise ValidationError(f"{name} must be a string")
    for name, value in (
        ("accuracy", manifest.metrics.accuracy),
        ("ece", manifest.metrics.ece),
        ("p95_latency_ms", manifest.metrics.p95_latency_ms),
        ("max_rss_mb", manifest.metrics.max_rss_mb),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"metrics.{name} must be numeric")
    if not isinstance(manifest.metadata, dict):
        raise ValidationError("metadata must be a JSON object")
    try:
        validate_semver(manifest.version)
        manifest.metrics.validate()
    except (TypeError, ValueError) as e:
        raise ValidationError(str(e)) from e
    if manifest.schema_version != 1:
        raise ValidationError("unsupported manifest schema")
    if not manifest.model_kind or not manifest.runtime_abi:
        raise ValidationError("model_kind and runtime_abi are required")
    if not _SHA256_RE.fullmatch(manifest.model_sha256):
        raise ValidationError("invalid SHA-256")
    _validate_created_utc(manifest.created_utc)
    _validate_metadata(manifest.metadata)


def build_manifest(
    model_path: str | Path,
    version: str,
    release_sequence: int,
    model_kind: str,
    runtime_abi: str,
    metrics: ModelMetrics,
    metadata: dict | None = None,
    created_utc: str | None = None,
) -> ArtifactManifest:
    try:
        validate_semver(version)
        metrics.validate()
    except (TypeError, ValueError) as e:
        raise ValidationError(str(e)) from e
    if (
        isinstance(release_sequence, bool)
        or not isinstance(release_sequence, int)
        or not 0 <= release_sequence <= MAX_RELEASE_SEQUENCE
    ):
        raise ValidationError("release_sequence must be an integer in the supported range")
    if not model_kind or not runtime_abi:
        raise ValidationError("model_kind and runtime_abi are required")
    p = Path(model_path)
    if not p.is_file():
        raise ValidationError("model artifact does not exist")
    manifest = ArtifactManifest(
        schema_version=1,
        release_sequence=int(release_sequence),
        version=version,
        model_kind=model_kind,
        runtime_abi=runtime_abi,
        model_sha256=sha256_file(p),
        model_bytes=p.stat().st_size,
        metrics=metrics,
        created_utc=created_utc or datetime.now(timezone.utc).isoformat(),
        metadata=metadata or {},
    )
    _validate_manifest(manifest)
    return manifest


def manifest_from_dict(data: Mapping) -> ArtifactManifest:
    try:
        if not isinstance(data, Mapping):
            raise TypeError("manifest must be an object")
        fields = set(data)
        if fields != _MANIFEST_FIELDS:
            missing = sorted(_MANIFEST_FIELDS - fields)
            unknown = sorted(fields - _MANIFEST_FIELDS)
            raise ValueError(f"manifest fields mismatch; missing={missing}, unknown={unknown}")
        raw_metrics = data["metrics"]
        if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != _METRIC_FIELDS:
            raise ValueError("metrics fields mismatch")
        metrics = ModelMetrics(
            accuracy=raw_metrics["accuracy"],
            ece=raw_metrics["ece"],
            p95_latency_ms=raw_metrics["p95_latency_ms"],
            max_rss_mb=raw_metrics["max_rss_mb"],
        )
        manifest = ArtifactManifest(
            schema_version=data["schema_version"],
            release_sequence=data["release_sequence"],
            version=data["version"],
            model_kind=data["model_kind"],
            runtime_abi=data["runtime_abi"],
            model_sha256=data["model_sha256"],
            model_bytes=data["model_bytes"],
            metrics=metrics,
            created_utc=data["created_utc"],
            metadata=data["metadata"],
        )
    except Exception as e:
        raise ValidationError(f"invalid manifest: {e}") from e
    _validate_manifest(manifest)
    return manifest


def sign_manifest(manifest: ArtifactManifest, private_key: bytes) -> str:
    _validate_manifest(manifest)
    try:
        key = Ed25519PrivateKey.from_private_bytes(private_key)
    except Exception as e:
        raise ValidationError("invalid Ed25519 private key") from e
    return base64.b64encode(key.sign(canonical_json_bytes(manifest.to_dict()))).decode("ascii")


def verify_manifest(manifest: ArtifactManifest, signature_b64: str, public_key: bytes) -> None:
    _validate_manifest(manifest)
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, canonical_json_bytes(manifest.to_dict())
        )
    except Exception as e:
        raise SignatureError("manifest signature verification failed") from e


def verify_artifact(model_path: str | Path, manifest: ArtifactManifest) -> None:
    p = Path(model_path)
    if p.is_symlink() or not p.is_file():
        raise ArtifactIntegrityError("model artifact missing or not a regular file")
    if p.stat().st_size != manifest.model_bytes:
        raise ArtifactIntegrityError("model size mismatch")
    if sha256_file(p) != manifest.model_sha256:
        raise ArtifactIntegrityError("model checksum mismatch")


class TrustedKeyring:
    def __init__(self, keys: Mapping[str, bytes], revoked: set[str] | None = None):
        self._keys = dict(keys)
        self._revoked = set(revoked or ())
        if not self._keys:
            raise ValidationError("at least one trusted key required")
        for key_id, public_key in self._keys.items():
            if not isinstance(key_id, str) or not key_id:
                raise ValidationError("trusted key IDs must be non-empty strings")
            try:
                Ed25519PublicKey.from_public_bytes(public_key)
            except Exception as e:
                raise ValidationError(f"invalid Ed25519 public key for {key_id!r}") from e

    def verify(self, key_id: str, manifest: ArtifactManifest, signature_b64: str) -> None:
        if key_id in self._revoked:
            raise SignatureError("signing key revoked")
        key = self._keys.get(key_id)
        if key is None:
            raise SignatureError("unknown signing key")
        verify_manifest(manifest, signature_b64, key)
