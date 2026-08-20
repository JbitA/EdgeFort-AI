from __future__ import annotations

import os
import shutil
import stat
import threading
from dataclasses import replace
from pathlib import Path

from .errors import DeploymentError, RollbackRejected
from .health import HealthPolicy, benchmark, enforce, load_runtime
from .qualification import QualificationLimits, qualify_model_bytes
from .model_io import DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES
from .registry import Registry
from .rollback_anchor import RollbackAnchor
from .types import DeploymentState, DeploymentStatus
from .util import MAX_RELEASE_SEQUENCE, InterProcessFileLock, atomic_write, canonical_json_bytes, safe_version, strict_json_loads

_VALID_SLOTS = {"A", "B"}
_VALID_STATUSES = {status.value for status in DeploymentStatus}
_MAX_STATE_BYTES = 64 * 1024
_MAX_ERROR_CHARS = 4096
_MAX_PENDING_BYTES = 16 * 1024


class ABDeployer:
    """Crash-consistent A/B deployment manager with anti-rollback release sequences.

    Writers are serialized across both threads and processes. State reads remain lock-free
    because state.json is committed by atomic rename; this lets a cached active runtime
    continue serving while a candidate is benchmarked under the writer lock.
    """

    def __init__(
        self,
        root: str | Path,
        registry: Registry,
        policy: HealthPolicy | None = None,
        *,
        max_model_uncompressed_bytes: int = DEFAULT_MAX_MODEL_UNCOMPRESSED_BYTES,
        rollback_anchor: RollbackAnchor | None = None,
        isolate_health_gate: bool = False,
        qualification_limits: QualificationLimits | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise DeploymentError("deployment root must be a regular directory")
        if (
            isinstance(max_model_uncompressed_bytes, bool)
            or not isinstance(max_model_uncompressed_bytes, int)
            or max_model_uncompressed_bytes < 1024
        ):
            raise ValueError("max_model_uncompressed_bytes must be an integer >= 1024")
        self.registry = registry
        self.policy = policy or HealthPolicy()
        self.max_model_uncompressed_bytes = max_model_uncompressed_bytes
        if not isinstance(isolate_health_gate, bool):
            raise ValueError("isolate_health_gate must be boolean")
        self.isolate_health_gate = isolate_health_gate
        self.qualification_limits = (qualification_limits or QualificationLimits()).validate()
        self.state_path = self.root / "state.json"
        self.pending_anchor_path = self.root / ".pending-anchor-commit.json"
        self.rollback_anchor = rollback_anchor
        self._writer_lock = threading.RLock()
        self._ipc_lock = InterProcessFileLock(self.root / ".deployment.lock")
        self._trusted_floor = self._read_anchor_floor() if rollback_anchor is not None else None
        self._anchor_recovery_required = False
        self._recover()

    @staticmethod
    def _parse_state(data: object) -> DeploymentState:
        if not isinstance(data, dict):
            raise DeploymentError("deployment state must be a JSON object")
        expected = set(DeploymentState.__dataclass_fields__)
        if set(data) != expected:
            raise DeploymentError("deployment state has missing or unknown fields")

        generation = data["generation"]
        floor = data["highest_release_sequence"]
        active = data["active_slot"]
        previous = data["previous_slot"]
        slots = data["slots"]
        status = data["status"]
        last_error = data["last_error"]

        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise DeploymentError("invalid deployment generation")
        if (
            isinstance(floor, bool)
            or not isinstance(floor, int)
            or floor < -1
            or floor > MAX_RELEASE_SEQUENCE
        ):
            raise DeploymentError("invalid anti-rollback floor")
        if active is not None and active not in _VALID_SLOTS:
            raise DeploymentError("invalid active slot")
        if previous is not None and previous not in _VALID_SLOTS:
            raise DeploymentError("invalid previous slot")
        if active is not None and active == previous:
            raise DeploymentError("active and previous slots must differ")
        if not isinstance(slots, dict) or set(slots) != _VALID_SLOTS:
            raise DeploymentError("deployment slots must contain exactly A and B")
        for slot, version in slots.items():
            if version is not None:
                try:
                    safe_version(version)
                except (TypeError, ValueError) as e:
                    raise DeploymentError(f"invalid version in slot {slot}") from e
        if active is not None and slots[active] is None:
            raise DeploymentError("active slot has no version")
        if previous is not None and slots[previous] is None:
            raise DeploymentError("previous slot has no version")
        populated = {slot for slot, version in slots.items() if version is not None}
        if active is None and populated:
            raise DeploymentError("deployment state has slot data but no active slot")
        if active is not None:
            allowed_populated = {active} | ({previous} if previous is not None else set())
            if not populated.issubset(allowed_populated):
                raise DeploymentError("populated slot is not active or previous")
        if status not in _VALID_STATUSES:
            raise DeploymentError("invalid deployment status")
        if last_error is not None and not isinstance(last_error, str):
            raise DeploymentError("invalid deployment error field")
        if isinstance(last_error, str) and len(last_error) > _MAX_ERROR_CHARS:
            raise DeploymentError("deployment error field exceeds size limit")

        return DeploymentState(
            generation=generation,
            active_slot=active,
            previous_slot=previous,
            slots=dict(slots),
            highest_release_sequence=floor,
            status=status,
            last_error=last_error,
        )

    @staticmethod
    def _validate_anchor_floor(value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < -1
            or value > MAX_RELEASE_SEQUENCE
        ):
            raise DeploymentError("rollback anchor returned an invalid floor")
        return value

    def _read_anchor_floor(self) -> int:
        if self.rollback_anchor is None:
            raise DeploymentError("rollback anchor is not configured")
        try:
            return self._validate_anchor_floor(self.rollback_anchor.read_floor())
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(f"rollback anchor read failed: {e}") from e

    def _advance_anchor(self, release_sequence: int) -> int:
        if self.rollback_anchor is None:
            return release_sequence
        try:
            floor = self._validate_anchor_floor(
                self.rollback_anchor.advance_to(release_sequence)
            )
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(f"rollback anchor advance failed: {e}") from e
        if floor != release_sequence:
            raise DeploymentError(
                "rollback anchor advanced to an unexpected release sequence"
            )
        self._trusted_floor = floor
        return floor

    def _validate_state_anchor(self, state: DeploymentState) -> None:
        if self._trusted_floor is None:
            return
        if self._anchor_recovery_required:
            raise DeploymentError("external rollback anchor transition requires recovery")

        # Avoid a TPM/secure-element read on every inference. Refresh when another
        # cooperating process committed a different floor or while a journal signals
        # that an anchor-first transaction may be in flight/crashed.
        pending_exists = self.pending_anchor_path.exists() or self.pending_anchor_path.is_symlink()
        if state.highest_release_sequence != self._trusted_floor or pending_exists:
            observed = self._read_anchor_floor()
            self._trusted_floor = observed
        if state.highest_release_sequence != self._trusted_floor:
            raise DeploymentError(
                "deployment state anti-rollback floor differs from external rollback anchor"
            )

    @staticmethod
    def _parse_pending_anchor(data: object) -> dict[str, object]:
        if not isinstance(data, dict):
            raise DeploymentError("pending anchor commit must be a JSON object")
        expected = {
            "schema_version",
            "previous_generation",
            "previous_floor",
            "target_slot",
            "version",
            "release_sequence",
        }
        if set(data) != expected or data.get("schema_version") != 1:
            raise DeploymentError("invalid pending anchor commit structure")
        generation = data["previous_generation"]
        previous_floor = data["previous_floor"]
        target_slot = data["target_slot"]
        version = data["version"]
        release_sequence = data["release_sequence"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise DeploymentError("invalid pending anchor generation")
        if isinstance(previous_floor, bool) or not isinstance(previous_floor, int) or previous_floor < -1:
            raise DeploymentError("invalid pending anchor previous floor")
        if target_slot not in _VALID_SLOTS:
            raise DeploymentError("invalid pending anchor target slot")
        try:
            safe_version(version)
        except (TypeError, ValueError) as e:
            raise DeploymentError("invalid pending anchor version") from e
        if (
            isinstance(release_sequence, bool)
            or not isinstance(release_sequence, int)
            or release_sequence < 0
            or release_sequence > MAX_RELEASE_SEQUENCE
        ):
            raise DeploymentError("invalid pending anchor release sequence")
        if release_sequence <= previous_floor:
            raise DeploymentError("pending anchor commit does not advance rollback floor")
        return dict(data)

    def _read_pending_anchor(self) -> dict[str, object] | None:
        path = self.pending_anchor_path
        if not path.exists():
            if path.is_symlink():
                raise DeploymentError("pending anchor commit must not be a symlink")
            return None
        if path.is_symlink():
            raise DeploymentError("pending anchor commit must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
            with os.fdopen(fd, "rb", closefd=True) as handle:
                st = os.fstat(handle.fileno())
                if not stat.S_ISREG(st.st_mode):
                    raise DeploymentError("pending anchor commit is not a regular file")
                raw = handle.read(_MAX_PENDING_BYTES + 1)
        except DeploymentError:
            raise
        except OSError as e:
            raise DeploymentError(f"pending anchor commit cannot be opened safely: {e}") from e
        if len(raw) > _MAX_PENDING_BYTES:
            raise DeploymentError("pending anchor commit exceeds size limit")
        try:
            return self._parse_pending_anchor(strict_json_loads(raw.decode("utf-8")))
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(f"invalid pending anchor commit: {e}") from e

    def _write_pending_anchor(
        self, state: DeploymentState, target_slot: str, version: str, release_sequence: int
    ) -> None:
        pending = {
            "schema_version": 1,
            "previous_generation": state.generation,
            "previous_floor": state.highest_release_sequence,
            "target_slot": target_slot,
            "version": version,
            "release_sequence": release_sequence,
        }
        self._parse_pending_anchor(pending)
        atomic_write(self.pending_anchor_path, canonical_json_bytes(pending) + b"\n")

    def _remove_pending_anchor(self) -> None:
        try:
            self.pending_anchor_path.unlink()
        except FileNotFoundError:
            return
        try:
            dfd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass

    def _recover_anchor_commit(
        self, state: DeploymentState, pending: dict[str, object] | None
    ) -> DeploymentState:
        if self._trusted_floor is None:
            if pending is not None:
                raise DeploymentError(
                    "pending external-anchor commit exists but no rollback anchor is configured"
                )
            return state

        anchor_floor = self._trusted_floor
        local_floor = state.highest_release_sequence
        if local_floor > anchor_floor:
            raise DeploymentError(
                "deployment state anti-rollback floor exceeds external rollback anchor"
            )
        if local_floor == anchor_floor:
            if pending is not None:
                # If the state commit already landed, the journal is stale and can be
                # removed only after authenticating that it describes that committed state.
                if (
                    pending["release_sequence"] == anchor_floor
                    and pending["previous_generation"] + 1 == state.generation
                    and state.active_slot == pending["target_slot"]
                    and state.slots[state.active_slot] == pending["version"]
                ):
                    self.registry.assert_release_sequence_identity(
                        pending["version"], pending["release_sequence"]
                    )
                    self._verified_slot_manifest(
                        pending["target_slot"], pending["version"]
                    )
                    self._remove_pending_anchor()
                else:
                    # Anchor did not advance; this is a pre-anchor crash. Normal backup
                    # recovery restores the durable state and the journal is discarded.
                    if (
                        pending["previous_floor"] != local_floor
                        or pending["previous_generation"] != state.generation
                    ):
                        raise DeploymentError("pending anchor journal conflicts with durable state")
            return state

        # The anchor advanced before the local state commit. Complete only the exact
        # authenticated transaction recorded before the irreversible anchor update.
        if pending is None:
            raise DeploymentError(
                "external rollback anchor is ahead of deployment state without recovery journal"
            )
        if (
            pending["previous_floor"] != local_floor
            or pending["previous_generation"] != state.generation
            or pending["release_sequence"] != anchor_floor
        ):
            raise DeploymentError("external rollback anchor and recovery journal disagree")
        version = pending["version"]
        target_slot = pending["target_slot"]
        self.registry.assert_release_sequence_identity(version, anchor_floor)
        manifest = self._verified_slot_manifest(target_slot, version)
        if manifest.release_sequence != anchor_floor:
            raise DeploymentError("pending slot does not match external rollback anchor")
        slots = dict(state.slots)
        slots[target_slot] = version
        recovered = DeploymentState(
            generation=state.generation + 1,
            active_slot=target_slot,
            previous_slot=state.active_slot,
            slots=slots,
            highest_release_sequence=anchor_floor,
            status=DeploymentStatus.ACTIVE.value,
            last_error=None,
        )
        self._write(recovered)
        self._remove_pending_anchor()
        return recovered

    def _slot_dir(self, slot: str) -> Path:
        if slot not in _VALID_SLOTS:
            raise DeploymentError("invalid deployment slot")
        return self.root / "slots" / slot

    def _slot_data_exists(self) -> bool:
        slots_root = self.root / "slots"
        if slots_root.is_symlink():
            return True
        return any(self._slot_dir(slot).exists() or self._slot_dir(slot).is_symlink() for slot in _VALID_SLOTS)

    def _read(self, *, validate_anchor: bool = True) -> DeploymentState:
        if not self.state_path.exists():
            if self.state_path.is_symlink() or self._slot_data_exists():
                raise DeploymentError("deployment state missing while slot data exists")
            return DeploymentState()
        if self.state_path.is_symlink():
            raise DeploymentError("deployment state must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.state_path, flags)
            with os.fdopen(fd, "rb", closefd=True) as handle:
                st = os.fstat(handle.fileno())
                if not stat.S_ISREG(st.st_mode):
                    raise DeploymentError("deployment state is not a regular file")
                raw = handle.read(_MAX_STATE_BYTES + 1)
        except DeploymentError:
            raise
        except OSError as e:
            raise DeploymentError(f"deployment state cannot be opened safely: {e}") from e
        if len(raw) > _MAX_STATE_BYTES:
            raise DeploymentError("deployment state exceeds size limit")
        try:
            state = self._parse_state(strict_json_loads(raw.decode("utf-8")))
        except DeploymentError:
            raise
        except Exception as e:
            raise DeploymentError(f"invalid deployment state: {e}") from e
        if validate_anchor:
            self._validate_state_anchor(state)
        return state

    def _write(self, state: DeploymentState) -> None:
        self._parse_state(state.to_dict())
        atomic_write(self.state_path, canonical_json_bytes(state.to_dict()) + b"\n")

    @staticmethod
    def _error_text(error: Exception) -> str:
        text = f"{type(error).__name__}: {error}"
        return text[:_MAX_ERROR_CHARS]

    def _canonical_manifest(self, expected_version: str):
        _, manifest = self.registry.inspect(expected_version)
        return manifest

    def _verified_directory_manifest(self, directory: Path, expected_version: str):
        canonical = self._canonical_manifest(expected_version)
        manifest = self.registry.verify_directory(directory, expected_version=expected_version)
        if manifest != canonical:
            raise DeploymentError("slot manifest differs from canonical registry release")
        return manifest

    def _verified_slot_manifest(self, slot: str, expected_version: str):
        return self._verified_directory_manifest(self._slot_dir(slot), expected_version)

    def _verified_slot_runtime(self, slot: str, expected_version: str):
        slot_dir = self._slot_dir(slot)
        canonical = self._canonical_manifest(expected_version)
        with self.registry.open_verified_model(
            slot_dir, expected_version=expected_version
        ) as (handle, manifest):
            if manifest != canonical:
                raise DeploymentError("slot manifest differs from canonical registry release")
            runtime = load_runtime(
                handle,
                manifest.model_kind,
                manifest.runtime_abi,
                max_uncompressed_bytes=self.max_model_uncompressed_bytes,
            )
        return runtime

    def _verified_slot_artifact(self, slot: str, expected_version: str):
        """Return bounded authenticated model bytes plus the canonical manifest.

        This is the process-isolation transfer boundary: the parent authenticates the
        exact active slot inode and the child receives immutable bytes rather than a
        pickled live runtime/session object.
        """
        slot_dir = self._slot_dir(slot)
        canonical = self._canonical_manifest(expected_version)
        with self.registry.open_verified_model(
            slot_dir, expected_version=expected_version
        ) as (handle, manifest):
            if manifest != canonical:
                raise DeploymentError("slot manifest differs from canonical registry release")
            if manifest.model_bytes > self.max_model_uncompressed_bytes:
                raise DeploymentError("model artifact exceeds process-transfer size limit")
            raw = handle.read(manifest.model_bytes + 1)
            if len(raw) != manifest.model_bytes:
                raise DeploymentError("model artifact size changed during stable-handle read")
        return raw, manifest

    def _recover_backups(self, state: DeploymentState) -> None:
        for slot in _VALID_SLOTS:
            backup = self.root / f".backup-{slot}"
            if not backup.exists() and not backup.is_symlink():
                continue
            if backup.is_symlink() or not backup.is_dir():
                raise DeploymentError(f"invalid backup for slot {slot}")
            expected = state.slots[slot]
            if expected is None:
                raise DeploymentError(f"unexpected backup exists for empty slot {slot}")
            final = self._slot_dir(slot)

            final_ok = False
            if final.exists() and not final.is_symlink():
                try:
                    self._verified_directory_manifest(final, expected)
                    final_ok = True
                except Exception:
                    final_ok = False
            if final_ok:
                shutil.rmtree(backup)
                continue

            # State was not committed to a newer slot; restore the bytes that match the
            # durable state. If neither copy matches, startup fails closed.
            self._verified_directory_manifest(backup, expected)
            if final.exists() or final.is_symlink():
                if final.is_symlink() or not final.is_dir():
                    raise DeploymentError(f"invalid slot path for {slot}")
                shutil.rmtree(final)
            os.replace(backup, final)

    def _recover(self) -> None:
        with self._writer_lock, self._ipc_lock.exclusive():
            slots_root = self.root / "slots"
            if slots_root.is_symlink():
                raise DeploymentError("slots root must not be a symlink")
            slots_root.mkdir(exist_ok=True)
            if not slots_root.is_dir():
                raise DeploymentError("slots root must be a directory")

            if not self.state_path.exists():
                if self.state_path.is_symlink() or self._slot_data_exists():
                    raise DeploymentError("deployment state missing while slot data exists")
                self._write(DeploymentState())
            state = self._read(validate_anchor=False)
            pending = self._read_pending_anchor()
            state = self._recover_anchor_commit(state, pending)
            self._recover_backups(state)

            # Candidate directories are safe to remove while holding the global writer
            # lock: no other process can be in the deployment transaction concurrently.
            for p in self.root.glob(".candidate-*"):
                if p.is_symlink() or not p.is_dir():
                    raise DeploymentError("invalid candidate path during recovery")
                shutil.rmtree(p)

            # State is authoritative. Remove an uncommitted first-write slot left by a
            # crash before state commit; authenticate every referenced slot and ensure
            # no slot claims a release sequence above the durable anti-rollback floor.
            for slot, version in state.slots.items():
                slot_dir = self._slot_dir(slot)
                if version is None:
                    if slot_dir.exists() or slot_dir.is_symlink():
                        if slot_dir.is_symlink() or not slot_dir.is_dir():
                            raise DeploymentError(f"invalid uncommitted slot path for {slot}")
                        shutil.rmtree(slot_dir)
                    continue
                slot_manifest = self._verified_slot_manifest(slot, version)
                if slot_manifest.release_sequence > state.highest_release_sequence:
                    raise DeploymentError("slot release sequence exceeds anti-rollback floor")

            # A pre-anchor crash leaves a journal while the external and local floors are
            # still equal. By this point backups/uncommitted slots have been restored to
            # the durable state, so the abandoned journal can be removed safely.
            if self.pending_anchor_path.exists() or self.pending_anchor_path.is_symlink():
                if self.pending_anchor_path.is_symlink():
                    raise DeploymentError("pending anchor commit must not be a symlink")
                self._remove_pending_anchor()
            self._validate_state_anchor(state)

    def state(self) -> DeploymentState:
        # Atomic rename makes readers observe either the old complete state or the new
        # complete state. Avoiding the writer lock keeps inference available during the
        # candidate benchmark phase.
        return self._read()

    def _health_metrics(self, model_bytes: bytes, manifest, x_validation, y_validation):
        if self.isolate_health_gate:
            return qualify_model_bytes(
                model_bytes=model_bytes,
                model_kind=manifest.model_kind,
                runtime_abi=manifest.runtime_abi,
                max_uncompressed_bytes=self.max_model_uncompressed_bytes,
                x_validation=x_validation,
                y_validation=y_validation,
                limits=self.qualification_limits,
            )
        # Compatibility/debug path. Production deployments should keep isolation enabled.
        import io

        runtime = load_runtime(
            io.BytesIO(model_bytes),
            manifest.model_kind,
            manifest.runtime_abi,
            max_uncompressed_bytes=self.max_model_uncompressed_bytes,
        )
        return benchmark(runtime, x_validation, y_validation)

    def deploy(self, version: str, x_validation, y_validation, allow_downgrade=False):
        with self._writer_lock, self._ipc_lock.exclusive():
            registry_dir, manifest = self.registry.inspect(version)
            self.registry.assert_release_sequence_identity(version, manifest.release_sequence)
            state = self._read()
            if (
                self.rollback_anchor is not None
                and manifest.release_sequence <= state.highest_release_sequence
            ):
                raise RollbackRejected(
                    "external rollback anchor cannot be bypassed by allow_downgrade"
                )
            if manifest.release_sequence <= state.highest_release_sequence and not allow_downgrade:
                raise RollbackRejected("release sequence is not newer than trusted floor")

            inactive = "B" if state.active_slot == "A" else "A"
            candidate = self.root / f".candidate-{inactive}"
            shutil.rmtree(candidate, ignore_errors=True)
            shutil.copytree(registry_dir, candidate)

            try:
                with self.registry.open_verified_model(
                    candidate, expected_version=version
                ) as (candidate_handle, staged_manifest):
                    if staged_manifest != manifest:
                        raise DeploymentError("registry entry changed between inspection and staging")
                    candidate_bytes = candidate_handle.read(staged_manifest.model_bytes + 1)
                    if len(candidate_bytes) != staged_manifest.model_bytes:
                        raise DeploymentError("candidate artifact size changed during stable-handle read")
                metrics = self._health_metrics(
                    candidate_bytes, staged_manifest, x_validation, y_validation
                )
                baseline = None
                if state.active_slot and state.slots.get(state.active_slot):
                    active_version = state.slots[state.active_slot]
                    baseline_bytes, baseline_manifest = self._verified_slot_artifact(
                        state.active_slot, active_version
                    )
                    baseline = self._health_metrics(
                        baseline_bytes, baseline_manifest, x_validation, y_validation
                    )
                enforce(self.policy, metrics, baseline)
            except Exception as e:
                shutil.rmtree(candidate, ignore_errors=True)
                self._write(
                    replace(
                        state,
                        generation=state.generation + 1,
                        status=DeploymentStatus.FAILED.value,
                        last_error=self._error_text(e),
                    )
                )
                raise

            final = self._slot_dir(inactive)
            backup = self.root / f".backup-{inactive}"
            shutil.rmtree(backup, ignore_errors=True)
            if final.exists() or final.is_symlink():
                if final.is_symlink() or not final.is_dir():
                    raise DeploymentError("inactive slot path is not a regular directory")
                os.replace(final, backup)
            anchor_transaction_started = False
            state_committed = False
            try:
                os.replace(candidate, final)
                # Re-open and authenticate the exact post-rename inode. The identical
                # signed bytes were already parser/benchmark-qualified in a killable
                # subprocess, so loading them again in the lifecycle process would
                # recreate the parser-crash boundary this transaction just removed.
                self._verified_slot_artifact(inactive, version)
                slots = dict(state.slots)
                slots[inactive] = version
                new_floor = max(state.highest_release_sequence, manifest.release_sequence)
                if self.rollback_anchor is not None:
                    # Journal the exact authenticated transition before the irreversible
                    # external monotonic update. If power fails after the anchor advances,
                    # startup can finish this state commit instead of bricking availability.
                    self._write_pending_anchor(
                        state, inactive, version, manifest.release_sequence
                    )
                    anchor_transaction_started = True
                    # From this point until local state commit, any ambiguous anchor
                    # failure must force recovery rather than allow stale service.
                    self._anchor_recovery_required = True
                    new_floor = self._advance_anchor(manifest.release_sequence)
                new = DeploymentState(
                    generation=state.generation + 1,
                    active_slot=inactive,
                    previous_slot=state.active_slot,
                    slots=slots,
                    highest_release_sequence=new_floor,
                    status=DeploymentStatus.ACTIVE.value,
                    last_error=None,
                )
                self._write(new)
                state_committed = True
                if self.rollback_anchor is not None:
                    self._anchor_recovery_required = False
                    self._remove_pending_anchor()
                shutil.rmtree(backup, ignore_errors=True)
                return metrics
            except Exception:
                # Once an external-anchor transaction has started, do not destructively
                # guess whether the monotonic update became durable. Leave final/backup/
                # journal intact so startup recovery can compare them to the anchor.
                if anchor_transaction_started or state_committed:
                    raise
                if final.exists() and not final.is_symlink():
                    shutil.rmtree(final, ignore_errors=True)
                if backup.exists():
                    os.replace(backup, final)
                raise

    def rollback(self) -> bool:
        with self._writer_lock, self._ipc_lock.exclusive():
            state = self._read()
            if not state.previous_slot or not state.slots.get(state.previous_slot):
                return False
            previous_version = state.slots[state.previous_slot]
            self._verified_slot_runtime(state.previous_slot, previous_version)
            new = DeploymentState(
                generation=state.generation + 1,
                active_slot=state.previous_slot,
                previous_slot=state.active_slot,
                slots=dict(state.slots),
                highest_release_sequence=state.highest_release_sequence,
                status=DeploymentStatus.ROLLED_BACK.value,
                last_error=None,
            )
            self._write(new)
            return True

    def active_identity(self) -> tuple[int, str | None]:
        state = self._read()
        version = state.slots.get(state.active_slot) if state.active_slot else None
        return state.generation, version

    def active_version(self):
        return self.active_identity()[1]

    def active_snapshot(self):
        """Return a stable ``(runtime, version, generation)`` under a shared process lock."""
        with self._ipc_lock.shared():
            state = self._read()
            if not state.active_slot:
                raise DeploymentError("no active model")
            version = state.slots[state.active_slot]
            runtime = self._verified_slot_runtime(state.active_slot, version)
            return runtime, version, state.generation

    def active_artifact_snapshot(self):
        """Return stable authenticated bytes for a process-isolated runtime bootstrap."""
        with self._ipc_lock.shared():
            state = self._read()
            if not state.active_slot:
                raise DeploymentError("no active model")
            version = state.slots[state.active_slot]
            model_bytes, manifest = self._verified_slot_artifact(state.active_slot, version)
            return model_bytes, manifest, state.generation

    def active_runtime(self):
        return self.active_snapshot()[0]
