from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from edgeai.config import RuntimeConfig, load_config
from edgeai.deploy import ABDeployer
from edgeai.engine import InferenceEngine
from edgeai.errors import BackendProcessError, ExecutionDeadlineExceeded
from edgeai.health import HealthPolicy
from edgeai.qualification import QualificationLimits
from edgeai.runtime import BoundedInferenceExecutor, _RollingRestartBudget
from edgeai.sandbox import apply_process_sandbox
from tests.helpers import signed_registry


def _deployed_engine(tmp_path):
    registry, *_ = signed_registry(tmp_path)
    deployer = ABDeployer(
        tmp_path / "deploy-iter5",
        registry,
        HealthPolicy(min_accuracy=.9, max_ece=1, max_p95_latency_ms=100, max_rss_mb=5000),
    )
    x = np.array([[2, 0], [0, 2]], np.float32)
    y = np.array([0, 1])
    deployer.deploy("1.0.0", x, y)
    return InferenceEngine(
        deployer,
        max_batch=4,
        max_input_features=2,
        max_input_values=8,
        max_output_classes=8,
        max_output_values=16,
    )


def test_process_sandbox_clears_environment_disables_cores_and_sets_no_new_privs(tmp_path):
    script = r'''
import json, os, resource
from edgeai.sandbox import apply_process_sandbox
apply_process_sandbox(memory_limit_bytes=None, nofile_limit=32)
status = {}
with open('/proc/self/status', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('NoNewPrivs:'):
            status['no_new_privs'] = int(line.split(':', 1)[1].strip())
            break
print(json.dumps({
    'env': dict(os.environ),
    'core': list(resource.getrlimit(resource.RLIMIT_CORE)),
    'nofile': list(resource.getrlimit(resource.RLIMIT_NOFILE)),
    **status,
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["EDGEAI_API_KEY"] = "must-not-reach-model-runtime"
    env["UNRELATED_SECRET"] = "also-removed"
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = json.loads(proc.stdout)
    assert result["env"] == {}
    assert result["core"] == [0, 0]
    assert result["nofile"] == [32, 32]
    assert result["no_new_privs"] == 1


def test_process_sandbox_validation_is_fail_closed():
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            apply_process_sandbox(memory_limit_bytes=value, nofile_limit=64)
    for value in (0, 15, True, 1.5):
        with pytest.raises(ValueError):
            apply_process_sandbox(memory_limit_bytes=None, nofile_limit=value)
    with pytest.raises(ValueError):
        apply_process_sandbox(memory_limit_bytes=None, nofile_limit=64, clear_environment="yes")
    with pytest.raises(ValueError):
        apply_process_sandbox(memory_limit_bytes=None, nofile_limit=64, no_new_privs=1)


def test_process_restart_budget_breaks_crash_loop_and_recovers_after_window(tmp_path):
    executor = BoundedInferenceExecutor(
        _deployed_engine(tmp_path),
        capacity=2,
        workers=1,
        execution_mode="process",
        execution_timeout_s=1e-9,
        process_control_timeout_s=10.0,
        process_restart_limit=1,
        process_restart_window_s=60.0,
    )
    try:
        with pytest.raises(ExecutionDeadlineExceeded):
            executor.submit([[2, 0]])
        with pytest.raises(ExecutionDeadlineExceeded):
            executor.submit([[2, 0]])
        with pytest.raises(BackendProcessError, match="restart budget exhausted"):
            executor.submit([[2, 0]])
        stats = executor.stats()
        assert stats["process_restart_throttled"] >= 1
        assert stats["process_restart_limit"] == 1
        assert stats["process_restart_window_s"] == pytest.approx(60.0)

        # Simulate the rolling window elapsing without making this test sleep for a minute.
        executor._restart_budget._times.clear()
        executor.execution_timeout_s = 1.0
        pred, _, _ = executor.submit([[2, 0]])
        assert pred.tolist() == [0]
    finally:
        executor.close(wait=True, timeout_s=2.0)


def test_iteration5_process_controls_validate_across_config_and_qualification():
    RuntimeConfig(
        "d", "r",
        execution_mode="process",
        execution_timeout_s=1.0,
        process_nofile_limit=32,
        process_restart_limit=2,
        process_restart_window_s=10.0,
    ).validate()
    for kwargs in (
        {"process_nofile_limit": 15},
        {"process_restart_limit": 0},
        {"process_restart_window_s": 0},
    ):
        with pytest.raises(ValueError):
            RuntimeConfig("d", "r", **kwargs).validate()
    with pytest.raises(ValueError, match="nofile_limit"):
        QualificationLimits(nofile_limit=15).validate()


def test_reference_profile_uses_hard_process_isolation():
    config = load_config("configs/reference.yaml")
    assert config.execution_mode == "process"
    assert config.execution_timeout_s == pytest.approx(2.0)
    assert config.process_nofile_limit == 64
    assert config.process_restart_limit == 8


def test_restart_budget_is_global_and_windowed(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("edgeai.runtime.time.monotonic", lambda: now[0])
    throttled = []
    budget = _RollingRestartBudget(limit=1, window_s=10.0, on_throttled=lambda: throttled.append(True))
    budget.reserve()
    assert budget.current() == 1
    with pytest.raises(BackendProcessError, match="restart budget exhausted"):
        budget.reserve()
    assert throttled == [True]
    now[0] = 111.0
    assert budget.current() == 0
    budget.reserve()
    assert budget.current() == 1
