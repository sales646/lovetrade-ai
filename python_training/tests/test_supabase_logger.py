"""Tests for the SupabaseLogger utility."""
import os
import sys
from pathlib import Path

import pytest

# Ensure the python_training directory is importable when tests are executed
MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import supabase_logger  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def _cleanup_env():
    original_env = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original_env)


def test_start_run_populates_required_columns(monkeypatch):
    """SupabaseLogger.start_run should provide every NOT NULL column."""

    # Provide fake credentials so the logger initializes in enabled mode
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    # Avoid spawning background threads during the unit test
    class _ThreadStub:
        def __init__(self, *args, **kwargs):
            self._target = kwargs.get("target")

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(supabase_logger.threading, "Thread", _ThreadStub)

    # Capture enqueued payloads instead of touching the real queue
    captured_events = []

    def _capture_enqueue(self, table, payload):
        captured_events.append((table, payload))

    monkeypatch.setattr(supabase_logger.SupabaseLogger, "_enqueue", _capture_enqueue, raising=False)

    # Provide a stub Supabase client factory to satisfy initialization
    monkeypatch.setattr(supabase_logger, "create_client", lambda url, key: object(), raising=False)

    logger = supabase_logger.SupabaseLogger(enabled=True)

    run_config = {
        "run_name": "20240101-120000-42-symbols",
        "phase": "initializing",
        "status": "running",
        "epochs": 10,
        "bc_epochs": 5,
        "symbol_limit": 100,
        "data_start": "2020-01-01",
        "data_end": "2024-01-01",
    }

    run_id = logger.start_run(run_config)

    assert run_id is not None
    assert captured_events, "start_run should enqueue a training_runs payload"

    table, payload = captured_events[0]
    assert table == "training_runs"
    assert payload["id"] == run_id
    assert payload["run_name"] == run_config["run_name"]
    assert payload["phase"] == run_config["phase"]
    assert payload["status"] == run_config["status"]

    # Hyperparameters should include the non-metadata values provided in the config
    assert payload["hyperparams"]["epochs"] == run_config["epochs"]
    assert payload["hyperparams"]["bc_epochs"] == run_config["bc_epochs"]
    assert payload["hyperparams"]["symbol_limit"] == run_config["symbol_limit"]

    # Config snapshot is stored for richer run metadata
    assert payload["config"]["run_name"] == run_config["run_name"]
    assert payload["config"]["data_start"] == run_config["data_start"]

    logger.close()
