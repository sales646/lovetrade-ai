"""Non-blocking Supabase logging utilities for training runs."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # Optional dependency
    from supabase import Client, create_client
except Exception:  # pragma: no cover - optional dependency
    Client = None  # type: ignore
    create_client = None  # type: ignore


@dataclass
class _LogEvent:
    table: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


class SupabaseLogger:
    """Background logger with retry/backoff to Supabase tables."""

    def __init__(self, enabled: bool = True) -> None:
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self._enabled = enabled and bool(self.url and self.key and create_client)
        self._client: Optional[Client] = None
        self._queue: "queue.Queue[_LogEvent]" = queue.Queue(maxsize=1024)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.run_id: Optional[str] = None
        self._fail_once = False

        if self._enabled:
            try:
                self._client = create_client(self.url, self.key)  # type: ignore[arg-type]
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()
            except Exception:
                self._enabled = False
                self._client = None

    @staticmethod
    def _jsonify(data: Any) -> Any:
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        try:
            return json.loads(json.dumps(data, default=str))
        except (TypeError, ValueError):
            return json.dumps(data, default=str)

    def start_run(self, config: Dict[str, Any]) -> Optional[str]:
        if not self._enabled:
            return None

        config_copy = dict(config)
        self.run_id = config_copy.get("id") or str(uuid.uuid4())

        run_name = config_copy.get("run_name") or f"run-{self.run_id[:8]}"
        phase = config_copy.get("phase") or "initializing"
        status = config_copy.get("status") or "running"

        hyperparams = config_copy.get("hyperparams")
        if hyperparams is None:
            excluded = {"id", "run_name", "phase", "status", "config"}
            hyperparams = {k: v for k, v in config_copy.items() if k not in excluded}

        payload = {
            "id": self.run_id,
            "run_name": run_name,
            "phase": phase,
            "hyperparams": self._jsonify(hyperparams or {}),
            "status": status,
            "config": self._jsonify(config_copy.get("config", config_copy)),
        }

        self._enqueue("training_runs", payload)
        return self.run_id

    def finalize_run(self, status: str) -> None:
        if not self._enabled or not self.run_id:
            return
        payload = {"status": status, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._enqueue("training_runs", {"id": self.run_id, **payload})

    def log_metrics(self, split: str, step: int, metrics: Dict[str, Any]) -> None:
        if not self._enabled or not self.run_id:
            return
        payload = {
            "run_id": self.run_id,
            "split": split,
            "step": step,
            **metrics,
        }
        self._enqueue("training_metrics", payload)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        if not self._enabled or not self.run_id:
            return
        payload = {"run_id": self.run_id, **trade}
        self._enqueue("trades_log", payload)

    def flush(self, timeout: Optional[float] = None) -> None:
        if not self._enabled:
            return
        start = time.time()
        while not self._queue.empty():
            if timeout is not None and time.time() - start > timeout:
                break
            time.sleep(0.1)

    def close(self) -> None:
        if not self._enabled:
            return
        self.flush(timeout=5.0)
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _enqueue(self, table: str, payload: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(_LogEvent(table=table, payload=payload))
        except queue.Full:
            if not self._fail_once:
                print("⚠️  Supabase logging queue full — dropping events")
                self._fail_once = True

    def _worker(self) -> None:
        assert self._client is not None
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            backoff = 1.0
            for attempt in range(5):
                try:
                    table = event.table
                    payload = event.payload
                    if table == "training_runs" and "id" in payload:
                        self._client.table(table).upsert(payload).execute()
                    else:
                        self._client.table(table).insert(payload).execute()
                    break
                except Exception as exc:
                    time.sleep(backoff + 0.1 * attempt)
                    backoff = min(backoff * 2, 30.0)
                    if attempt == 4 and not self._fail_once:
                        print(f"⚠️  Supabase logging failure: {exc}")
                        self._fail_once = True
            self._queue.task_done()


__all__ = ["SupabaseLogger"]
