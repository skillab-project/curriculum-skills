"""
In-memory registry of analyses that are still running.

Results are only written to the DB when an analysis finishes, so until then a
run is invisible to the /runs listings. This registry lets those endpoints show
in-progress runs too (status "running"), without any frontend change: the
description of a running entry is prefixed with a "still running" notice,
which the UI already displays.

The service runs as a single uvicorn process, so a process-local dict is
enough. A restart drops the registry — which is correct, because a restart
also kills the background work that the entries describe.
"""
import threading
from datetime import datetime
from typing import Any, Dict, List

RUNNING_NOTICE = "⏳ Still running — results will appear here when the analysis finishes. Refresh to update."

_lock = threading.Lock()
_runs: Dict[str, Dict[str, Dict[str, Any]]] = {}  # kind -> run_id -> entry


def start(kind: str, run_id: str, entry: Dict[str, Any]) -> None:
    item = dict(entry)
    item["run_id"] = run_id
    item.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    with _lock:
        _runs.setdefault(kind, {})[run_id] = item


def finish(kind: str, run_id: str) -> None:
    with _lock:
        _runs.get(kind, {}).pop(run_id, None)


def is_running(kind: str, run_id: str) -> bool:
    with _lock:
        return run_id in _runs.get(kind, {})


def title_running(kind: str, title: str) -> bool:
    with _lock:
        return any(e.get("title") == title for e in _runs.get(kind, {}).values())


def listing(kind: str) -> List[Dict[str, Any]]:
    """Running entries shaped like /runs items, newest first."""
    with _lock:
        items = [dict(e) for e in _runs.get(kind, {}).values()]
    items.sort(key=lambda e: e.get("created_at") or "", reverse=True)
    for e in items:
        desc = e.get("description")
        e["description"] = f"{RUNNING_NOTICE} {desc}" if desc else RUNNING_NOTICE
        e["status"] = "running"
    return items
