"""
persistence.py — Lightweight JSON-file case state for Sentinel Phase 2.

Persists per-alert case fields across process restarts:
  - status
  - assignee
  - disposition
  - reviewer_notes
  - audit

WARNING: This is prototype-level file persistence.
It is intentionally simple and is NOT suitable for production.
It will not survive infrastructure replacement or redeploy without copying the
case_state.json file. Replace this module with a database-backed implementation
for any production or multi-instance deployment.
"""

import json, os, datetime as dt

# Location: same directory as app.py / engine.py
_STORE_PATH = os.path.join(os.path.dirname(__file__), "case_state.json")

# Fields we persist (everything else is re-derived from engine on startup)
_PERSIST_KEYS = ("status", "assignee", "disposition", "reviewer_notes", "audit")


def load() -> dict:
    """Load persisted case state. Returns {alert_id: {field: value}}."""
    if not os.path.exists(_STORE_PATH):
        return {}
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def save(alerts: list) -> None:
    """Write current case state to disk. Call after every mutation."""
    state = {}
    for a in alerts:
        state[a["id"]] = {k: a.get(k) for k in _PERSIST_KEYS}
    try:
        with open(_STORE_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)
    except OSError:
        pass   # non-fatal — prototype constraint


def apply(alerts: list) -> None:
    """
    Merge persisted case state back onto freshly-built alerts.
    Called once at startup after engine.build().
    Fields in the store win over the engine's defaults (OPEN / None etc.).
    """
    state = load()
    for a in alerts:
        saved = state.get(a["id"])
        if not saved:
            continue
        for k in _PERSIST_KEYS:
            if k in saved and saved[k] is not None:
                a[k] = saved[k]


def add_audit(alert: dict, by: str, note: str) -> None:
    """Append an audit entry with ISO timestamp."""
    alert["audit"].append(dict(
        ts=dt.datetime.now().isoformat(),
        by=by,
        note=note,
    ))
