"""SQLite analysis history. Local, single-user, unencrypted (see README: limitations)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    preview TEXT NOT NULL,
    ml_label TEXT,
    verdict TEXT,
    source_count INTEGER NOT NULL DEFAULT 0,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""


class HistoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if str(self.path) == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = self._mem or sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                if not self._mem:
                    conn.close()

    def save(self, result: dict) -> None:
        ml = (result.get("ml") or {}).get("result") or {}
        ver = result.get("verification") or {}
        preview = (result.get("input", {}).get("headline") or result.get("input", {}).get("text_preview") or "")[:200]
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?)",
                (result["id"], result["created_at"], result["mode"], preview, ml.get("label"),
                 ver.get("verdict"), len(result.get("evidence") or []), json.dumps(result)),
            )

    def list(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, created_at, mode, preview, ml_label, verdict, source_count FROM analyses "
                "ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()
        return [dict(r) for r in rows]

    def get(self, id_: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result_json FROM analyses WHERE id=?", (id_,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def delete(self, id_: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM analyses WHERE id=?", (id_,)).rowcount > 0

    def clear(self) -> int:
        with self._conn() as c:
            return c.execute("DELETE FROM analyses").rowcount
