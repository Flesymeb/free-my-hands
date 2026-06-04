from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from fmh.models import DeploymentRequest, RequestStatus, StateEvent
from fmh.time_utils import utc_now_iso


class StateStore:
    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path).expanduser()
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_request(self, request: DeploymentRequest) -> DeploymentRequest:
        now = utc_now_iso()
        request.created_at = request.created_at or now
        request.updated_at = now
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO requests (request_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    request.status.value,
                    json.dumps(request.to_dict(), ensure_ascii=False),
                    request.created_at,
                    request.updated_at,
                ),
            )
            conn.execute(
                """
                INSERT INTO events (
                    request_id, timestamp, state_from, state_to, summary, raw_output_ref
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    now,
                    "",
                    request.status.value,
                    "request created",
                    "",
                ),
            )
        return request

    def upsert_request(self, request: DeploymentRequest) -> DeploymentRequest:
        existing = self.get_request(request.request_id)
        if existing:
            return existing
        return self.create_request(request)

    def get_request(self, request_id: str) -> DeploymentRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if not row:
            return None
        return DeploymentRequest.from_dict(json.loads(row["payload_json"]))

    def list_requests(self, limit: int = 50) -> list[DeploymentRequest]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM requests
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [DeploymentRequest.from_dict(json.loads(row["payload_json"])) for row in rows]

    def transition(
        self,
        request_id: str,
        status: RequestStatus,
        summary: str,
        *,
        raw_output_ref: str = "",
        **updates: object,
    ) -> DeploymentRequest:
        request = self.get_request(request_id)
        if request is None:
            raise KeyError(f"unknown request_id: {request_id}")

        old_status = request.status
        request.status = status
        request.updated_at = utc_now_iso()
        for key, value in updates.items():
            if not hasattr(request, key):
                raise AttributeError(f"DeploymentRequest has no field {key!r}")
            setattr(request, key, value)

        event = StateEvent(
            request_id=request_id,
            state_from=old_status.value,
            state_to=status.value,
            summary=summary,
            raw_output_ref=raw_output_ref,
        )
        self._save_request_and_event(request, event)
        return request

    def add_event(self, event: StateEvent) -> None:
        with self._connect() as conn:
            self._insert_event(conn, event)

    def events_for(self, request_id: str) -> list[StateEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT request_id, timestamp, state_from, state_to, summary, raw_output_ref
                FROM events
                WHERE request_id = ?
                ORDER BY id ASC
                """,
                (request_id,),
            ).fetchall()
        return [
            StateEvent(
                request_id=row["request_id"],
                timestamp=row["timestamp"],
                state_from=row["state_from"],
                state_to=row["state_to"],
                summary=row["summary"],
                raw_output_ref=row["raw_output_ref"] or "",
            )
            for row in rows
        ]

    def get_cursor(self, source_key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cursor_value FROM source_cursors WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        if not row:
            return None
        return str(row["cursor_value"])

    def set_cursor(self, source_key: str, cursor_value: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO source_cursors (source_key, cursor_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    cursor_value = excluded.cursor_value,
                    updated_at = excluded.updated_at
                """,
                (source_key, cursor_value, now),
            )

    def has_processed_item(self, source_key: str, item_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM processed_items
                WHERE source_key = ? AND item_id = ?
                """,
                (source_key, item_id),
            ).fetchone()
        return row is not None

    def get_processed_item(self, source_key: str, item_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT source_key, item_id, request_id, status, summary, processed_at
                FROM processed_items
                WHERE source_key = ? AND item_id = ?
                """,
                (source_key, item_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_key": row["source_key"],
            "item_id": row["item_id"],
            "request_id": row["request_id"],
            "status": row["status"],
            "summary": row["summary"],
            "processed_at": row["processed_at"],
        }

    def list_processed_items(self, source_key: str, limit: int = 500) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_key, item_id, request_id, status, summary, processed_at
                FROM processed_items
                WHERE source_key = ?
                ORDER BY processed_at DESC
                LIMIT ?
                """,
                (source_key, limit),
            ).fetchall()
        return [
            {
                "source_key": row["source_key"],
                "item_id": row["item_id"],
                "request_id": row["request_id"],
                "status": row["status"],
                "summary": row["summary"],
                "processed_at": row["processed_at"],
            }
            for row in rows
        ]

    def mark_processed_item(
        self,
        source_key: str,
        item_id: str,
        status: str,
        *,
        request_id: str = "",
        summary: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_items (
                    source_key, item_id, request_id, status, summary, processed_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_key, item_id) DO UPDATE SET
                    request_id = excluded.request_id,
                    status = excluded.status,
                    summary = excluded.summary,
                    processed_at = excluded.processed_at
                """,
                (source_key, item_id, request_id, status, summary, utc_now_iso()),
            )

    def list_todo_task_ids(self, limit: int = 100) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_key, MAX(processed_at) AS last_processed_at
                FROM processed_items
                WHERE source_key LIKE 'todo:%'
                GROUP BY source_key
                ORDER BY last_processed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        task_ids: list[str] = []
        for row in rows:
            task_id = str(row["source_key"]).removeprefix("todo:")
            if task_id:
                task_ids.append(task_id)
        return task_ids

    def delete_legacy_aggregate_task_statuses(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM runtime_settings
                WHERE key LIKE 'task_status:todo:%'
                  AND key NOT LIKE 'task_status:todo:%:item:%'
                """
            )
        return int(cursor.rowcount or 0)

    def reconcile_processed_items_from_reviews(self) -> int:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.source_key,
                    p.item_id,
                    p.request_id,
                    r.status AS review_status,
                    r.decision_json
                FROM processed_items AS p
                JOIN operator_reviews AS r ON r.review_id = p.request_id
                WHERE p.request_id != ''
                  AND p.status IN ('review_pending', 'codex_reviewing', 'approved', 'deploying', 'submitted')
                  AND r.status IN ('deployed', 'deploy_failed', 'needs_human')
                """
            ).fetchall()
            updated = 0
            for row in rows:
                decision = _json_dict(str(row["decision_json"] or "{}"))
                summary = str(
                    decision.get("summary")
                    or decision.get("execution_summary")
                    or decision.get("error")
                    or row["review_status"]
                )
                conn.execute(
                    """
                    UPDATE processed_items
                    SET status = ?, summary = ?, processed_at = ?
                    WHERE source_key = ? AND item_id = ?
                    """,
                    (
                        row["review_status"],
                        summary,
                        utc_now_iso(),
                        row["source_key"],
                        row["item_id"],
                    ),
                )
                updated += 1
        return updated

    def create_review(
        self,
        review_id: str,
        stage: str,
        subject_id: str,
        payload: dict[str, object],
        *,
        status: str = "pending",
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO operator_reviews (
                    review_id, stage, subject_id, status, payload_json, decision_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    review_id,
                    stage,
                    subject_id,
                    status,
                    json.dumps(payload, ensure_ascii=False),
                    "{}",
                    now,
                    now,
                ),
            )

    def get_review(self, review_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT review_id, stage, subject_id, status, payload_json, decision_json, created_at, updated_at
                FROM operator_reviews
                WHERE review_id = ?
                """,
                (review_id,),
            ).fetchone()
        if not row:
            return None
        return _review_row_to_dict(row)

    def list_reviews(self, limit: int = 20, *, status: str | None = None) -> list[dict[str, object]]:
        query = """
            SELECT review_id, stage, subject_id, status, payload_json, decision_json, created_at, updated_at
            FROM operator_reviews
        """
        params: tuple[object, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params = (*params, limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_review_row_to_dict(row) for row in rows]

    def decide_review(self, review_id: str, status: str, decision: dict[str, object]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE operator_reviews
                SET status = ?, decision_json = ?, updated_at = ?
                WHERE review_id = ?
                """,
                (status, json.dumps(decision, ensure_ascii=False), utc_now_iso(), review_id),
            )

    def claim_review(
        self,
        review_id: str,
        *,
        from_statuses: Iterable[str],
        to_status: str,
        decision: dict[str, object],
    ) -> bool:
        statuses = tuple(from_statuses)
        if not statuses:
            return False
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE operator_reviews
                SET status = ?, decision_json = ?, updated_at = ?
                WHERE review_id = ? AND status IN ({placeholders})
                """,
                (
                    to_status,
                    json.dumps(decision, ensure_ascii=False),
                    utc_now_iso(),
                    review_id,
                    *statuses,
                ),
            )
        return cursor.rowcount == 1

    def get_setting(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM runtime_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def set_setting(self, key: str, value: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )

    def delete_setting(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runtime_settings WHERE key = ?", (key,))

    def get_task_status(self, task_key: str) -> dict[str, object]:
        raw = self.get_setting(f"task_status:{task_key}")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def set_task_status(self, task_key: str, state: dict[str, object]) -> None:
        self.set_setting(f"task_status:{task_key}", json.dumps(state, ensure_ascii=False))

    def increment_issue_count(self, issue_key: str, summary: str = "") -> dict[str, object]:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO issue_counts (issue_key, count, summary, alerted, updated_at)
                VALUES (?, 1, ?, 0, ?)
                ON CONFLICT(issue_key) DO UPDATE SET
                    count = count + 1,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (issue_key, summary, now),
            )
            row = conn.execute(
                "SELECT issue_key, count, summary, alerted, updated_at FROM issue_counts WHERE issue_key = ?",
                (issue_key,),
            ).fetchone()
        return {
            "issue_key": row["issue_key"],
            "count": int(row["count"]),
            "summary": row["summary"],
            "alerted": bool(row["alerted"]),
            "updated_at": row["updated_at"],
        }

    def mark_issue_alerted(self, issue_key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE issue_counts SET alerted = 1, updated_at = ? WHERE issue_key = ?",
                (utc_now_iso(), issue_key),
            )

    def _save_request_and_event(self, request: DeploymentRequest, event: StateEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE requests
                SET status = ?, payload_json = ?, updated_at = ?
                WHERE request_id = ?
                """,
                (
                    request.status.value,
                    json.dumps(request.to_dict(), ensure_ascii=False),
                    request.updated_at,
                    request.request_id,
                ),
            )
            self._insert_event(conn, event)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_requests_updated_at
                ON requests(updated_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    state_from TEXT NOT NULL,
                    state_to TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    raw_output_ref TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(request_id) REFERENCES requests(request_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_cursors (
                    source_key TEXT PRIMARY KEY,
                    cursor_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_items (
                    source_key TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    request_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY(source_key, item_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operator_reviews (
                    review_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS issue_counts (
                    issue_key TEXT PRIMARY KEY,
                    count INTEGER NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    alerted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _insert_event(conn: sqlite3.Connection, event: StateEvent) -> None:
        conn.execute(
            """
            INSERT INTO events (
                request_id, timestamp, state_from, state_to, summary, raw_output_ref
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.request_id,
                event.timestamp,
                event.state_from,
                event.state_to,
                event.summary,
                event.raw_output_ref,
            ),
        )


def serialize_requests(requests: Iterable[DeploymentRequest]) -> list[dict[str, object]]:
    return [request.to_dict() for request in requests]


def _review_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "review_id": row["review_id"],
        "stage": row["stage"],
        "subject_id": row["subject_id"],
        "status": row["status"],
        "payload": _json_dict(row["payload_json"]),
        "decision": _json_dict(row["decision_json"] or "{}"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _json_dict(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
