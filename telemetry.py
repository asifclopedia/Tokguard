import sqlite3
from datetime import datetime, timedelta
from typing import Any


class TelemetryStore:
    def __init__(self, db_path: str = "tokguard.db") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    estimated_output INTEGER NOT NULL,
                    saved_tokens INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    problem_category TEXT NOT NULL
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            defaults: dict[str, str] = {
                "project_name": "Tokguard",
                "project_description": "Floating Windows pre-flight prompt & token telemetry widget in Python/PyQt6.",
                "model_description": "Frontier coding LLM (128k context, structured code & diff output).",
                "session_budget": "12000",
                "encoding": "cl100k_base",
                "auto_clipboard": "1",
                "rewriter_endpoint": "http://localhost:11434/v1/chat/completions",
                "rewriter_model": "qwen2.5-coder:1.5b",
                "rewriter_api_key": "",
            }
            for k, v in defaults.items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (k, v),
                )

    def get_settings(self) -> dict[str, str]:
        cur = self.conn.cursor()
        cur.execute("SELECT key, value FROM settings")
        rows = cur.fetchall()
        result: dict[str, str] = {}
        for row in rows:
            result[str(row["key"])] = str(row["value"])
        return result

    def save_settings(self, updates: dict[str, str]) -> None:
        with self.conn:
            for k, v in updates.items():
                self.conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, v),
                )

    def log_event(
        self,
        project_name: str,
        prompt_tokens: int,
        estimated_output: int,
        saved_tokens: int,
        status: str,
        problem_category: str,
    ) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO prompt_events (
                    timestamp, project_name, prompt_tokens, estimated_output,
                    saved_tokens, status, problem_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso,
                    project_name,
                    int(prompt_tokens),
                    int(estimated_output),
                    int(saved_tokens),
                    status,
                    problem_category,
                ),
            )

    def get_weekly_series(self) -> list[dict[str, Any]]:
        """Returns 7 daily buckets (oldest to newest) with spent and saved tokens."""
        today = datetime.now().date()
        start_day = today - timedelta(days=6)
        buckets: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []

        for offset in range(7):
            d = start_day + timedelta(days=offset)
            key = d.isoformat()
            label = d.strftime("%a")
            buckets[key] = {"day": label, "spent": 0, "saved": 0}
            ordered_keys.append(key)

        cutoff = datetime.combine(start_day, datetime.min.time()).isoformat(timespec="seconds")
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT timestamp, prompt_tokens, estimated_output, saved_tokens, status
            FROM prompt_events
            WHERE timestamp >= ?
            """,
            (cutoff,),
        )
        for row in cur.fetchall():
            ts = str(row["timestamp"])
            day_key = ts[:10]
            if day_key in buckets:
                total_tok = int(row["prompt_tokens"]) + int(row["estimated_output"])
                buckets[day_key]["spent"] += total_tok
                buckets[day_key]["saved"] += int(row["saved_tokens"])

        output: list[dict[str, Any]] = []
        for k in ordered_keys:
            output.append(buckets[k])
        return output

    def get_waste_by_category(self) -> dict[str, int]:
        """Aggregates non-green / wasted token volume by problem_category."""
        categories: dict[str, int] = {
            "truncation_risk": 0,
            "cache_prefix": 0,
            "noisy_pruned": 0,
            "ambiguity_spec": 0,
            "local_offload": 0,
            "clean_ready": 0,
        }
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT problem_category, prompt_tokens, estimated_output, saved_tokens, status
            FROM prompt_events
            """
        )
        for row in cur.fetchall():
            cat = str(row["problem_category"])
            status = str(row["status"])
            tok = int(row["prompt_tokens"]) + int(row["estimated_output"]) + int(row["saved_tokens"])
            if status == "GREEN":
                categories["clean_ready"] += int(row["prompt_tokens"])
            else:
                if cat not in categories:
                    categories[cat] = 0
                categories[cat] += tok
        return categories

    def get_summary_stats(self) -> dict[str, Any]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) AS total_prompts,
                COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                COALESCE(SUM(estimated_output), 0) AS total_est_output,
                COALESCE(SUM(saved_tokens), 0) AS total_saved_tokens,
                COALESCE(SUM(CASE WHEN status != 'GREEN' THEN prompt_tokens + estimated_output ELSE 0 END), 0) AS at_risk_tokens,
                COALESCE(SUM(CASE WHEN status = 'GREEN' THEN prompt_tokens + estimated_output ELSE 0 END), 0) AS clean_tokens,
                MIN(timestamp) AS first_ts,
                MAX(timestamp) AS last_ts
            FROM prompt_events
            """
        )
        row = cur.fetchone()
        total_prompts = int(row["total_prompts"]) if row else 0
        total_prompt_tokens = int(row["total_prompt_tokens"]) if row else 0
        total_est_output = int(row["total_est_output"]) if row else 0
        total_saved_tokens = int(row["total_saved_tokens"]) if row else 0
        at_risk_tokens = int(row["at_risk_tokens"]) if row else 0
        clean_tokens = int(row["clean_tokens"]) if row else 0
        first_ts = row["first_ts"] if row else None
        last_ts = row["last_ts"] if row else None

        elapsed_minutes = 1.0
        if first_ts and last_ts:
            try:
                t1 = datetime.fromisoformat(str(first_ts))
                t2 = datetime.fromisoformat(str(last_ts))
                diff_mins = (t2 - t1).total_seconds() / 60.0
                if diff_mins >= 1.0:
                    elapsed_minutes = diff_mins
            except ValueError:
                elapsed_minutes = 1.0

        total_spend = total_prompt_tokens + total_est_output
        burn_rate = round(total_spend / elapsed_minutes, 1) if total_prompts > 0 else 0.0
        efficiency_pct = (
            round((clean_tokens / total_spend) * 100.0, 1) if total_spend > 0 else 100.0
        )

        return {
            "total_prompts": total_prompts,
            "total_prompt_tokens": total_prompt_tokens,
            "total_est_output": total_est_output,
            "total_spend": total_spend,
            "total_saved_tokens": total_saved_tokens,
            "at_risk_tokens": at_risk_tokens,
            "clean_tokens": clean_tokens,
            "burn_rate": burn_rate,
            "efficiency_pct": efficiency_pct,
        }

    def close(self) -> None:
        self.conn.close()
