import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from src.storage_schema import (
    JOB_COLUMNS,
    PAPER_COLUMNS,
    RUNTIME_EVENT_FIELDS,
    JsonStoreSpec,
    JsonlStoreSpec,
    SqliteStoreSpec,
    SqliteTableSpec,
    storage_schema_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _create_table(db_path: Path, table_name: str, columns: tuple[str, ...]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    column_sql = ", ".join(f"{column} TEXT" for column in columns)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"CREATE TABLE {table_name} ({column_sql})")


def test_storage_schema_status_reports_valid_local_contract_without_contents(tmp_path: Path):
    corpus_json = tmp_path / "metadata" / "corpus.json"
    runtime_events = tmp_path / "metadata" / "runtime_events.jsonl"
    jobs_sqlite = tmp_path / "jobs" / "jobs.sqlite3"
    corpus_sqlite = tmp_path / "metadata" / "corpus.sqlite3"
    corpus_json.parent.mkdir(parents=True)
    corpus_json.write_text(json.dumps({"version": 1, "papers": {}}), encoding="utf-8")
    runtime_events.write_text(
        json.dumps(
            {
                "event_id": "evt-1",
                "kind": "query_usage",
                "code": "query_ok",
                "message": "ok",
                "created_at": "2026-06-08T00:00:00+00:00",
                "metadata": {"prompt": "secret question"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _create_table(corpus_sqlite, "papers", PAPER_COLUMNS)
    _create_table(jobs_sqlite, "jobs", JOB_COLUMNS)

    status = storage_schema_status(
        json_stores=(JsonStoreSpec("corpus_metadata_json", corpus_json, ("version", "papers")),),
        jsonl_stores=(JsonlStoreSpec("runtime_events_jsonl", runtime_events, RUNTIME_EVENT_FIELDS),),
        sqlite_stores=(
            SqliteStoreSpec("corpus_metadata_sqlite", corpus_sqlite, (SqliteTableSpec("papers", PAPER_COLUMNS),)),
            SqliteStoreSpec("jobs_sqlite", jobs_sqlite, (SqliteTableSpec("jobs", JOB_COLUMNS),)),
        ),
    )

    assert status["ok"] is True
    assert status["problem_count"] == 0
    assert {store["name"] for store in status["stores"]} == {
        "corpus_metadata_json",
        "runtime_events_jsonl",
        "corpus_metadata_sqlite",
        "jobs_sqlite",
    }
    dumped = json.dumps(status, ensure_ascii=False)
    assert "secret question" not in dumped
    assert str(tmp_path) not in dumped


def test_storage_schema_status_flags_version_jsonl_and_sqlite_drift(tmp_path: Path):
    corpus_json = tmp_path / "metadata" / "corpus.json"
    runtime_events = tmp_path / "metadata" / "runtime_events.jsonl"
    corpus_sqlite = tmp_path / "metadata" / "corpus.sqlite3"
    corpus_json.parent.mkdir(parents=True)
    corpus_json.write_text(json.dumps({"version": 2}), encoding="utf-8")
    runtime_events.write_text('{"event_id": "evt-1"}\nnot-json\n', encoding="utf-8")
    _create_table(corpus_sqlite, "papers", ("source_path",))

    status = storage_schema_status(
        json_stores=(JsonStoreSpec("corpus_metadata_json", corpus_json, ("version", "papers")),),
        jsonl_stores=(JsonlStoreSpec("runtime_events_jsonl", runtime_events, RUNTIME_EVENT_FIELDS),),
        sqlite_stores=(
            SqliteStoreSpec("corpus_metadata_sqlite", corpus_sqlite, (SqliteTableSpec("papers", PAPER_COLUMNS),)),
        ),
    )

    assert status["ok"] is False
    assert status["problem_count"] >= 4
    stores = {store["name"]: store for store in status["stores"]}
    assert stores["corpus_metadata_json"]["errors"] == [
        "schema_version_mismatch",
        "missing_required_keys",
    ]
    assert "invalid_jsonl_lines" in stores["runtime_events_jsonl"]["errors"]
    assert "missing_required_fields" in stores["runtime_events_jsonl"]["errors"]
    assert "missing_required_columns" in stores["corpus_metadata_sqlite"]["errors"]
    assert "paper_id" in stores["corpus_metadata_sqlite"]["tables"][0]["missing_columns"]


def test_storage_schema_cli_reports_markdown_and_fails_on_drift(tmp_path: Path):
    corpus_json = tmp_path / "metadata" / "corpus.json"
    corpus_json.parent.mkdir(parents=True)
    corpus_json.write_text(json.dumps({"version": 2}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/storage_schema.py",
            "--format",
            "markdown",
            "--target-root",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "# FluxMind Storage Schema" in result.stdout
    assert "schema_version_mismatch" in result.stdout
    assert "row contents" in result.stdout
    assert str(tmp_path) not in result.stdout
    assert result.stderr == ""
