import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from src.storage_schema import (
    API_KEY_COLUMNS,
    ARTIFACT_COLUMNS,
    CHUNK_COLUMNS,
    JOB_COLUMNS,
    JOB_IDEMPOTENCY_COLUMNS,
    PAPER_COLUMNS,
    RUNTIME_EVENT_FIELDS,
    JsonStoreSpec,
    JsonlStoreSpec,
    SqliteStoreSpec,
    SqliteTableSpec,
    format_storage_schema_markdown,
    inspect_json_store,
    inspect_jsonl_store,
    inspect_sqlite_store,
    storage_schema_status,
    storage_schema_status_for_root,
)
import src.storage_schema as storage_schema


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


def test_storage_schema_required_missing_stores_report_required_errors(tmp_path: Path):
    json_result = inspect_json_store(
        JsonStoreSpec("missing_json", tmp_path / "missing.json", ("version",), required=True)
    )
    jsonl_result = inspect_jsonl_store(
        JsonlStoreSpec("missing_jsonl", tmp_path / "missing.jsonl", ("event_id",), required=True)
    )
    sqlite_result = inspect_sqlite_store(
        SqliteStoreSpec(
            "missing_sqlite",
            tmp_path / "missing.sqlite3",
            (SqliteTableSpec("papers", PAPER_COLUMNS),),
            required=True,
        )
    )

    for result in (json_result, jsonl_result, sqlite_result):
        assert result["exists"] is False
        assert result["ok"] is False
        assert result["errors"] == ["missing_required_store"]


def test_storage_schema_reports_invalid_json_and_unreadable_jsonl(tmp_path: Path, monkeypatch):
    bad_json = tmp_path / "bad.json"
    events = tmp_path / "runtime_events.jsonl"
    bad_json.write_text("{not-json", encoding="utf-8")
    events.write_text('{"event_id": "evt-1"}\n', encoding="utf-8")

    json_result = inspect_json_store(JsonStoreSpec("bad_json", bad_json, ("version",)))
    assert json_result["errors"] == [
        "invalid_json",
        "schema_version_mismatch",
        "missing_required_keys",
    ]

    def fail_read(_path: Path, _sample_limit: int) -> list[str]:
        raise OSError("blocked")

    monkeypatch.setattr(storage_schema, "_read_jsonl_sample", fail_read)

    jsonl_result = inspect_jsonl_store(JsonlStoreSpec("events", events, ("event_id",)))
    assert jsonl_result["exists"] is True
    assert jsonl_result["sampled_events"] == 0
    assert jsonl_result["errors"] == ["unreadable_jsonl"]


def test_storage_schema_reports_missing_table_and_invalid_sqlite(tmp_path: Path):
    missing_table_db = tmp_path / "missing-table.sqlite3"
    invalid_db = tmp_path / "invalid.sqlite3"
    _create_table(missing_table_db, "other", ("id",))
    invalid_db.write_text("not sqlite", encoding="utf-8")

    missing_table = inspect_sqlite_store(
        SqliteStoreSpec(
            "missing_table",
            missing_table_db,
            (SqliteTableSpec("papers", PAPER_COLUMNS),),
        )
    )
    assert missing_table["errors"] == ["missing_required_tables", "missing_required_columns"]
    assert missing_table["missing_tables"] == ["papers"]
    assert missing_table["tables"][0]["missing_columns"] == list(PAPER_COLUMNS)

    invalid = inspect_sqlite_store(
        SqliteStoreSpec(
            "invalid_sqlite",
            invalid_db,
            (SqliteTableSpec("papers", PAPER_COLUMNS),),
        )
    )
    assert invalid["errors"] == ["invalid_sqlite"]
    assert invalid["table_count"] == 0


def test_storage_schema_for_root_and_markdown_cover_all_store_kinds(tmp_path: Path):
    metadata = tmp_path / "metadata"
    jobs = tmp_path / "jobs"
    artifacts = tmp_path / "artifacts"
    metadata.mkdir()
    jobs.mkdir()
    artifacts.mkdir()

    (metadata / "corpus.json").write_text(json.dumps({"version": 1, "papers": {}}), encoding="utf-8")
    (metadata / "corpus_profiles.json").write_text(
        json.dumps({"version": 1, "profiles": {}}),
        encoding="utf-8",
    )
    (metadata / "runtime_events.jsonl").write_text(
        json.dumps({field: "value" for field in RUNTIME_EVENT_FIELDS}) + "\n",
        encoding="utf-8",
    )
    _create_table(metadata / "corpus.sqlite3", "papers", PAPER_COLUMNS)
    _create_table(metadata / "chunks.sqlite3", "chunks", CHUNK_COLUMNS)
    _create_table(jobs / "jobs.sqlite3", "jobs", JOB_COLUMNS)
    _create_table(jobs / "jobs.sqlite3", "job_idempotency", JOB_IDEMPOTENCY_COLUMNS)
    _create_table(artifacts / "artifacts.sqlite3", "artifacts", ARTIFACT_COLUMNS)
    _create_table(metadata / "api_keys.sqlite3", "api_keys", API_KEY_COLUMNS)

    status = storage_schema_status_for_root(tmp_path)
    markdown = format_storage_schema_markdown(status)

    assert status["ok"] is True
    assert status["store_count"] == 8
    assert "kind=json" in markdown
    assert "kind=jsonl" in markdown
    assert "kind=sqlite" in markdown
    assert "sampled_events=1" in markdown
    assert "table job_idempotency" in markdown
    assert "table api_keys" in markdown
    assert str(tmp_path) not in markdown


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
