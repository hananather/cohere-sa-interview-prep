from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_model_upgrade_runner_dry_run_supports_safe_selectors() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "defence_agent/scripts/run_model_upgrade_eval_matrix.py",
            "--phase",
            "full",
            "--timestamp",
            "pytest",
            "--models",
            "plus",
            "--route",
            "simple_rag",
            "--case-id",
            "model_upgrade_bilingual_cross_language_citations",
            "--skip-existing",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Command A Plus / simple_rag" in result.stdout
    assert "Command A / simple_rag" not in result.stdout
    assert "--skip-existing" in result.stdout
    assert "--case-id model_upgrade_bilingual_cross_language_citations" in result.stdout


def test_model_upgrade_runner_coverage_only_reports_missing_without_live_call() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "defence_agent/scripts/run_model_upgrade_eval_matrix.py",
            "--phase",
            "full",
            "--timestamp",
            "pytest_missing",
            "--models",
            "plus",
            "--route",
            "reviewed_agent",
            "--case-id",
            "model_upgrade_bilingual_cross_language_citations",
            "--coverage-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Command A Plus | reviewed_agent | 0 | 1" in result.stdout
    assert "model_upgrade_bilingual_cross_language_citations" in result.stdout


def test_registry_runner_skip_existing_reuses_transcript_without_live_call(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
cases:
  - id: already_done
    persona_id: clearance_unclassified
    query: Already done?
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "transcripts"
    output_dir.mkdir()
    (output_dir / "already_done.json").write_text(
        json.dumps(
            {
                "case_id": "already_done",
                "passed": True,
                "failures": [],
                "persona_id": "clearance_unclassified",
                "routing": {"selected_mode": "simple_rag"},
                "search_count": 1,
                "doc_ids": ["DOC-1"],
                "excluded_doc_ids": [],
                "citation_count": 1,
                "mode": "cohere_native_accurate_default",
                "critic": {"status": "not_run", "score": None, "gate": "not_applicable"},
                "demo_point": "",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "defence_agent/scripts/run_demo_query_registry.py",
            "--registry",
            str(registry_path),
            "--run-mode",
            "simple_rag",
            "--output-dir",
            str(output_dir),
            "--case-id",
            "already_done",
            "--skip-existing",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "already_done | SKIP" in result.stdout


def test_model_upgrade_finalizer_dry_run_reports_coverage_and_outputs(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
cases:
  - id: case_one
    persona_id: clearance_unclassified
    query: Case one?
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "defence_agent/scripts/finalize_model_upgrade_eval_matrix.py",
            "--timestamp",
            "pytest",
            "--registry",
            str(registry_path),
            "--output-root",
            str(tmp_path / "transcripts"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Coverage status: incomplete" in result.stdout
    assert "Expected cases per route/model: 1" in result.stdout
    assert "command_a_plus_model_upgrade_comparison_report.json" in result.stdout
    assert not (tmp_path / "reports").exists()


def test_model_upgrade_finalizer_refuses_incomplete_matrix_without_allow_partial(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
cases:
  - id: case_one
    persona_id: clearance_unclassified
    query: Case one?
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "defence_agent/scripts/finalize_model_upgrade_eval_matrix.py",
            "--timestamp",
            "pytest",
            "--registry",
            str(registry_path),
            "--output-root",
            str(tmp_path / "transcripts"),
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Coverage status: incomplete" in result.stdout
    assert "Refusing to write model-upgrade reports" in result.stderr
