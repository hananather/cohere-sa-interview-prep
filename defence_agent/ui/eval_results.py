"""Load saved demo registry transcripts for the Streamlit Eval tab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRANSCRIPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "transcripts"
PREFERRED_READINESS_RUN = "live_readiness_final_20260511_013937"
INSUFFICIENT_EVIDENCE_READINESS_RUN = "insufficient_evidence_strategy_realignment_20260511_131708"
PRESENTATION_READINESS_RUNS = (
    PREFERRED_READINESS_RUN,
    INSUFFICIENT_EVIDENCE_READINESS_RUN,
)


@dataclass(frozen=True)
class TranscriptRunOption:
    """One selectable transcript set for the Eval tab."""

    label: str
    paths: tuple[Path, ...]
    is_bundle: bool = False


def latest_transcript_dir(root: Path = DEFAULT_TRANSCRIPTS_DIR) -> Path | None:
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def preferred_transcript_dir(
    root: Path = DEFAULT_TRANSCRIPTS_DIR,
    *,
    preferred_name: str = PREFERRED_READINESS_RUN,
) -> Path | None:
    preferred = root / preferred_name
    if preferred.is_dir():
        return preferred
    return latest_transcript_dir(root)


def transcript_dirs(root: Path = DEFAULT_TRANSCRIPTS_DIR) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def transcript_run_options(root: Path = DEFAULT_TRANSCRIPTS_DIR) -> list[TranscriptRunOption]:
    dirs = transcript_dirs(root)
    options: list[TranscriptRunOption] = []
    bundle_paths = tuple(
        root / name
        for name in PRESENTATION_READINESS_RUNS
        if (root / name).is_dir()
    )
    if len(bundle_paths) > 1:
        options.append(TranscriptRunOption("Presentation readiness bundle", bundle_paths, is_bundle=True))
    options.extend(TranscriptRunOption(path.name, (path,)) for path in dirs)
    return options


def transcript_run_label(transcript_dir: Path | TranscriptRunOption) -> str:
    rows = load_eval_rows(transcript_dir)
    total = len(rows)
    passed = sum(1 for row in rows if row.get("passed"))
    name = transcript_dir.label if isinstance(transcript_dir, TranscriptRunOption) else transcript_dir.name
    return f"{name} · {passed}/{total} passed"


def load_eval_rows(transcript_dir: Path | TranscriptRunOption | None) -> list[dict[str, Any]]:
    if transcript_dir is None:
        return []
    if isinstance(transcript_dir, TranscriptRunOption):
        rows: list[dict[str, Any]] = []
        for path in transcript_dir.paths:
            rows.extend(load_eval_rows(path))
        return rows
    if not transcript_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(transcript_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(_row_from_outcome(data, transcript_run=transcript_dir.name))
    return rows


def filter_eval_rows(rows: list[dict[str, Any]], filter_name: str) -> list[dict[str, Any]]:
    if filter_name == "failed only":
        return [row for row in rows if not row.get("passed")]
    if filter_name == "ACL":
        return [row for row in rows if _contains_any(row, ("acl", "access", "clearance", "restricted"))]
    if filter_name == "citation":
        return [row for row in rows if _contains_any(row, ("citation", "cite", "source"))]
    if filter_name == "refusal":
        return [row for row in rows if _contains_any(row, ("refusal", "refuse", "insufficient"))]
    if filter_name == "bilingual":
        return [
            row
            for row in rows
            if _contains_any(row, ("bilingual", "multilingual", "french", "francais", "français"))
        ]
    return rows


def _row_from_outcome(data: dict[str, Any], *, transcript_run: str = "") -> dict[str, Any]:
    final_audit = data.get("final_answer_audit", {}) if isinstance(data.get("final_answer_audit"), dict) else {}
    retrieval = final_audit.get("retrieval", {}) if isinstance(final_audit.get("retrieval"), dict) else {}
    generation = final_audit.get("generation", {}) if isinstance(final_audit.get("generation"), dict) else {}
    answerability = _answerability_label(retrieval)
    answerability_reason = _answerability_reason(retrieval)
    documents_sent = int(generation.get("document_count", 0) or 0)
    return {
        "transcript_run": transcript_run,
        "case_id": data.get("case_id", ""),
        "persona": data.get("persona_id", ""),
        "query": data.get("query", ""),
        "demo_point": data.get("demo_point", ""),
        "passed": bool(data.get("passed", False)),
        "failures": "; ".join(str(item) for item in data.get("failures", []) or []),
        "search_count": data.get("search_count", 0),
        "documents_found": ", ".join(data.get("doc_ids", []) or []),
        "excluded_documents": _excluded_document_summary(data.get("excluded_doc_ids", [])),
        "documents_sent_to_model": documents_sent,
        "citation_count": data.get("citation_count", 0),
        "citation_mode": data.get("mode", generation.get("citation_mode", "")),
        "citation_quality": _citation_quality_label(
            data.get("citation_quality", generation.get("citation_quality", {}))
        ),
        "final_answerability": answerability,
        "answerability_reason": answerability_reason,
        "zero_doc_refusal": answerability == "refused" and documents_sent == 0,
    }


def _answerability_label(retrieval: dict[str, Any]) -> str:
    records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    if any(item.get("answerable") is False for item in records):
        return "refused"
    if any(item.get("answerable") is True for item in records):
        return "answered"
    if retrieval.get("sources_sent_to_answer"):
        return "answered"
    return "not explicit"


def _answerability_reason(retrieval: dict[str, Any]) -> str:
    records = [item for item in retrieval.get("answerability", []) or [] if isinstance(item, dict)]
    if not records:
        return ""
    return str(records[0].get("reason", "") or "")


def _citation_quality_label(value: Any) -> str:
    if not isinstance(value, dict):
        return "not reported"
    raw = value.get("citation_recall_proxy")
    try:
        return "not reported" if raw is None else f"{float(raw):.2f} span coverage proxy"
    except (TypeError, ValueError):
        return "not reported"


def _excluded_document_summary(value: Any) -> str:
    count = len(value) if isinstance(value, list) else 0
    return f"{count} excluded" if count else "0 excluded"


def _contains_any(row: dict[str, Any], terms: tuple[str, ...]) -> bool:
    text = " ".join(str(value).lower() for value in row.values())
    return any(term in text for term in terms)
