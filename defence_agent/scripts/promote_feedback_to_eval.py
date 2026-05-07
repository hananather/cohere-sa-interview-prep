from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT / "data" / "feedback" / "feedback_events.jsonl"
OUTPUT_PATH = ROOT / "data" / "evals" / "feedback_draft_eval_cases.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft regression eval cases from negative feedback.")
    parser.add_argument("--input", default=str(FEEDBACK_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    if input_path.exists():
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("rating") not in {"no", "somewhat"} and event.get("helpful") is not False:
                continue
            query = event.get("user_query") or ""
            if not query:
                continue
            cases.append(
                {
                    "query_id": f"FEEDBACK_DRAFT_{len(cases) + 1:03d}",
                    "user_query": query,
                    "task_type": "demo_candidate",
                    "topic": "feedback",
                    "capability_tags": ["feedback", "regression_candidate"],
                    "complexity_level": "L3",
                    "complexity_score": 5,
                    "complexity_dimensions": ["citation_density_required"],
                    "user_context": {"user_id": "alex_analyst", "role": "planning_analyst", "access_level": "public_internal", "language": "en"},
                    "expected_route": event.get("route") or "evidence_lookup",
                    "expected_tools": event.get("tools_called") or [],
                    "disallowed_tools": [],
                    "expected_sources": [],
                    "disallowed_sources": [],
                    "expected_key_facts": [],
                    "forbidden_claims": [],
                    "expected_answer_behavior": "human_review_required_before_regression",
                    "expected_refusal": False,
                    "citation_required": bool(event.get("citations")),
                    "exact_expected_result": None,
                    "grader_config": {"route": True, "retrieval": False, "filter": False, "answer_key_facts": False, "citation_validation": bool(event.get("citations")), "llm_judge": False},
                    "demo_notes": f"Drafted from feedback trace {event.get('trace_id')}; human review required before promotion.",
                }
            )
    output_path.write_text(yaml.safe_dump({"suite": "feedback_draft", "generated_by": "scripts/promote_feedback_to_eval.py", "cases": cases}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"draft_cases": len(cases), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
