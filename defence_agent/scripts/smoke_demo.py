from __future__ import annotations

import argparse
import sys
from typing import Any

import requests


DEMO_QUERIES = [
    ("planning_analyst", "What review steps are required before a planning brief is approved?"),
    ("planning_analyst", "What is the current approved procedure for approving a planning brief? Do not use drafts or old versions."),
    ("planning_analyst", "What changed between the 2024 and 2025 planning-brief review process? Cite both versions."),
    ("planning_analyst", "Which planning procedures are overdue for review? Group them by owner and show how many days overdue."),
    ("planning_analyst", "What restricted annex handling steps apply before external distribution?"),
    ("planning_lead", "What restricted annex handling steps apply before external distribution?"),
    ("planning_analyst", "Quels sont les délais dans la procédure de communications d’urgence?"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Defence Agent demo smoke checks.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    base = args.api_url.rstrip("/")
    health = requests.get(f"{base}/health", timeout=10)
    health.raise_for_status()
    print("health", health.json())

    failures: list[str] = []
    for persona, query in DEMO_QUERIES:
        response = requests.post(
            f"{base}/v1/agent/query",
            headers={"X-Demo-User": persona},
            json={"query": query},
            timeout=45,
        )
        if not response.ok:
            failures.append(f"{persona}: {response.status_code} {response.text[:160]}")
            continue
        payload: dict[str, Any] = response.json()
        print(persona, payload["route"], payload["trace_id"], len(payload.get("sources", [])))
        if not payload.get("trace_id"):
            failures.append(f"{persona}: missing trace_id")
        if payload["route"] not in {"ambiguous_query", "human_review"} and payload.get("sources") and not payload.get("citations"):
            failures.append(f"{persona}: missing citations for {payload['route']}")

    if failures:
        print("FAILURES")
        for failure in failures:
            print("-", failure)
        return 1
    print("smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
