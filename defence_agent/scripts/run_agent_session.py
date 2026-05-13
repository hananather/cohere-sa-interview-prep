#!/usr/bin/env python3
"""Run one or two turns through the ADK agent with a persistent SQLite session."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from defence_agent.auth.context import DEFAULT_PERSONA_ID
from defence_agent.session import default_session_db_path, run_turn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="User question for the first turn.")
    parser.add_argument("--follow-up", help="Optional follow-up question in the same session.")
    parser.add_argument(
        "--persona",
        default=DEFAULT_PERSONA_ID,
        help="Persona id: clearance_unclassified, clearance_secret, or clearance_top_secret.",
    )
    parser.add_argument("--user-id", help="User id. Defaults to the persona id.")
    parser.add_argument("--session-id", help="Existing session id to continue.")
    parser.add_argument(
        "--target-answer-language",
        default="auto",
        choices=("auto", "en", "fr"),
        help="Final answer language. Defaults to the user's query language.",
    )
    parser.add_argument("--show-audit", action="store_true", help="Print structured answer audit JSON.")
    return parser


async def _main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    first = await run_turn(
        args.query,
        persona_id=args.persona,
        user_id=args.user_id,
        session_id=args.session_id,
        target_answer_language=args.target_answer_language,
    )
    _print_result("Turn 1", first, show_audit=args.show_audit)
    if args.follow_up:
        second = await run_turn(
            args.follow_up,
            persona_id=args.persona,
            user_id=args.user_id,
            session_id=first.session_id,
            target_answer_language=args.target_answer_language,
        )
        _print_result("Turn 2", second, show_audit=args.show_audit)
    print(f"\nSession DB: {default_session_db_path()}")


def _print_result(label: str, result: object, *, show_audit: bool = False) -> None:
    print(f"\n== {label} ==")
    print(f"session_id: {result.session_id}")
    print(f"user_id: {result.user_id}")
    print(f"persona_id: {result.persona_id}")
    print(f"events_seen: {result.events_seen}")
    print(f"tool_calls: {', '.join(result.tool_calls) or 'none'}")
    print(f"citation_mode: {result.citation_mode}")
    print(f"citations: {len(result.citations)}")
    print(f"citation_validation: {result.citation_validation}")
    print(f"grounded_model: {result.grounded_model or 'none'}")
    print(f"documents_sent_to_model: {result.documents_sent_to_model}")
    print(f"retrieval_status: {result.retrieval_status or 'none'}")
    print(f"answer:\n{result.answer}")
    if show_audit:
        print("answer_audit:")
        print(json.dumps(result.answer_audit, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
