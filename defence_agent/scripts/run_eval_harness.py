from __future__ import annotations

import argparse
import json

from defence_agent.evals.advanced_runner import advanced_eval_runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Defence Agent layered evaluations.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.set_defaults(func=lambda args: advanced_eval_runner.validate())

    run = sub.add_parser("run")
    run.add_argument("--suite", default="canonical")
    run.add_argument("--mode", default="fixture", choices=["fixture", "live"])
    run.add_argument("--variant", default="agentic_rag_tools")
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=lambda args: advanced_eval_runner.run_suite(args.suite, args.mode, args.variant, args.limit).model_dump())

    case = sub.add_parser("case")
    case.add_argument("--suite", default="canonical")
    case.add_argument("--query-id", required=True)
    case.add_argument("--mode", default="fixture", choices=["fixture", "live"])
    case.add_argument("--variant", default="agentic_rag_tools")
    case.set_defaults(func=lambda args: advanced_eval_runner.run_case(args.query_id, args.suite, args.mode, args.variant).model_dump())

    compare = sub.add_parser("compare")
    compare.add_argument("--suite", default="canonical")
    compare.add_argument("--limit", type=int, default=30)
    compare.set_defaults(func=lambda args: advanced_eval_runner.compare_variants(args.suite, args.limit))

    demo = sub.add_parser("select-demo")
    demo.add_argument("--suite", default="demo_candidates")
    demo.add_argument("--mode", default="fixture", choices=["fixture", "live"])
    demo.add_argument("--runs", type=int, default=3)
    demo.set_defaults(func=lambda args: advanced_eval_runner.select_demo_sequence(args.suite, args.mode, args.runs))

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
