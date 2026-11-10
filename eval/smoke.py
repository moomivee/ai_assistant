"""Live regression smoke: actual configured LLM, RAG and tools; no fixtures."""

import argparse
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.config import ROOT
from src.validation import normalized, numbers

from .run import load_cases, run

DATA = ROOT / "eval/data/regressions.jsonl"


def check_result(result, cases):
    expectations = {case["id"]: case["expected"] for case in cases}
    checks = []
    for row in result["rows"]:
        expected, response = expectations[row["case_id"]], row["response"]
        route_ok = (
            all(
                row["metrics"][key] == 1
                for key in (
                    "tool_sequence_exact",
                    "intent_correct",
                    "status_correct",
                    "plan_fields_correct",
                )
            )
            and row["metrics"]["forbidden_tool_violation"] == 0
        )
        facts = []
        for fact in expected.get("answer_facts", []):
            aliases = [normalized(name) for name in fact.get("aliases", [fact["entity"]])]
            values = set()
            for claim in response["claims"]:
                if any(name in normalized(claim["text"]) for name in aliases):
                    values.update(numbers(claim["text"]))
            facts.append(
                {
                    "entity": fact["entity"],
                    "covered": {Decimal(str(v)) for v in fact["values"]} <= values,
                }
            )
        checks.append(
            {
                "case_id": row["case_id"],
                "route_ok": route_ok,
                "facts": facts,
                "passed": route_ok and all(f["covered"] for f in facts),
            }
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/eval" / f"live-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json",
    )
    args = parser.parse_args()
    result = run("agent", data=DATA, output=args.output)
    result["regression_checks"] = check_result(result, load_cases(DATA))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    # Known public regression questions only. Preserve outputs in CI logs for review.
    print("LIVE_EVAL_RESULT=" + json.dumps(result, ensure_ascii=False))
    if not all(check["passed"] for check in result["regression_checks"]):
        raise SystemExit("Live regression checks failed; inspect saved answers and traces.")


if __name__ == "__main__":
    main()
