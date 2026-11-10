"""Score saved answers with an explicitly selected, calibrated second model."""

import argparse
import json
import os
from pathlib import Path

from src.config import ROOT
from src.models.schemas import JudgeResult

from .run import load_cases, make_judge, provenance


def acceptable(grade):
    return min(grade[k] for k in ("groundedness", "relevance", "completeness", "tone")) >= 3


def assess(results, output, data):
    if Path(output).exists():
        raise FileExistsError("Choose a new output path to preserve previous measurements.")
    if not os.getenv("JUDGE_MODEL"):
        raise ValueError("Set JUDGE_MODEL explicitly; self-review is not independent evaluation.")
    client = make_judge()
    controls = load_cases(ROOT / "eval/data/judge_controls.jsonl")
    calibration = []
    for case in controls:
        grade = client.generate("judge", case["payload"], JudgeResult).model_dump()
        calibration.append(
            {
                "id": case["id"],
                "expected_acceptable": case["acceptable"],
                "grade": grade,
                "passed": acceptable(grade) == case["acceptable"],
            }
        )
    calibrated = all(row["passed"] for row in calibration)
    if not calibrated:
        result = {
            "judge_model": client.model,
            "calibration_passed": False,
            "calibration": calibration,
            "rows": [],
            "task_success_rate": None,
            "warning": "Scoring skipped: the judge failed calibration controls.",
            "provenance": provenance(ROOT / "eval/data/judge_controls.jsonl"),
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"calibrated": False, "n": 0, "task_success_rate": None}))
        return
    gold = {row["id"]: row for path in data for row in load_cases(path)}
    rows = []
    for path in results:
        source = json.loads(Path(path).read_text())
        if source["model"] == client.model:
            raise ValueError(
                "Judge must differ from the answering model; use another model or report manual review."
            )
        for row in source["rows"]:
            case = gold[row["case_id"]]
            response = row["response"]
            try:
                grade = client.generate(
                    "judge",
                    {
                        "question": case["question"],
                        "rubric": case["expected"]["answer_rubric"],
                        "reference_contexts": case["relevant"],
                        "response": {k: response[k] for k in ["answer", "status", "forecast"]},
                        "evidence": response["sources"],
                    },
                    JudgeResult,
                ).model_dump()
                passed = (
                    calibrated
                    and acceptable(grade)
                    and row.get("contract_valid", False)
                    and all(
                        row["metrics"][key] == 1
                        for key in (
                            "tool_policy_pass",
                            "required_tool_success",
                            "status_correct",
                            "intent_correct",
                            "plan_fields_correct",
                        )
                    )
                )
                rows.append(
                    {
                        "file": str(path),
                        "case_id": row["case_id"],
                        "grade": grade,
                        "passed": passed if calibrated else None,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": str(path),
                        "case_id": row["case_id"],
                        "error": type(exc).__name__,
                        "passed": False,
                    }
                )
    result = {
        "judge_model": client.model,
        "calibration_passed": calibrated,
        "calibration": calibration,
        "rows": rows,
        "task_success_rate": sum(bool(r["passed"]) for r in rows) / len(rows)
        if calibrated and rows
        else None,
        "warning": "A model judge remains fallible. Calibration controls and all raw grades are retained.",
        "provenance": provenance(ROOT / "eval/data/judge_controls.jsonl"),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "calibrated": calibrated,
                "n": len(rows),
                "task_success_rate": result["task_success_rate"],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--data",
        nargs="+",
        type=Path,
        default=[
            ROOT / "eval/data/cases.jsonl",
            ROOT / "eval/data/regressions.jsonl",
            ROOT / "eval/data/audit.jsonl",
        ],
    )
    assess(**vars(p.parse_args()))
