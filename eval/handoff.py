"""Live evidence handoff diagnosis. Actual LLM outputs, never mocked answers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.config import ROOT, settings
from src.evidence import generate_answer, render_claims
from src.integrations.llm import llm
from src.integrations.telemetry import flush, request_trace

from .run import provenance

FIXTURE = ROOT / "eval/fixtures/handoff-evidence.json"


def run(output, probe_only=False, suite=False, group="all"):
    if suite:
        return run_suite(output, group)
    fixture = json.loads(FIXTURE.read_text())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "live_llm_saved_public_evidence",
        "provenance": provenance(FIXTURE),
        "rows": [],
    }

    def save(row):
        result["rows"].append(row)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("HANDOFF_ROW=" + json.dumps(row, ensure_ascii=False), flush=True)

    for label, sources in [
        ("first_passage", fixture["sources"][:1]),
        ("six_passages", fixture["sources"]),
    ]:
        start = perf_counter()
        with request_trace(fixture["question"]) as (trace_id, obs):
            draft, evidence, feedback = generate_answer(llm(), fixture["question"], sources)
            blocks, claims, used = render_claims(draft, evidence, sources)
            row = {
                "case_id": label,
                "model": settings().llm_model,
                "question": fixture["question"],
                "trace_id": trace_id,
                "input_source_ids": [s["id"] for s in sources],
                "draft": draft.model_dump(),
                "answer": "\n\n".join(blocks),
                "claims": claims,
                "used_source_ids": [s["id"] for s in used],
                "feedback": feedback,
                "ms": round((perf_counter() - start) * 1000, 2),
            }
            obs.update(output=row)
        save(row)
    if not probe_only:
        from src.agent import ask

        start = perf_counter()
        response = ask(fixture["question"])
        save(
            {
                "case_id": "end_to_end",
                "response": response,
                "ms": round((perf_counter() - start) * 1000, 2),
            }
        )
    flush()
    print("HANDOFF_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


def run_suite(output, group="all"):
    from .counterfactual import run as counterfactual
    from .run import load_cases
    from .run import run as evaluate
    from .smoke import check_result

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = True
    if group != "counterfactual":
        data = ROOT / "eval/data/handoff.jsonl"
        if group != "all":
            comparisons = {"regression-urals-espo-grounding-003", "handoff-paraphrase"}
            selected = [
                c for c in load_cases(data) if (c["id"] in comparisons) == (group == "comparison")
            ]
            data = output.with_name("selected-cases.jsonl")
            data.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in selected))
        result = evaluate("agent", data=data, output=output)
        result["regression_checks"] = check_result(result, load_cases(data))
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("HANDOFF_SUITE=" + json.dumps(result, ensure_ascii=False), flush=True)
        passed = all(check["passed"] for check in result["regression_checks"])
    if group in {"all", "counterfactual"}:
        synthetic = output.with_name("handoff-counterfactual.json")
        counterfactual(synthetic)
        print("HANDOFF_COUNTERFACTUAL=" + synthetic.read_text().replace("\n", ""), flush=True)
    flush()
    if not passed:
        raise SystemExit("Live checks failed; preserve the results and inspect the answers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument(
        "--group", choices=["all", "comparison", "other", "counterfactual"], default="all"
    )
    run(**vars(parser.parse_args()))
