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


def run(output, probe_only=False):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    run(**vars(parser.parse_args()))
