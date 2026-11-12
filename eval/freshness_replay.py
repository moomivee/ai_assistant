"""One real model decision from the reported stale-report context; no live web calls."""

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

from src.agent import action_schema, allowed_actions, decision_sources, research_status
from src.budget import remaining_seconds, request_budget, research_phase
from src.config import ROOT, settings
from src.integrations.llm import LLMClient
from src.integrations.telemetry import flush

from .run import provenance

FIXTURE = ROOT / "eval/fixtures/latest-meeting.json"


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output path")
    fixture = json.loads(FIXTURE.read_text())
    state = {
        **fixture,
        "trace": [{"tool": "report_search"}],
        "calc": None,
    }
    actions = allowed_actions(state)
    cfg = settings()
    result = {
        "scope": "isolated_decision_replay",
        "provenance": provenance(FIXTURE),
        "fixture_provenance": fixture["provenance"],
        "model": cfg.llm_model,
        "expected_action": "web_search",
        "answer_quality_measured": False,
        "live_web_measured": False,
    }
    start = perf_counter()
    with request_budget(cfg.agent_timeout_seconds, cfg.agent_answer_reserve_seconds):
        with research_phase():
            payload = {
                "question": fixture["question"],
                "previous": None,
                "task": fixture["plan"],
                "today": date.today().isoformat(),
                "allowed_actions": actions,
                "research_status": research_status(state),
                "steps_remaining": cfg.agent_max_steps - 1,
                "observations": fixture["observations"],
                "sources": decision_sources(fixture["sources"]),
                "calculation_available": False,
                "answer_feedback": fixture["feedback"],
                "current_answer": fixture["draft"],
                "no_progress_steps": 0,
                "no_progress_limit": cfg.agent_no_progress_limit,
                "seconds_remaining": remaining_seconds(),
                "previous_assessment": {
                    "assessment": "need_evidence",
                    "missing_facts": [],
                    "evidence_checks": [],
                    "unknown_source_ids": [],
                },
            }
            result["input"] = payload
            try:
                decision = LLMClient().generate("decide", payload, action_schema(tuple(actions)))
                result["decision"] = decision.model_dump()
                result["passed"] = decision.action == result["expected_action"]
            except Exception as exc:
                result.update(passed=False, error=type(exc).__name__)
    result["ms"] = round((perf_counter() - start) * 1000, 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    flush()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({k: result[k] for k in ("scope", "passed", "ms")}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit("Model did not choose the required freshness check")


if __name__ == "__main__":
    main()
