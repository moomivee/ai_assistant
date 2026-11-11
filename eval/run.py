"""uv run python -m eval.run {validate-data,retrieval,routing,agent}."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from openai import OpenAI

from src.config import ROOT, manifest, settings
from src.integrations.llm import LLMClient
from src.integrations.prompts import PROMPT_NAMES, get_prompt
from src.integrations.telemetry import flush, langfuse_client, span
from src.logging import log
from src.models.schemas import AgentResponse, AnswerReview, EvidenceAnswer, JudgeResult
from src.planning import make_plan
from src.rag.index import MODEL_SPECS, digest, index, pdf_chunks
from src.tools import run_tool

from .metrics import mean_metrics, relevant, retrieval_metrics, route_metrics

DEFAULT_DATA = ROOT / "eval/data/cases.jsonl"


def load_cases(path=DEFAULT_DATA, split="all", limit=None):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if split == "all" or r["split"] == split][:limit]


def corpus():
    return [c for r in manifest() for c in pdf_chunks(r, settings().data / "reports" / r["file"])]


def validate_data(path=DEFAULT_DATA):
    rows, chunks = load_cases(path), corpus()
    assert rows and len({r["id"] for r in rows}) == len(rows), "Duplicate ids"
    assert len({(r["question"], r.get("previous")) for r in rows}) == len(rows), (
        "Duplicate questions"
    )
    groups = {}
    for row in rows:
        assert row["split"] in ("dev", "test")
        assert row["split"] == groups.setdefault(row["group"], row["split"]), "Group leakage"
        assert row["expected"]["answer_rubric"]
        for gold in row["relevant"]:
            assert any(relevant(c, gold) for c in chunks), f"Missing evidence: {row['id']} {gold}"
    return {
        "cases": len(rows),
        "retrieval_cases": sum(bool(r["relevant"]) for r in rows),
        "groups": len(groups),
        "chunks": len(chunks),
        "categories": {
            k: sum(r["category"] == k for r in rows) for k in sorted({r["category"] for r in rows})
        },
        "splits": {k: sum(r["split"] == k for r in rows) for k in ["dev", "test"]},
    }


def provenance(path):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("eval/**/*.py"),
            *ROOT.glob("prompts/*.txt"),
        ]
    )
    code = hashlib.sha256()
    for p in files:
        code.update(str(p.relative_to(ROOT)).encode())
        code.update(p.read_bytes())
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "code_sha256": code.hexdigest(),
        "dataset_sha256": digest(Path(path)),
        "lock_sha256": digest(ROOT / "uv.lock"),
        "corpus": {r["id"]: digest(settings().data / "reports" / r["file"]) for r in manifest()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "configuration": {
            "model": settings().llm_model,
            "agent_mode": settings().agent_mode,
            "agent_max_steps": settings().agent_max_steps,
            "agent_no_progress_limit": settings().agent_no_progress_limit,
            "agent_timeout_seconds": settings().agent_timeout_seconds,
            "llm_http_timeout": settings().llm_timeout,
            "answer_self_review": settings().answer_review,
            "embedding": settings().embedding,
            "retrieval_mode": settings().rag_search,
            "top_k": settings().top_k,
            "max_output_tokens": settings().llm_max_tokens,
        },
        "prompts": {name: get_prompt(name).version for name in PROMPT_NAMES},
    }


class RecordingClient:
    def __init__(self, client, controlled=False):
        self.client, self.controlled, self.raw_plan = client, controlled, None
        self.calls = 0

    def generate(self, name, payload, schema):
        if self.controlled and name == "cited_answer":
            evidence = payload["evidence"]
            key = next(iter(evidence), None)
            return schema.model_validate(
                {
                    "claims": [{"text": evidence[key], "evidence_ids": [key]}] if key else [],
                    "complete": bool(key),
                }
            )
        if self.controlled and name == "citation_review":
            return schema.model_validate(
                {
                    "supported": list(range(len(payload["claims"]))),
                    "complete": bool(payload["claims"]),
                    "feedback": "",
                }
            )
        if self.controlled and name == "answer_review":
            claims = payload["draft"]["claims"]
            return AnswerReview(
                checks=[{"index": i, "verdict": "supported"} for i, _ in enumerate(claims)],
                complete=bool(claims),
                repair_evidence=[],
            )
        if self.controlled and name == "answer":
            sources = payload["sources"]
            claims = [
                {"source": s["id"], "quote": s["text"], "text": s["text"]} for s in sources[:1]
            ]
            return EvidenceAnswer(complete=bool(claims), claims=claims, limitations=[])
        self.calls += 1
        result = self.client.generate(name, payload, schema)
        if name == "plan":
            self.raw_plan = result
        return result


class RecordingPlanner:
    def __init__(self):
        self.plan = None

    def __call__(self, client, question, previous=None):
        self.plan = make_plan(client, question, previous)
        return self.plan


class RecordingExecutor:
    def __init__(self, execute):
        self.execute, self.retrievals = execute, []

    def __call__(self, registry, name, **args):
        result = self.execute(registry, name, **args)
        if name in {"report_search", "read_report"}:
            self.retrievals.append({"tool": name, "arguments": args, "sources": result[0] or []})
        return result


def controlled_executor(case):
    """Only a routing harness: retrieval/answer quality must be measured in agent mode."""
    from src.calculations.forecast import forecast

    reports = manifest()

    def execute(registry, name, **args):
        start = perf_counter()
        with span(name, "tool", input=args, metadata={"controlled_fixture": True}) as obs:
            if name in {"report_search", "read_report"}:
                data = []
                for i, g in enumerate(case["relevant"], 1):
                    meta = next(r for r in reports if r["id"] == g["report"])
                    data.append(
                        {
                            **g,
                            "id": f"R{i}",
                            "kind": "report",
                            "title": meta["title"],
                            "date": meta["date"],
                            "url": meta["url"],
                            "text": g["quote"],
                        }
                    )
            elif name == "web_search":
                data = {
                    "items": [],
                    "notices": [
                        "Контролируемый веб-инструмент eval: свежие источники не запрашивались."
                    ],
                }
            else:
                data = forecast(**args)
            obs.update(output=data)
        return (
            data,
            None,
            {
                "tool": name,
                "engine": "eval.controlled_fixture",
                "status": "ok",
                "ms": round((perf_counter() - start) * 1000),
            },
        )

    return execute


def make_judge():
    cfg = settings()
    key = os.getenv("JUDGE_API_KEY", cfg.llm_key)
    if not key:
        raise ValueError("Set JUDGE_API_KEY or LLM_API_KEY")
    return LLMClient(
        OpenAI(
            api_key=key,
            base_url=os.getenv("JUDGE_BASE_URL", cfg.llm_base),
            timeout=cfg.llm_timeout,
            max_retries=0,
        ),
        model=os.getenv("JUDGE_MODEL", cfg.llm_model),
    )


def score_trace(trace_id, scores):
    lf = langfuse_client()
    if lf:
        for name, value in scores.items():
            try:
                lf.create_score(
                    trace_id=trace_id,
                    name="eval/" + name,
                    value=float(value),
                    data_type="NUMERIC",
                )
            except Exception as exc:
                log("eval.score_failed", error_type=type(exc).__name__)


def run(
    mode,
    *,
    data=DEFAULT_DATA,
    split="all",
    models=None,
    limit=None,
    judge=False,
    output=None,
):
    if mode == "validate-data":
        return validate_data(data)
    if mode == "routing" and judge:
        raise ValueError("Judge is unavailable for controlled routing fixtures")
    output = Path(
        output or ROOT / ".cache/eval" / f"{mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".jsonl")
    if output.exists() or partial.exists():
        raise FileExistsError("Use a new output path")
    cases = load_cases(data, split)
    if mode == "retrieval":
        cases = [r for r in cases if r["relevant"]]
    cases = cases[:limit]
    if not cases:
        raise ValueError("No cases selected")
    result = {
        "mode": mode,
        "scope": "controlled_routing" if mode == "routing" else mode,
        "provenance": provenance(data),
        "split": split,
        "rows": [],
    }

    def save(row):
        result["rows"].append(row)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if mode == "retrieval":
        for model in models or list(MODEL_SPECS):
            idx = index(model)
            for variant in ["clean_query", "full_question"]:
                for case in cases:
                    start = perf_counter()
                    ranked = idx.search(
                        case["retrieval_query"] if variant == "clean_query" else case["question"],
                        10,
                    )
                    save(
                        {
                            "case_id": case["id"],
                            "split": case["split"],
                            "model": model,
                            "variant": variant,
                            "fingerprint": idx.fingerprint,
                            "metrics": retrieval_metrics(ranked, case["relevant"]),
                            "ms": round((perf_counter() - start) * 1000, 2),
                            "ranking": [
                                {"chunk_id": c["chunk_id"], "score": c["score"]} for c in ranked
                            ],
                        }
                    )
        result["summary"] = {}
        for model in models or list(MODEL_SPECS):
            for variant in ["clean_query", "full_question"]:
                for subset in ["all", "dev", "test"]:
                    rows = [
                        r
                        for r in result["rows"]
                        if r["model"] == model
                        and r["variant"] == variant
                        and (subset == "all" or r["split"] == subset)
                    ]
                    result["summary"][f"{model}/{variant}/{subset}"] = {
                        "n": len(rows),
                        **mean_metrics([r["metrics"] for r in rows]),
                    }
    else:
        from src.agent import ask

        judge_client = make_judge() if judge else None
        result["model"] = settings().llm_model
        result["agent_mode"] = settings().agent_mode
        result["judge_model"] = judge_client.model if judge_client else None
        result["judge_same_model"] = bool(
            judge_client and judge_client.model == settings().llm_model
        )
        for case in cases:
            client, start = (
                RecordingClient(LLMClient(), controlled=mode == "routing"),
                perf_counter(),
            )
            executor = RecordingExecutor(
                controlled_executor(case) if mode == "routing" else run_tool
            )
            kwargs = {"execute": executor}
            recorder = RecordingPlanner()
            error = None
            try:
                response = ask(
                    case["question"],
                    case.get("previous"),
                    client=client,
                    planner=recorder,
                    **kwargs,
                )
                AgentResponse.model_validate(response)
            except Exception as exc:
                error = type(exc).__name__
                response = {
                    "question": case["question"],
                    "answer": "Evaluation request failed.",
                    "status": "unavailable",
                    "intent": "unknown",
                    "mode": settings().llm_model,
                    "sources": [],
                    "claims": [],
                    "trace": [],
                    "trace_id": "",
                    "forecast": None,
                    "agent_mode": settings().agent_mode,
                }
            plan = recorder.plan.model_dump() if recorder.plan else None
            metrics = route_metrics(case["expected"], response, plan)
            row = {
                "case_id": case["id"],
                "split": case["split"],
                "metrics": metrics,
                "plan": plan,
                "initial_plan": client.raw_plan.model_dump() if client.raw_plan else None,
                "response": response,
                "llm_calls": client.calls,
                "ms": round((perf_counter() - start) * 1000, 2),
                "answer_quality_measured": bool(judge_client),
                "execution_error": error,
                "contract_valid": error is None,
                "retrieval_steps": [
                    {
                        "tool": step["tool"],
                        "arguments": step["arguments"],
                        "chunk_ids": [s.get("chunk_id") for s in step["sources"]],
                        "metrics": retrieval_metrics(step["sources"], case["relevant"])
                        if case["relevant"]
                        else None,
                    }
                    for step in executor.retrievals
                ],
            }
            # Status depends on evidence availability in fixtures and is not answer quality.
            if mode == "routing":
                row["metrics"].pop("status_correct")
            if judge_client:
                try:
                    grade = judge_client.generate(
                        "judge",
                        {
                            "question": case["question"],
                            "rubric": case["expected"]["answer_rubric"],
                            "reference_contexts": case["relevant"],
                            "response": {k: response[k] for k in ["answer", "status", "forecast"]},
                            "evidence": response["sources"],
                        },
                        JudgeResult,
                    )
                    row["judge"] = grade.model_dump()
                    score_trace(
                        response["trace_id"],
                        {k: v for k, v in row["judge"].items() if k != "explanation"},
                    )
                except Exception as exc:
                    row["judge_error"] = type(exc).__name__
            score_trace(response["trace_id"], row["metrics"])
            save(row)
        result["summary"] = {
            "n": len(result["rows"]),
            **mean_metrics([r["metrics"] for r in result["rows"]]),
            "response_contract_valid_rate": sum(r["contract_valid"] for r in result["rows"])
            / len(result["rows"]),
            "llm_plan_valid_rate": sum(r["plan"] is not None for r in result["rows"])
            / len(result["rows"]),
            "judge_coverage": sum("judge" in r for r in result["rows"]) / len(result["rows"]),
            "task_success_rate": None,
            "judge_calibrated": False,
            "judge_assisted_success_rate": sum(
                r["metrics"]["tool_policy_pass"] == 1
                and r["metrics"]["intent_correct"] == 1
                and r["metrics"]["status_correct"] == 1
                and "judge" in r
                and min(r["judge"][k] for k in ("groundedness", "relevance", "completeness")) >= 3
                for r in result["rows"]
            )
            / len(result["rows"])
            if judge_client
            else None,
        }
        result["judge_summary"] = mean_metrics(
            [
                {k: v for k, v in r["judge"].items() if k != "explanation"}
                for r in result["rows"]
                if "judge" in r
            ]
        )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flush()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["validate-data", "retrieval", "routing", "agent"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--models", nargs="+", choices=list(MODEL_SPECS))
    p.add_argument("--limit", type=int)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--output", type=Path)
    args = vars(p.parse_args())
    result = run(**args)
    print(json.dumps(result.get("summary", result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
