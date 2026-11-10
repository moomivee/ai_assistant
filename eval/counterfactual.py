"""Actual LLM + synthetic tool evidence: test whether answers follow changed data."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.agent import ask
from src.config import ROOT, settings

from .run import provenance

QUESTION = "Сравни тестовые сорта нефти Orion и Vega по API и сере. Какой легче и менее сернистый? Используй только загруженный тестовый отчёт."


def run(output):
    if Path(output).exists() or Path(output).with_suffix(".jsonl").exists():
        raise FileExistsError("Choose a new output path to preserve previous measurements.")
    rows = []
    for key, table, expected, empty_first in [
        (
            "original",
            "Orion | 42 to 44 | 0.1% to 0.2%\nVega | 27 to 29 | 1.5% to 1.7%",
            "Orion",
            False,
        ),
        (
            "swapped",
            "Orion | 27 to 29 | 1.5% to 1.7%\nVega | 42 to 44 | 0.1% to 0.2%",
            "Vega",
            False,
        ),
        (
            "empty-first",
            "Orion | 42 to 44 | 0.1% to 0.2%\nVega | 27 to 29 | 1.5% to 1.7%",
            "Orion",
            True,
        ),
    ]:
        source = {
            "id": "R1",
            "report": "synthetic-evaluation",
            "kind": "report",
            "title": "Synthetic evaluation fixture, not real market data",
            "date": "2026-11-10",
            "page": 1,
            "url": "https://example.invalid/synthetic",
            "chunk_id": "synthetic:page1",
            "text": "Test crude oil samples.\nGrade | API gravity (degrees API) | Sulfur content (wt%)\n"
            + table
            + "\nHigher API gravity means lower physical density. Sulfur content is a separate property.",
        }

        calls = []

        def execute(registry, name, **args):
            calls.append({"tool": name, "arguments": args})
            if name not in {"report_search", "read_report"}:
                return (
                    None,
                    "Only the synthetic report is available.",
                    {"tool": name, "engine": "eval.synthetic_evidence", "status": "error", "ms": 0},
                )
            return (
                []
                if empty_first and len(calls) == 1
                else [{**source, "full_page": name == "read_report"}],
                None,
                {"tool": name, "engine": "eval.synthetic_evidence", "status": "ok", "ms": 0},
            )

        start = perf_counter()
        response = ask(QUESTION, execute=execute)
        rows.append(
            {
                "case_id": "counterfactual-" + key,
                "question": QUESTION,
                "expected_lighter_and_less_sulfur": expected,
                "empty_first_search": empty_first,
                "tool_calls": calls,
                "source": source,
                "response": response,
                "ms": round((perf_counter() - start) * 1000, 2),
            }
        )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with Path(output).with_suffix(".jsonl").open("a") as f:
            f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
    result = {
        "model": settings().llm_model,
        "scope": "actual_llm_controlled_synthetic_evidence",
        "provenance": provenance(ROOT / "eval/data/judge_controls.jsonl"),
        "rows": rows,
        "note": "Outputs must follow the supplied values; changed output alone is not a semantic accuracy metric.",
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
