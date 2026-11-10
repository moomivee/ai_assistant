"""Compare embedders and dense/BM25+RRF retrieval on unchanged gold contexts."""

import argparse
import json
from pathlib import Path

from src.rag.index import MODEL_SPECS, index

from .metrics import mean_metrics, retrieval_metrics
from .run import DEFAULT_DATA, load_cases, provenance


def run(output):
    if Path(output).exists():
        raise FileExistsError("Choose a new output path to preserve previous measurements.")
    rows = []
    cases = [c for c in load_cases() if c["relevant"]]
    for model in MODEL_SPECS:
        idx = index(model)
        for strategy in ["dense", "hybrid"]:
            for variant in ["clean_query", "full_question"]:
                for case in cases:
                    query = (
                        case["retrieval_query"] if variant == "clean_query" else case["question"]
                    )
                    ranked = idx.search(query, 10, mode=strategy)
                    rows.append(
                        {
                            "case_id": case["id"],
                            "split": case["split"],
                            "model": model,
                            "strategy": strategy,
                            "variant": variant,
                            "metrics": retrieval_metrics(ranked, case["relevant"]),
                            "ranking": [r["chunk_id"] for r in ranked],
                        }
                    )
    summary = {}
    for model in MODEL_SPECS:
        for strategy in ["dense", "hybrid"]:
            for variant in ["clean_query", "full_question"]:
                for split in ["dev", "test", "all"]:
                    selected = [
                        r
                        for r in rows
                        if r["model"] == model
                        and r["strategy"] == strategy
                        and r["variant"] == variant
                        and (split == "all" or r["split"] == split)
                    ]
                    summary[f"{model}/{strategy}/{variant}/{split}"] = {
                        "n": len(selected),
                        **mean_metrics([r["metrics"] for r in selected]),
                    }
    result = {
        "provenance": provenance(DEFAULT_DATA),
        "scope": "retrieval_only",
        "selection_rule": "Compare development evidence_recall@5; test is report-only, not a tuning set.",
        "summary": summary,
        "rows": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
