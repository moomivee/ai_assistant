"""Real-encoder infrastructure smoke, isolated from the user's knowledge catalog.

Run: uv run python -m tools.check_managed_rag --output /tmp/managed-rag-smoke.json
This checks ingestion/retrieval/version mechanics; it is not a retrieval-quality
benchmark. The CSV fixture is deliberately synthetic, not a market observation.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from eval.managed_retrieval import export_snapshot
from eval.managed_retrieval import run as evaluate_snapshot
from src.config import manifest, settings
from src.knowledge.store import get_store
from src.rag.index import MODEL_SPECS, _index, digest
from src.rag.managed import DocumentScopeError, _cache, managed_search, managed_status

FIXTURE = b"Grade,API gravity,Sulfur pct\nSYNTHETIC-MANAGED-SMOKE,42.3,0.17\n"
QUESTION = "SYNTHETIC-MANAGED-SMOKE oil API gravity and sulfur content"


def run(output: Path, model: str) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new output path to preserve previous measurements.")
    original = settings()
    reports = manifest()
    before_env = {name: os.environ.get(name) for name in ("DATA_DIR", "CACHE_DIR")}
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="managed-rag-real-") as temporary:
        root = Path(temporary)
        data, cache = root / "data", root / "cache"
        (data / "reports").mkdir(parents=True)
        cache.mkdir()
        shutil.copy2(original.data / "reports.json", data / "reports.json")
        for report in reports:
            shutil.copy2(
                original.data / "reports" / report["file"], data / "reports" / report["file"]
            )
        # Reuse checksum-verified model downloads; never reuse the user's catalog.
        (original.cache / "models").mkdir(parents=True, exist_ok=True)
        (cache / "models").symlink_to(original.cache / "models", target_is_directory=True)
        for baseline in original.cache.glob("index-*"):
            if baseline.is_dir():
                shutil.copytree(baseline, cache / baseline.name)
        try:
            os.environ["DATA_DIR"] = str(data)
            os.environ["CACHE_DIR"] = str(cache)
            settings.cache_clear()
            _index.cache_clear()
            _cache.clear()
            store = get_store()
            document = store.upload(
                "synthetic-oil.csv", FIXTURE, title="Synthetic RAG smoke fixture"
            )
            document_id, version_id = document["id"], document["current_version_id"]
            store.process(version_id)
            global_results = managed_search(QUESTION, 10, model=model, mode="dense")
            scoped = managed_search(QUESTION, 5, [document_id], model=model, mode="dense")
            source_blocks = store.blocks(document_id, version_id)
            checks = {
                "scoped_retrieval_returns_only_upload": bool(scoped)
                and all(row["document_id"] == document_id for row in scoped),
                "scope_preserves_version": bool(scoped)
                and all(row["version_id"] == version_id for row in scoped),
                "quotes_are_exact_source_substrings": bool(scoped)
                and all(
                    any(row["text"] in block["text"] for block in source_blocks) for row in scoped
                ),
                "global_top10_contains_upload": any(
                    row.get("document_id") == document_id for row in global_results
                ),
            }
            index_status = managed_status(model)
            frozen_path, labels_path = root / "snapshot.json", root / "labels.jsonl"
            export_snapshot(frozen_path, labels_path, model, [document_id])
            label = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
            label["question"] = QUESTION
            label["gold"][0]["quote"] = "42.3"
            label["synthetic_infrastructure_fixture"] = True
            labels_path.write_text(json.dumps(label) + "\n", encoding="utf-8")
            store.archive(document_id)
            after_archive = managed_search(QUESTION, 10, model=model, mode="dense")
            checks["archive_excluded_from_global_retrieval"] = all(
                row.get("document_id") != document_id for row in after_archive
            )
            try:
                managed_search(QUESTION, 5, [document_id], model=model, mode="dense")
                checks["archived_explicit_scope_rejected"] = False
            except DocumentScopeError:
                checks["archived_explicit_scope_rejected"] = True
            checks["archived_original_still_available_for_citation"] = (
                store.version_file(document_id, version_id).read_bytes() == FIXTURE
            )
            offline = evaluate_snapshot(
                frozen_path, labels_path, root / "snapshot-eval.json", model=model
            )
            checks["frozen_snapshot_evaluates_after_live_document_archive"] = (
                offline["summary"]["all"]["all_evidence@5"] == 1.0
            )
            result = {
                "time": datetime.now(timezone.utc).isoformat(),
                "scope": "real_encoder_managed_corpus_infrastructure_smoke",
                "quality_benchmark": False,
                "model": model,
                "model_spec": MODEL_SPECS[model],
                "fixture_sha256": hashlib.sha256(FIXTURE).hexdigest(),
                "fixture_note": "Synthetic table; not oil-market data or a labeled evaluation set.",
                "baseline_corpus": {
                    row["id"]: digest(data / "reports" / row["file"]) for row in reports
                },
                "question": QUESTION,
                "checks": checks,
                "passed": all(checks.values()),
                "index": index_status,
                "global_ranking": [
                    {key: row.get(key) for key in ("report", "version_id", "chunk_id", "score")}
                    for row in global_results
                ],
                "scoped_quotes": [row["text"] for row in scoped],
                "seconds": round(perf_counter() - started, 3),
            }
        finally:
            for name, value in before_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            settings.cache_clear()
            _index.cache_clear()
            _cache.clear()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_SPECS, default="minilm-multi")
    args = parser.parse_args()
    report = run(args.output, args.model)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
