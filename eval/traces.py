"""Trace export -> review candidates -> labelled dataset; gold is never inferred from the agent."""

import argparse
import hashlib
import json
from pathlib import Path

from src.integrations.telemetry import flush, langfuse_client

from .run import load_cases


def read_events(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("name") == "agent.request" and isinstance(event.get("input"), dict):
            yield event


def candidates(path):
    seen = set()
    for event in read_events(path):
        inp = event["input"]
        key = (inp["question"], inp.get("previous"))
        if key in seen:
            continue
        seen.add(key)
        yield {
            "id": "trace-" + event["trace_id"],
            "question": key[0],
            "previous": key[1],
            "origin_trace_id": event["trace_id"],
            "observed_response": event.get("output"),
            "expected": None,
            "relevant": [],
            "review": "pending",
            "instructions": "Разметить ожидаемый маршрут и цитаты по первоисточнику независимо от ответа агента; затем назначить group и split.",
        }


def link_cases(cases, events):
    lookup = {}
    for event in events:
        inp = event["input"]
        lookup.setdefault((inp["question"], inp.get("previous")), set()).add(event["trace_id"])
    return [
        {
            **r,
            "observed_trace_ids": sorted(lookup.get((r["question"], r.get("previous")), [])),
        }
        for r in cases
    ]


def sync_dataset(cases, name):
    client = langfuse_client()
    if client is None:
        raise ValueError("Configure Langfuse first")
    sha = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    # Use an immutable/versioned dataset name. Existing versions must not be silently overwritten.
    try:
        client.get_dataset(name)
    except Exception as exc:
        if getattr(exc, "status_code", None) != 404:
            raise
    else:
        raise ValueError("Dataset already exists; choose a new versioned name")
    client.create_dataset(
        name=name,
        description="Authored source/requirements seed; review state stored per item",
        metadata={"sha256": sha},
    )
    for row in cases:
        client.create_dataset_item(
            dataset_name=name,
            id=hashlib.sha256((name + ":" + row["id"]).encode()).hexdigest(),
            input={"question": row["question"], "previous": row.get("previous")},
            expected_output={"expected": row["expected"], "relevant": row["relevant"]},
            metadata={
                **{k: row[k] for k in ["id", "group", "split", "provenance"]},
                "observed_trace_ids": row.get("observed_trace_ids", []),
            },
        )
    flush()
    return {"dataset": name, "items": len(cases), "sha256": sha}


def write_new(path, rows):
    with Path(path).open("x", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("traces", type=Path)
    c.add_argument("output", type=Path)
    c = sub.add_parser("link")
    c.add_argument("data", type=Path)
    c.add_argument("traces", type=Path)
    c.add_argument("output", type=Path)
    c = sub.add_parser("sync-dataset")
    c.add_argument("data", type=Path)
    c.add_argument("--name", required=True)
    args = p.parse_args()
    if args.command == "candidates":
        write_new(args.output, candidates(args.traces))
    elif args.command == "link":
        write_new(args.output, link_cases(load_cases(args.data), read_events(args.traces)))
    else:
        print(json.dumps(sync_dataset(load_cases(args.data), args.name), ensure_ascii=False))


if __name__ == "__main__":
    main()
