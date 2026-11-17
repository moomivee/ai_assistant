"""Version the existing gold set for source-adaptive routing without changing old results."""

import json
from copy import deepcopy
from pathlib import Path

DATA = Path(__file__).parent / "data"


def adapt(case):
    """Use authored category/scope only; never let a predicted plan relax gold."""
    row = deepcopy(case)
    expected = row["expected"]
    required = list(expected["tools"])
    groups = []
    checks = expected.get("plan_checks", {})
    category = row["category"]
    if category == "web":
        required = ["web_search"]
    elif category == "forecast" and checks.get("wants_context") is False:
        required = ["price_forecast"]
        expected["forbidden"] = ["report_search", "read_report", "web_search", "read_web"]
    elif category == "research" and not checks.get("report_only"):
        required = []
        groups = [["report_search", "web_search"]]
    expected.update(
        routing_policy="adaptive-v3", required_tools=required, required_tool_groups=groups
    )
    row["provenance"].update(version="seed-v3-adaptive", parent_version="seed-v2")
    return row


def build():
    return [
        adapt(json.loads(line))
        for line in (DATA / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    (DATA / "cases_v3.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in build()), encoding="utf-8"
    )
