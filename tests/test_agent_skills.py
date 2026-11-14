import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest
from ouroboros.skill_loader import discover_skills

from src.integrations.agent_skills import (
    BUNDLED_SKILLS,
    SkillUnavailable,
    current_skill_context,
    get_skill_catalog,
    load_skill,
    set_skill_enabled,
    skill_context,
)


@pytest.fixture
def skills(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLED_SKILLS, bundle)
    return {"runtime_root": tmp_path / "runtime", "bundle_root": bundle}


def test_real_ouroboros_discovery_and_trusted_admission(skills):
    catalog = get_skill_catalog(**skills)
    assert {row["name"] for row in catalog} == {
        "report_analysis",
        "web_research",
        "price_forecast",
    }
    assert all(row["enabled"] and row["available"] for row in catalog)
    native = discover_skills(skills["runtime_root"])
    assert len(native) == 3
    for skill in native:
        assert skill.location == "external"
        assert skill.manifest.is_instruction()
        assert skill.review.gate_for(skill.content_hash)["executable_review"]
        assert skill.review.acceptance["basis"] == "trusted"
        state = skills["runtime_root"] / "state" / "skills" / skill.name
        assert not (state / "review.json").exists()  # no fabricated model review
        record = json.loads((state / "install.json").read_text())
        assert record["source"]["id"] == "ai-assistant-bundled"
        assert record["release"]["digest"] == skill.content_hash
        loaded = load_skill(skill.name, **skills)
        assert loaded["hash"] == skill.content_hash
        assert loaded["instructions"] == skill.manifest.body


def test_disable_persists_across_discovery_and_blocks_instructions(skills):
    set_skill_enabled("report_analysis", False, **skills)
    for _ in range(2):
        row = next(row for row in get_skill_catalog(**skills) if row["name"] == "report_analysis")
        assert not row["enabled"] and not row["available"]
    with pytest.raises(SkillUnavailable, match="disabled"):
        load_skill("report_analysis", **skills)
    set_skill_enabled("report_analysis", True, **skills)
    assert load_skill("report_analysis", **skills)["instructions"]


def test_modified_runtime_payload_is_not_replaced_or_reapproved(skills):
    load_skill("report_analysis", **skills)
    path = skills["runtime_root"] / "skills/external/report_analysis/SKILL.md"
    modified = path.read_text() + "\nUnapproved new instruction.\n"
    path.write_text(modified)
    with pytest.raises(SkillUnavailable, match="allowlist"):
        load_skill("report_analysis", **skills)
    assert path.read_text() == modified
    row = next(s for s in discover_skills(skills["runtime_root"]) if s.name == "report_analysis")
    assert not row.review.gate_for(row.content_hash)["executable_review"]


def test_modified_bundle_and_extra_payload_files_fail_closed(skills):
    path = skills["bundle_root"] / "web_research/SKILL.md"
    path.write_text(path.read_text() + "\nchanged\n")
    with pytest.raises(SkillUnavailable, match="allowlist"):
        load_skill("web_research", **skills)
    load_skill("price_forecast", **skills)
    payload = skills["runtime_root"] / "skills/external/price_forecast/extra.py"
    payload.write_text("print('should never run')")
    with pytest.raises(SkillUnavailable, match="Unexpected files"):
        load_skill("price_forecast", **skills)


def test_native_admission_record_must_still_match(skills):
    load_skill("report_analysis", **skills)
    path = skills["runtime_root"] / "state/skills/report_analysis/install.json"
    record = json.loads(path.read_text())
    record["release"]["digest"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(SkillUnavailable, match="not admitted"):
        load_skill("report_analysis", **skills)


def test_unknown_skill_and_path_cannot_be_selected(skills):
    for name in ["../report_analysis", "invented", ""]:
        with pytest.raises(SkillUnavailable, match="Unknown"):
            load_skill(name, **skills)


def test_context_is_request_local_and_resets_on_errors(skills):
    assert current_skill_context()["text"] == ""

    def worker(name):
        with skill_context(name, **skills):
            assert current_skill_context()["metadata"]["name"] == name
            return current_skill_context()["text"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(worker, ["report_analysis", "web_research"]))
    assert "report_analysis" in first and "web_research" in second
    assert current_skill_context()["text"] == ""
    with pytest.raises(RuntimeError), skill_context("report_analysis", **skills):
        raise RuntimeError("request aborted")
    assert current_skill_context()["text"] == ""
