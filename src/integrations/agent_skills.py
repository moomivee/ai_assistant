"""App-bundled instruction skills through the real Ouroboros admission gate.

This is a host adapter, not a second Ouroboros autonomous runtime. Checked-in
allowlist hashes admit exact app-owned instruction packages as a trusted local
source. We never manufacture a critic PASS or mark them as Ouroboros native seeds.
Enable state, install provenance, payload hashing and admission are native APIs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from filelock import FileLock
from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
from ouroboros.skill_catalogue import discover_context_skills
from ouroboros.skill_install_record import read_install_record, write_install_record
from ouroboros.skill_loader import load_skill as native_load_skill
from ouroboros.skill_loader import save_enabled

from ..config import ROOT, settings
from .telemetry import span

BUNDLED_SKILLS = ROOT / "agent_skills"
_selected: ContextVar[dict | None] = ContextVar("analyst_instruction_skill", default=None)


class SkillUnavailable(ValueError):
    """An unknown, disabled, changed or unadmitted skill cannot enter model context."""


def _root(runtime_root: Path | None) -> Path:
    return Path(runtime_root or settings().data / "ouroboros").resolve()


def _pins(bundle_root: Path) -> dict:
    document = json.loads((bundle_root / "approved.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("skills"), dict):
        raise SkillUnavailable("Invalid bundled skill allowlist")
    return document["skills"]


def _exact_manifest(folder: Path, expected: str) -> str:
    # Bundled instruction packages contain one file, with no executable payload.
    if folder.is_symlink() or any(p.name != "SKILL.md" for p in folder.iterdir()):
        raise SkillUnavailable("Unexpected files in instruction package")
    path = folder / "SKILL.md"
    if path.is_symlink():
        raise SkillUnavailable("Skill manifests must be regular files")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SkillUnavailable("Skill bytes differ from the application allowlist")
    return raw.decode("utf-8")


def _upgrade_bundled(runtime: Path, destination: Path, name: str, pin: dict, body: str) -> bool:
    """Replace only an explicitly pinned, unedited, app-installed prior release."""
    path = destination / "SKILL.md"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == pin["manifest_sha256"]:
        return False
    if actual not in pin.get("previous_manifest_sha256", []):
        raise SkillUnavailable("Skill bytes differ from the application allowlist")
    _exact_manifest(destination, actual)
    old = native_load_skill(destination, runtime)
    record = read_install_record(runtime, name)
    if (
        old is None
        or old.load_error
        or old.name != name
        or not record
        or record.get("basis") != "trusted"
        or record.get("source", {}).get("id") != "ai-assistant-bundled"
        or record.get("source", {}).get("kind") != "local"
        or record.get("release", {}).get("digest") != old.content_hash
        or record.get("release", {}).get("version") != old.manifest.version
        or not old.review.gate_for(old.content_hash)["executable_review"]
    ):
        raise SkillUnavailable("Prior skill release is not admitted as an application bundle")
    # Stage outside the package: extra files inside it would change its hash.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(body.encode("utf-8"))
    try:
        _exact_manifest(destination, actual)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _install_bundled(runtime: Path, bundle_root: Path) -> dict[str, str]:
    """Install/upgrade pinned releases; never overwrite edits or reset disablement."""
    errors: dict[str, str] = {}
    runtime.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime / ".bundled-skills.lock"), timeout=10):
        for name, pin in _pins(bundle_root).items():
            try:
                if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
                    raise SkillUnavailable("Invalid bundled skill name")
                source = bundle_root / name
                body = _exact_manifest(source, pin["manifest_sha256"])
                manifest = parse_skill_manifest_text(body)
                if (
                    manifest.name != name
                    or manifest.version != pin["version"]
                    or not manifest.is_instruction()
                    or manifest.permissions
                    or manifest.scripts
                    or manifest.entry
                ):
                    raise SkillUnavailable("Bundled skill contract does not match allowlist")
                destination = runtime / "skills" / "external" / name
                destination.resolve().relative_to(runtime)
                upgraded = False
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
                    try:
                        (temporary / "SKILL.md").write_text(body, encoding="utf-8")
                        temporary.rename(destination)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                else:
                    upgraded = _upgrade_bundled(runtime, destination, name, pin, body)
                _exact_manifest(destination, pin["manifest_sha256"])
                loaded = native_load_skill(destination, runtime)
                if loaded is None or loaded.load_error:
                    raise SkillUnavailable("Ouroboros could not load this instruction skill")
                record = read_install_record(runtime, name)
                if record is None or upgraded:
                    write_install_record(
                        runtime,
                        name,
                        {
                            "basis": "trusted",
                            "source": {
                                "id": "ai-assistant-bundled",
                                "kind": "local",
                                "name": "Application instruction allowlist",
                            },
                            "release": {"version": manifest.version, "digest": loaded.content_hash},
                            "installed_by": "application-bootstrap",
                            "admission_note": "Exact checked-in bundle; no model review claimed",
                        },
                    )
                enabled_path = runtime / "state" / "skills" / name / "enabled.json"
                if not enabled_path.exists():
                    save_enabled(
                        runtime,
                        name,
                        True,
                        actor="application-bootstrap",
                        reason="Bundled instruction skill enabled on first install",
                    )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors[name] = str(exc)
    return errors


def _discovered(runtime_root: Path | None, bundle_root: Path | None):
    runtime = _root(runtime_root)
    bundle = Path(bundle_root or BUNDLED_SKILLS)
    errors = _install_bundled(runtime, bundle)
    pins = _pins(bundle)
    skills = discover_context_skills(runtime)
    return runtime, pins, errors, skills


def get_skill_catalog(
    *, runtime_root: Path | None = None, bundle_root: Path | None = None
) -> list[dict]:
    """Compact model/UI descriptions with real enablement and current admission."""
    _, pins, errors, skills = _discovered(runtime_root, bundle_root)
    rows = []
    for name, pin in pins.items():
        matches = [s for s in skills if s.name == name]
        skill = matches[0] if len(matches) == 1 else None
        error = errors.get(name, "")
        if len(matches) > 1:
            error = "Ambiguous skill identity"
        if skill is None:
            rows.append(
                {
                    "name": name,
                    "version": pin["version"],
                    "enabled": False,
                    "available": False,
                    "error": error or "Skill not discovered",
                }
            )
            continue
        admitted = skill.review.gate_for(skill.content_hash)["executable_review"]
        available = bool(
            not error
            and not skill.load_error
            and skill.enabled
            and skill.manifest.is_instruction()
            and admitted
        )
        rows.append(
            {
                "name": name,
                "version": skill.manifest.version,
                "description": skill.manifest.description,
                "when_to_use": skill.manifest.when_to_use,
                "hash": skill.content_hash,
                "enabled": skill.enabled,
                "available": available,
                "admission_basis": "trusted_application_bundle",
                "error": error
                or skill.load_error
                or ("" if available else "Disabled or exact bytes not admitted"),
            }
        )
    return rows


def load_skill(
    name: str, *, runtime_root: Path | None = None, bundle_root: Path | None = None
) -> dict:
    """Read admitted instruction bytes, checking enablement and hash every time."""
    with span("skill.load", metadata={"skill": name, "engine": "ouroboros.skill_loader"}) as obs:
        _, pins, errors, skills = _discovered(runtime_root, bundle_root)
        if name not in pins:
            raise SkillUnavailable("Unknown instruction skill")
        matches = [s for s in skills if s.name == name]
        if name in errors or len(matches) != 1:
            raise SkillUnavailable(errors.get(name, "Skill unavailable or ambiguous"))
        skill = matches[0]
        if (
            skill.load_error
            or not skill.enabled
            or not skill.manifest.is_instruction()
            or not skill.review.gate_for(skill.content_hash)["executable_review"]
        ):
            raise SkillUnavailable("Instruction skill disabled or exact bytes not admitted")
        # Discovery reads manifest and hashes separately. Re-read once and check
        # the approved bytes so an intervening edit cannot enter model context.
        text = _exact_manifest(skill.skill_dir, pins[name]["manifest_sha256"])
        manifest = parse_skill_manifest_text(text)
        result = {
            "name": name,
            "version": manifest.version,
            "hash": skill.content_hash,
            "instructions": manifest.body,
            "admission_basis": "trusted_application_bundle",
        }
        obs.update(metadata={k: v for k, v in result.items() if k != "instructions"})
        return result


def set_skill_enabled(
    name: str, enabled: bool, *, runtime_root: Path | None = None, bundle_root: Path | None = None
) -> None:
    runtime, pins, _, _ = _discovered(runtime_root, bundle_root)
    if name not in pins:
        raise SkillUnavailable("Unknown instruction skill")
    save_enabled(
        runtime,
        name,
        enabled,
        actor="application-operator",
        reason="Explicit instruction skill toggle",
    )


def instruction_context(name: str | None, **kwargs) -> dict:
    if not name:
        return {"text": "", "metadata": {}}
    skill = load_skill(name, **kwargs)
    metadata = {k: v for k, v in skill.items() if k != "instructions"}
    return {
        "text": f"Selected instruction skill: {name}\n\n{skill['instructions']}",
        "metadata": metadata,
    }


@contextmanager
def skill_context(name: str | None, **kwargs):
    """Request-local selection; independent concurrent requests never share it."""
    value = instruction_context(name, **kwargs)
    token = _selected.set(value)
    try:
        yield value
    finally:
        _selected.reset(token)


def current_skill_context() -> dict:
    return _selected.get() or {"text": "", "metadata": {}}
