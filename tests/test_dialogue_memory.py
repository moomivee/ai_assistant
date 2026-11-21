import hashlib
import subprocess
import sys

import pytest
from ouroboros import dialogue_store

from src.integrations import dialogue_memory as memory

SCOPE = {"document_ids": ["oil"], "report_only": False, "note_id": None}


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


def test_native_store_load_rotate_close_and_isolate_chains():
    original = {"users": ["Рассчитай Urals"], "benchmark": "Urals"}
    updated = {**original, "months": 3}
    first = memory.open_dialogue(original, SCOPE)
    other = memory.open_dialogue({"benchmark": "WTI"}, {})
    with memory.resume_dialogue(first, SCOPE) as turn:
        assert turn.snapshot == original
        second = turn.advance(updated)
        with pytest.raises(memory.DialogueConflict):
            turn.close()
    assert first != second
    with pytest.raises(memory.DialogueConflict):
        with memory.resume_dialogue(first, SCOPE):
            pytest.fail("superseded turn resumed")
    with memory.resume_dialogue(other, {}) as separate:
        assert separate.snapshot == {"benchmark": "WTI"}
    with memory.resume_dialogue(second, SCOPE) as turn:
        assert turn.snapshot == updated
        turn.close()
    with pytest.raises(memory.DialogueConflict):
        with memory.resume_dialogue(second, SCOPE):
            pytest.fail("closed turn resumed")
    bucket = next(memory._root().glob("day-*"))
    rows = dialogue_store.room_candidates(bucket, "chat", int(first[8:24], 16), limit=3)
    assert [row.row["type"] for row in rows] == [
        "forecast_clarification_closed",
        "forecast_clarification",
        "forecast_clarification",
    ]
    assert rows[1].row["token_hash"] == hashlib.sha256(second.encode()).hexdigest()
    assert second not in str(rows)


def test_failed_request_remains_retryable_but_released_turn_cannot_write():
    token = memory.open_dialogue({"months": 3}, {})
    with pytest.raises(OSError, match="upstream"):
        with memory.resume_dialogue(token, {}) as turn:
            raise OSError("upstream")
    with pytest.raises(memory.DialogueConflict):
        turn.advance({})
    with memory.resume_dialogue(token, {}) as retry:
        assert retry.snapshot == {"months": 3}


def test_check_capability_and_scope_without_opening_new_rooms():
    token = memory.open_dialogue({"months": 3}, SCOPE)
    bucket = next(memory._root().glob("day-*"))
    lock_count = len(list(bucket.glob("*.lock")))
    wrong_secret = token[:-1] + ("0" if token[-1] != "0" else "1")
    for candidate, scope in [
        (token, {}),
        (wrong_secret, SCOPE),
        (token[:8] + "f" * 16 + token[24:], SCOPE),
        (token[:8] + "0" * 16 + token[24:], SCOPE),
        ("not-a-token", SCOPE),
    ]:
        with pytest.raises(memory.DialogueConflict):
            with memory.resume_dialogue(candidate, scope):
                pytest.fail("unrelated scope or token resumed")
    assert len(list(bucket.glob("*.lock"))) == lock_count


def test_lease_works_across_processes_and_only_locks_one_chain():
    token = memory.open_dialogue({"months": 3}, {})
    other = memory.open_dialogue({"months": 5}, {})
    script = """
import sys
from src.integrations.dialogue_memory import DialogueBusy, resume_dialogue
with resume_dialogue(sys.argv[2], {}) as independent:
    assert independent.snapshot == {"months": 5}
try:
    with resume_dialogue(sys.argv[1], {}) as turn:
        assert turn.snapshot == {"months": 3}
        print("ready")
except DialogueBusy:
    print("busy")
"""
    command = [sys.executable, "-c", script, token, other]
    with memory.resume_dialogue(token, {}):
        assert subprocess.check_output(command, text=True).strip() == "busy"
    assert subprocess.check_output(command, text=True).strip() == "ready"


def test_expiry_purge_and_active_worker_across_daily_boundary(monkeypatch):
    now = 20700 * memory.TTL_SECONDS + 10
    monkeypatch.setattr(memory, "time", lambda: now)
    first = memory.open_dialogue({"benchmark": "Urals"}, {})
    old_bucket = next(memory._root().glob("day-*"))
    with memory.resume_dialogue(first, {}):
        now += memory.TTL_SECONDS * 2
        token = memory.open_dialogue({"months": 6}, {})
        assert old_bucket.exists(), "old store removed under a still-active worker"
    memory.open_dialogue({"months": 5}, {})
    assert not old_bucket.exists()
    with memory.resume_dialogue(token, {}) as latest:
        assert latest.snapshot == {"months": 6}


def test_absolute_expiration_does_not_extend_on_advance(monkeypatch):
    now = 20700 * memory.TTL_SECONDS + 10
    monkeypatch.setattr(memory, "time", lambda: now)
    token = memory.open_dialogue({"benchmark": "Urals"}, {})
    now += memory.TTL_SECONDS - 1
    with memory.resume_dialogue(token, {}) as turn:
        token = turn.advance({"months": 3})
    now += 1
    with pytest.raises(memory.DialogueConflict):
        with memory.resume_dialogue(token, {}):
            pytest.fail("expired clarification resumed")
