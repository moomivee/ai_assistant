import json

import pytest

from eval import assess_saved
from src.models.schemas import JudgeResult


@pytest.mark.parametrize("unavailable", [False, True])
def test_failed_calibration_does_not_score_answers_or_publish_success(
    monkeypatch, tmp_path, unavailable
):
    class ApprovesEverything:
        model = "test-judge"
        calls = 0

        def generate(self, *args):
            self.calls += 1
            if unavailable:
                raise RuntimeError("Judge unavailable")
            return JudgeResult(
                groundedness=4,
                relevance=4,
                tone=4,
                completeness=4,
                explanation="Everything is good.",
            )

    client = ApprovesEverything()
    monkeypatch.setenv("JUDGE_MODEL", client.model)
    monkeypatch.setattr(assess_saved, "make_judge", lambda: client)
    monkeypatch.setattr(assess_saved, "provenance", lambda _: {})
    output = tmp_path / "judged.json"
    assess_saved.assess([tmp_path / "not-read.json"], output, [])
    result = json.loads(output.read_text())
    assert client.calls == 5
    assert not result["calibration_passed"]
    assert result["task_success_rate"] is None and result["rows"] == []
