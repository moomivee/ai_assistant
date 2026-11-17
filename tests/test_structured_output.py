import json

import pytest

from src.integrations.gigachat import _retry_messages
from src.integrations.structured_output import is_schema_echo
from src.models.schemas import CitedAnswer


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"type": "object", "properties": {"claims": {"type": "array"}}}, True),
        ({"$defs": {}, "additionalProperties": False, "properties": {}}, True),
        ({"claims": [], "complete": False}, False),
        ({"claims": [], "complete": False, "properties": {}, "type": "object"}, False),
        ({"wrong": "shape"}, False),
        ([], False),
        (None, False),
    ],
)
def test_schema_echo_detection_does_not_classify_data_or_other_errors_as_schema(value, expected):
    assert is_schema_echo(json.dumps(value), CitedAnswer.model_json_schema()) is expected


def test_invalid_json_is_not_schema_echo():
    assert not is_schema_echo("```json\n{}\n```", CitedAnswer.model_json_schema())


def test_transport_only_retry_adds_one_contract_and_keeps_user_data():
    schema = CitedAnswer.model_json_schema()
    original = [{"role": "user", "content": "Question and evidence"}]
    once = _retry_messages(original, schema)
    twice = _retry_messages(once, schema)
    for messages in (once, twice):
        assert messages[1:] == original
        assert messages[0]["content"].count(json.dumps(schema, ensure_ascii=False)) == 1
    assert original == [{"role": "user", "content": "Question and evidence"}]
