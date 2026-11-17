"""Keep the description of an output separate from the data the model must return."""

import json


def schema_contract(schema):
    properties = list(schema.get("properties", {}))
    required = list(schema.get("required", []))
    return (
        "\n\nСформируй JSON-объект с результатом исходной задачи. "
        "Следующий блок — только описание допустимого формата, а не содержимое ответа.\n"
        "BEGIN_OUTPUT_SCHEMA\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nEND_OUTPUT_SCHEMA\n"
        + "Обязательные поля результата: "
        + json.dumps(required, ensure_ascii=False)
        + ". Допустимые поля результата: "
        + json.dumps(properties, ensure_ascii=False)
        + ".\nЗаполни значения полей результата по входным данным и исходным инструкциям. "
        "Не возвращай описание формата, properties, $defs или additionalProperties "
        "вместо значений этих полей. Верни только объект данных, без Markdown и пояснений."
    )


def is_schema_echo(content, schema):
    """Recognize a schema document, never convert it into an invented answer."""
    if content is None:
        return False
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return False
    expected = set(schema.get("required", []))
    return (
        isinstance(value, dict)
        and bool(expected - value.keys())
        and "properties" not in expected
        and isinstance(value.get("properties"), dict)
        and bool({"$defs", "type", "additionalProperties", "required"} & value.keys())
    )


def validation_issues(error, schema):
    """Only known schema paths and error codes: no input values or provider text."""
    names = {"$defs", "properties", "additionalProperties", "required", "type", "title", "$schema"}

    def collect(node):
        if isinstance(node, dict):
            names.update(node.get("properties", {}))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(schema)
    return [
        {
            "path": [
                part if isinstance(part, int) or part in names else "<unknown_field>"
                for part in item["loc"]
            ],
            "type": item["type"],
        }
        for item in error.errors(include_input=False, include_context=False, include_url=False)[:10]
    ]
