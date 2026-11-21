"""Evidence packing preserves exact blocks while giving every source a turn."""

from src.evidence import catalog
from src.progress import context_blocks, context_sources


def source(source_id, text, *, page=None, full_page=False):
    return {
        "id": source_id,
        "text": text,
        "kind": "report" if source_id.startswith("R") else "web",
        "url": f"https://example.org/{source_id}",
        "report": "report",
        "page": page,
        "full_page": full_page,
    }


def test_opened_old_pages_cannot_exhaust_context_before_new_web_evidence():
    old = [source(f"R{i}", "Old record " * 450, page=i, full_page=True) for i in range(5)]
    web = source("W1", "New primary announcement with the requested event.")
    evidence = catalog(old + [web])
    assert evidence["W1:B1"]["text"] == web["text"]
    assert {row["source"] for row in evidence.values()} == {s["id"] for s in old + [web]}
    assert sum(len(row["text"]) for row in evidence.values()) <= 18000


def test_latest_sources_get_a_turn_even_with_more_sources_than_budget():
    old = [source(f"R{i}", "Z" * 5000, page=i, full_page=True) for i in range(15)]
    web = source("W1", "New evidence.")
    evidence = catalog(old + [web], limit=1600, priority_source_ids=["W1"])
    assert list(evidence) == ["W1:B1"]
    assert evidence["W1:B1"]["text"] == web["text"]


def test_block_identity_and_exact_offsets_do_not_depend_on_limit_or_priority():
    rows = [source("R1", "Header\n" + "value 123.45\n" * 500), source("W1", "Current text " * 350)]
    full = catalog(rows, limit=20000)
    packed = catalog(rows, limit=4000, priority_source_ids=["W1"])
    originals = {row["id"]: row["text"] for row in rows}
    assert next(iter(packed)) == "W1:B1"
    for key, row in packed.items():
        assert row == full[key]
        assert (
            row["text"]
            == originals[row["source"]][row["source_text_start"] : row["source_text_end"]]
        )
        assert len(row["text"]) <= 1600
    assert sum(len(row["text"]) for row in packed.values()) <= 4000


def test_priority_follows_excerpt_when_full_page_replaces_it():
    unrelated = source("R0", "Another opened page", page=2, full_page=True)
    excerpt = source("R1", "API 30; sulfur 1.5%", page=1)
    page = source("R2", "Table header\n" + excerpt["text"], page=1, full_page=True)
    selected = context_sources([unrelated, excerpt, page], priority_source_ids=["R1"])
    assert [row["id"] for row in selected] == ["R2", "R0"]
    evidence = catalog([unrelated, excerpt, page], priority_source_ids=["R1"])
    assert next(iter(evidence)) == "R2:B1"
    assert evidence["R2:B1"]["text"] == page["text"]
    assert "R1:B1" not in evidence


def test_dated_citation_remains_available_with_newer_retrievals():
    last = source("W1", "A" * 1600 + "Earlier confirmed date.")
    new = source("W2", "B" * 1550)
    evidence = catalog(
        [last, new], limit=1600, priority_source_ids=["W2"], retained_block_id="W1:B2"
    )
    assert list(evidence) == ["W1:B2", "W2:B1"]
    assert evidence["W1:B2"]["text"] == last["text"][1600:]
    assert sum(len(row["text"]) for row in evidence.values()) <= 1600


def test_excluded_source_cannot_return_via_retained_citation():
    source_after_screening = source("W2", "A different event.")
    evidence = catalog([source_after_screening], retained_block_id="W1:B2")
    assert list(evidence) == ["W2:B1"]


def test_whole_small_table_and_normal_numeric_tokens_are_preserved():
    table = "Grade | API | Sulfur\nUrals | 31 | 1.5%\nESPO | 35 | 0.5%"
    text = "A" * 1595 + " 1234567.89 USD"
    evidence = catalog([source("R1", table), source("R2", text)])
    assert evidence["R1:B1"]["text"] == table
    assert "1234567.89" in evidence["R2:B2"]["text"]
    assert "1234567.89" not in evidence["R2:B1"]["text"]


def test_budget_does_not_take_tail_after_skipping_a_large_initial_block():
    rows = [source("R1", "A" * 1600 + "small tail"), source("W1", "Fits.")]
    blocks = context_blocks(rows, limit=20)
    assert [(s["id"], i) for s, i, *_ in blocks] == [("W1", 1)]
    assert not context_blocks(rows, limit=0)
