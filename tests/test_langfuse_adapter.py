"""Adapter robustness against payloads a real trace store actually returns.

An adapter reads data it did not write. Every case here is a shape Langfuse permits
that took down the whole batch rather than one turn, because `turns()` builds eagerly.
"""
from datetime import timezone

from postflight import Config, run
from postflight.adapters.langfuse import LangfuseAdapter, text_of

ADAPTER = LangfuseAdapter()


def span(**kw):
    row = {"traceId": "t", "type": "SPAN", "startTime": "2026-08-13T12:00:00Z"}
    row.update(kw)
    return row


def test_non_dict_metadata_does_not_crash():
    """`metadata` is arbitrary JSON — a string and a list are both legal."""
    for metadata in ("a note", ["a", "list"], 7, None):
        built = ADAPTER.turn("t", [span(name="tool.x", output="{}", metadata=metadata)])
        assert built.tool_calls[0].is_error is False


def test_metadata_is_error_still_read_when_it_is_a_dict():
    built = ADAPTER.turn("t", [span(name="tool.x", output="{}",
                                    metadata={"isError": True})])
    assert built.tool_calls[0].is_error is True


def test_level_error_marks_the_call():
    built = ADAPTER.turn("t", [span(name="tool.x", output="{}", level="ERROR")])
    assert built.tool_calls[0].is_error is True


def test_timestamp_without_an_offset_is_assumed_utc():
    built = ADAPTER.turn("t", [span(name="a.turn", startTime="2026-08-13T12:00:00")])
    assert built.started_at is not None
    assert built.started_at.tzinfo is not None
    assert built.started_at.utcoffset() == timezone.utc.utcoffset(None)


def test_mixed_naive_and_aware_timestamps_still_sort():
    """One offset-less row among aware ones raised TypeError in the sort and killed
    the entire pull, not just the turn it came from."""
    built = ADAPTER.turns([
        {"traceId": "a", "type": "SPAN", "name": "a.turn",
         "startTime": "2026-08-13T12:00:00Z"},
        {"traceId": "b", "type": "SPAN", "name": "b.turn",
         "startTime": "2026-08-13T13:00:00"},
    ])
    assert [t.id for t in built] == ["b", "a"]   # newest first


def test_unparseable_timestamp_is_dropped_not_fatal():
    built = ADAPTER.turn("t", [span(name="a.turn", startTime="not a date")])
    assert built.started_at is None


def test_usage_details_supply_cache_numbers():
    built = ADAPTER.turn("t", [span(
        name="gen", type="GENERATION", model="claude-haiku-4-5",
        output='[{"type": "text", "text": "done"}]',
        usageDetails={"input": 9000, "output": 40,
                      "cache_read_input_tokens": 8000,
                      "cache_creation_input_tokens": 500},
    )])
    generation = built.generations[0]
    assert generation.input_tokens == 9000
    assert generation.cache_read_tokens == 8000
    assert generation.cache_write_tokens == 500
    assert generation.text == "done"


def test_zero_in_usage_falls_through_to_usage_details():
    """Keyed on `usage` alone, every turn reported 0 tokens and the cache detector
    never fired, because a zero prompt is never 'big'."""
    built = ADAPTER.turn("t", [span(
        name="gen", type="GENERATION", usage={"input": 0},
        usageDetails={"input": 9000},
    )])
    assert built.generations[0].input_tokens == 9000


def test_tool_output_that_is_not_json_is_kept_raw():
    """The error-prefix check reads the string, so it must survive the parse attempt."""
    built = ADAPTER.turn("t", [span(name="tool.x",
                                    output="Error executing tool x: boom")])
    assert "TOOL_ERROR" in {f.code for f in run(built, Config())}


def test_kind_comes_from_the_root_span_name():
    built = ADAPTER.turn("t", [span(name="pm_chat.turn"), span(name="tool.x")])
    assert built.kind == "pm_chat.turn"


def test_kind_falls_back_to_unknown():
    assert ADAPTER.turn("t", [span(name="tool.x")]).kind == "unknown"


def test_first_non_empty_user_id_wins():
    """A root span opened before the user is resolved carries an empty string."""
    built = ADAPTER.turn("t", [span(name="a.turn", userId=""),
                               span(name="tool.x", userId="u1")])
    assert built.user_id == "u1"


def test_text_of_handles_the_three_output_shapes():
    assert text_of('[{"type": "text", "text": "hi"}]') == "hi"
    assert text_of([{"type": "text", "text": "hi"}]) == "hi"
    assert text_of("plain reply") == "plain reply"
    assert text_of(None) == ""
