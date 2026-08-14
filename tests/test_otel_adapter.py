"""The portability test: a trace this package did not grow up on.

The fixture is REAL output — an Anthropic tool-calling agent instrumented with
OpenInference and exported through the OpenTelemetry SDK, captured verbatim and then
stripped of the prompt text. Nothing about its shape was chosen here: flattened indexed
attributes, ISO timestamps, span kinds, `output.value` as an opaque string. A detector
that only ever met Langfuse's observation model has no business passing this by luck, so
these assert the DETECTIONS, not just that parsing did not raise.

The agent was given two deliberately unhelpful tools. `send_notification` declines in its
own body (`{"sent": false}`) with no error status, which is the shape that satisfies every
did-it-run guard, and the model then said — unprompted, in its own words — that it
*wasn't able to* send. So the same turn exercises both directions at once: a refusal that
must be caught, and a negated claim that must not be.
"""
import json
import pathlib

import pytest

from postflight import Config, Generation, Outcome, Turn, run, tool_outcome
from postflight.adapters.otel import reply_text, turns, turns_from_jsonl

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "openinference_support_turn.jsonl")


@pytest.fixture
def turn_():
    built = turns_from_jsonl(FIXTURE)
    assert len(built) == 1
    return built[0]


def codes(findings):
    return {f.code for f in findings}


def test_span_tree_becomes_one_ordered_turn(turn_):
    assert turn_.kind == "support.turn"          # from the AGENT root, not a span name
    assert len(turn_.generations) == 3
    assert [c.name for c in turn_.tool_calls] == ["search_orders", "send_notification"]
    assert turn_.duration_s > 0


def test_reply_is_reassembled_from_flattened_attributes(turn_):
    """OpenInference spreads a reply over
    `llm.output_messages.<i>.message.contents.<j>.message_content.text`. It is not a
    field to read; it has to be rebuilt."""
    assert "RX-2231" in turn_.reply
    assert turn_.reply.startswith("I found order")


def test_content_indices_sort_numerically_not_lexically():
    """`contents.10` must follow `contents.2`. String-sorted keys scramble any reply
    long enough to span ten blocks, and it reads as the model producing word salad."""
    attrs = {f"llm.output_messages.0.message.contents.{i}.message_content.text": f"{i} "
             for i in range(12)}
    assert reply_text(attrs).split() == [str(i) for i in range(12)]


def test_last_output_message_wins():
    """Earlier output messages are the model talking to itself mid-loop."""
    attrs = {"llm.output_messages.0.message.contents.0.message_content.text": "thinking",
             "llm.output_messages.1.message.contents.0.message_content.text": "answer"}
    assert reply_text(attrs) == "answer"


def test_in_body_refusal_is_caught_with_DEFAULT_config(turn_):
    """No Base tuning, no domain vocabulary — the shipped defaults on a foreign trace."""
    found = run(turn_, Config())
    assert "TOOL_REFUSAL" in codes(found)
    refusal = next(f for f in found if f.code == "TOOL_REFUSAL")
    assert refusal.detail["calls"][0]["tool"] == "send_notification"


def test_the_models_own_negated_sentence_is_not_a_claim(turn_):
    """The reply says it *wasn't able to* send. Prose nobody wrote for this test."""
    assert "UNVERIFIED_CLAIM" not in codes(run(turn_, Config()))


def test_unreported_cache_does_not_fabricate_a_miss(turn_):
    """OpenInference emits no cache attribute at all. Treating that as zero made every
    sufficiently large turn from this instrumentation a NO_CACHE_HIT — confirmed against
    a real 4,852-token prompt before this was fixed."""
    assert all(g.cache_read_tokens is None for g in turn_.generations)
    assert "NO_CACHE_HIT" not in codes(run(turn_, Config()))


def test_tokens_still_add_up(turn_):
    assert turn_.total_tokens > 0
    assert turn_.total_tokens == sum(g.total_tokens for g in turn_.generations)


def test_error_status_marks_a_tool_call():
    span = {"name": "tool.x", "context": {"trace_id": "t"}, "start_time": None,
            "end_time": None, "status": {"status_code": "ERROR"},
            "attributes": {"openinference.span.kind": "TOOL", "tool.name": "x",
                           "output.value": "{}"}}
    built = turns([span])[0]
    assert tool_outcome(built.tool_calls[0], Config()) is Outcome.ERRORED


def test_nanosecond_timestamps_are_accepted():
    """`to_json()` gives ISO strings; OTLP gives integer nanos. Both are real."""
    span = {"name": "a.turn", "context": {"trace_id": "t"}, "parent_id": None,
            "start_time": 1_700_000_000_000_000_000,
            "end_time": 1_700_000_002_000_000_000, "attributes": {}}
    assert turns([span])[0].duration_s == 2.0


def test_spans_with_no_trace_id_are_dropped_not_crashed():
    assert turns([{"name": "orphan", "attributes": {}}]) == []


def test_generation_default_is_unknown_not_zero():
    """The distinction the cache detector depends on."""
    assert Generation(text="x").cache_read_tokens is None
    assert Turn(id="t", steps=(Generation(text="x"),)).cache_read_tokens == 0
