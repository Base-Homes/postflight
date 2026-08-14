"""`coverage()` exists because an inert detector and a clean agent look identical.

Each case here is a way an adapter or config can be wired such that a detector never
fires. The report would show a zero and a reader would draw the wrong conclusion; these
assert that `coverage()` says so out loud instead.
"""

import re
from datetime import UTC, datetime, timedelta

import pytest

from postflight import ClaimRule, Config, Generation, ToolCall, Turn, coverage

T0 = datetime(2026, 8, 14, tzinfo=UTC)


def rows(turns, cfg=None):
    return {r.code: r for r in coverage(turns, cfg or Config())}


def wired(**kw):
    """A turn with every input an adapter can supply."""
    steps = kw.pop(
        "steps",
        (
            Generation(
                text="I've sent them a message.",
                input_tokens=900,
                cache_read_tokens=0,
                model="claude-haiku-4-5",
            ),
            ToolCall(name="send_email", result={"sent": True}),
        ),
    )
    return Turn(
        id="t",
        kind="chat.turn",
        steps=steps,
        started_at=kw.pop("started_at", T0),
        ended_at=kw.pop("ended_at", T0 + timedelta(seconds=3)),
    )


def test_a_fully_wired_setup_reports_everything_live():
    cfg = Config(
        conversational_kinds=frozenset({"chat.turn"}),
        quiet_kinds=frozenset({"group.turn"}),
    )
    assert all(r.live and not r.misleading for r in coverage([wired()], cfg))


def test_no_turns_at_all():
    assert coverage([], Config())[0].live is False


def test_missing_timestamps_make_slow_turn_inert():
    r = rows([wired(started_at=None, ended_at=None)])["SLOW_TURN"]
    assert not r.live and "timestamp" in r.reason


def test_missing_tokens_make_no_cache_hit_inert():
    r = rows([wired(steps=(Generation(text="hi", cache_read_tokens=0),))])[
        "NO_CACHE_HIT"
    ]
    assert not r.live


def test_unreported_cache_makes_no_cache_hit_inert():
    """The OpenInference case: prompt sizes present, cache absent."""
    r = rows([wired(steps=(Generation(text="hi", input_tokens=9000),))])["NO_CACHE_HIT"]
    assert not r.live and "cache" in r.reason


def test_no_tool_spans_makes_all_four_tool_detectors_inert():
    got = rows([wired(steps=(Generation(text="hi", input_tokens=10),))])
    for code in ("TOOL_ERROR", "TOOL_REFUSAL", "REPEATED_TOOL", "TOOL_STORM"):
        assert not got[code].live, code


def test_unstructured_tool_results_make_refusal_inert():
    got = rows(
        [
            wired(
                steps=(
                    Generation(text="hi", input_tokens=10),
                    ToolCall(name="x", result="did not work"),
                )
            )
        ]
    )
    assert not got["TOOL_REFUSAL"].live
    # ...but the volume detectors still see the call.
    assert got["TOOL_STORM"].live


def test_a_refusal_predicate_revives_it():
    cfg = Config(refusal_predicates=(lambda r: r == "did not work",))
    got = rows(
        [
            wired(
                steps=(
                    Generation(text="hi", input_tokens=10),
                    ToolCall(name="x", result="did not work"),
                )
            )
        ],
        cfg,
    )
    assert got["TOOL_REFUSAL"].live


def test_no_reply_text_is_flagged_in_both_directions():
    """The coupled failure: the honesty detector goes silent AND the empty-reply
    detector fires on everything. One broken mapping, two wrong columns."""
    got = rows([wired(steps=(Generation(text="", input_tokens=10),))])
    assert not got["UNVERIFIED_CLAIM"].live
    assert got["EMPTY_REPLY"].misleading


def test_tool_vocabulary_mismatch_is_flagged_as_misleading_not_inert():
    """Rules that match no tool name produce FALSE POSITIVES, not silence — a real
    action reads as an unbacked claim. Louder than inertness, and just as wrong."""
    got = rows(
        [
            wired(
                steps=(
                    Generation(text="I've sent them a message.", input_tokens=10),
                    ToolCall(name="dispatch_email", result={"ok": True}),
                )
            )
        ]
    )
    assert got["UNVERIFIED_CLAIM"].misleading


def test_a_satisfying_tool_that_FAILED_still_counts_as_wired():
    """The question is whether the vocabulary lines up, not whether the tool succeeded
    in this sample. Keying on success made a real capture report a false mismatch."""
    got = rows(
        [
            wired(
                steps=(
                    Generation(text="I've sent them a message.", input_tokens=10),
                    ToolCall(name="send_email", result={"sent": False}),
                )
            )
        ]
    )
    assert not got["UNVERIFIED_CLAIM"].misleading


def test_claim_rules_removed_entirely():
    cfg = Config(claim_rules=())
    assert not rows([wired()], cfg)["UNVERIFIED_CLAIM"].live


def test_custom_rules_are_matched_against_your_own_tool_names():
    cfg = Config(
        claim_rules=(
            ClaimRule(
                name="filed",
                pattern=re.compile("filed"),
                satisfied_by=frozenset({"open_ticket"}),
            ),
        )
    )
    got = rows(
        [
            wired(
                steps=(
                    Generation(text="Filed it.", input_tokens=10),
                    ToolCall(name="open_ticket", result={"ok": True}),
                )
            )
        ],
        cfg,
    )
    assert got["UNVERIFIED_CLAIM"].live and not got["UNVERIFIED_CLAIM"].misleading


@pytest.mark.parametrize(
    "configured,live", [(frozenset({"group.turn"}), True), (frozenset(), False)]
)
def test_gate_filtered_needs_quiet_kinds(configured, live):
    assert rows([wired()], Config(quiet_kinds=configured))["GATE_FILTERED"].live is live
