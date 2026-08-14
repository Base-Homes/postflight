"""Detector behaviour, in both directions.

Every "does NOT fire" case pins a healthy pattern the rule must stay clear of. Those are
the half worth keeping green: a detector is easy to make fire and hard to keep quiet.
"""
import re
from datetime import datetime, timedelta, timezone

import pytest

from postflight import (ClaimRule, Config, Finding, Generation, Outcome, Severity,
                        ToolCall, Turn, faults, run, tool_outcome)

T0 = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


def turn(*steps, kind="chat.turn", **kw) -> Turn:
    return Turn(id="t1", kind=kind, steps=tuple(steps), started_at=T0,
                ended_at=T0 + timedelta(seconds=kw.pop("seconds", 1)), **kw)


def gen(text="ok", **kw) -> Generation:
    return Generation(text=text, model=kw.pop("model", "claude-haiku-4-5"), **kw)


def tool(name, result=None, **kw) -> ToolCall:
    return ToolCall(name=name, result=result, **kw)


def codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


# --- tool outcome ---------------------------------------------------------------

def test_transport_error_is_errored():
    assert tool_outcome(tool("x", is_error=True), Config()) is Outcome.ERRORED


def test_error_text_prefix_is_errored():
    call = tool("x", result="Error executing tool x: boom")
    assert tool_outcome(call, Config()) is Outcome.ERRORED


def test_in_body_decline_is_refused_not_errored():
    assert tool_outcome(tool("x", result={"updated": False}), Config()) is Outcome.REFUSED


def test_zero_count_is_not_a_refusal():
    """A bulk create returns an integer, and `0 == False` is True in Python."""
    assert tool_outcome(tool("x", result={"created": 0}), Config()) is Outcome.OK


def test_queued_handoff_is_not_a_refusal():
    call = tool("send", result={"sent": False, "queued": True, "outbox_id": "1"})
    assert tool_outcome(call, Config()) is Outcome.OK


# --- unverified claim -----------------------------------------------------------

def test_claim_without_tool_flags():
    found = run(turn(gen("I've sent them a message about the repair.")))
    assert "UNVERIFIED_CLAIM" in codes(found)


def test_claim_with_matching_tool_prefix_is_clean():
    found = run(turn(tool("send_email", result={"sent": True}),
                     gen("I've emailed them the details.")))
    assert "UNVERIFIED_CLAIM" not in codes(found)


def test_claim_backed_by_a_REFUSED_tool_still_flags():
    """The whole point: the tool ran, so every did-it-run guard is satisfied."""
    found = run(turn(tool("send_email", result={"sent": False}),
                     gen("I've emailed them the details.")))
    assert "UNVERIFIED_CLAIM" in codes(found)


def test_negated_claim_is_not_a_claim():
    found = run(turn(gen("I could not send them a message — no phone on file.")))
    assert "UNVERIFIED_CLAIM" not in codes(found)


def test_negation_two_clauses_back_does_not_rescue_a_claim():
    found = run(turn(gen("The address was not on file. I've sent them a message.")))
    assert "UNVERIFIED_CLAIM" in codes(found)


def test_relayed_third_party_action_is_not_a_claim():
    """"The owner emailed you" is the agent reporting what somebody ELSE did. The
    verb-object pair is identical to a real claim; only the subject differs."""
    found = run(turn(gen("The owner emailed you about the invoice.")))
    assert "UNVERIFIED_CLAIM" not in codes(found)


@pytest.mark.parametrize("reply", [
    "I've sent them a message.",
    "Sent them a message just now.",
    # The subject guard must be adjacent: "the call" here is an object two clauses
    # back, and the real subject is first-person.
    "After the call I emailed them the details.",
])
def test_first_party_claims_survive_the_subject_guard(reply):
    assert "UNVERIFIED_CLAIM" in codes(run(turn(gen(reply))))


def test_narrating_kind_is_exempt():
    cfg = Config(narrating_kinds=frozenset({"digest.turn"}))
    found = run(turn(gen("A message was sent to them in June."), kind="digest.turn"), cfg)
    assert "UNVERIFIED_CLAIM" not in codes(found)


def test_unknown_kind_is_NOT_exempt():
    """The deny-list's reason for existing: an unresolved root must fail closed."""
    cfg = Config(narrating_kinds=frozenset({"digest.turn"}))
    found = run(turn(gen("I've sent them a message."), kind="unknown"), cfg)
    assert "UNVERIFIED_CLAIM" in codes(found)


def test_custom_claim_rule():
    cfg = Config(claim_rules=(ClaimRule(
        name="ticket_filed",
        pattern=re.compile(r"\bfiled\b[^.\n]{0,40}\bticket\b", re.IGNORECASE),
        satisfied_by=frozenset({"create_ticket"}),
    ),))
    assert "UNVERIFIED_CLAIM" in codes(run(turn(gen("Filed a ticket for you.")), cfg))
    clean = run(turn(tool("create_ticket", result={"ok": True}),
                     gen("Filed a ticket for you.")), cfg)
    assert "UNVERIFIED_CLAIM" not in codes(clean)


def test_unsatisfiable_claim_rule_is_rejected():
    with pytest.raises(ValueError, match="no satisfying tool"):
        Config(claim_rules=(ClaimRule(name="x", pattern=re.compile("x")),))


# --- volume + timing ------------------------------------------------------------

def test_repeated_and_storm():
    found = run(turn(*[tool("get_thing", result={"ok": True}) for _ in range(8)],
                     gen("done")))
    assert {"REPEATED_TOOL", "TOOL_STORM"} <= codes(found)


def test_slow_turn_uses_configured_threshold():
    slow = turn(gen("done"), seconds=45)
    assert "SLOW_TURN" not in codes(run(slow))
    assert "SLOW_TURN" in codes(run(slow, Config(slow_turn_s=30.0)))


# --- cache ----------------------------------------------------------------------

def test_no_cache_hit_on_a_big_repeat_prompt():
    """Explicit zeros: the producer REPORTED no cache reads, which is a real miss."""
    found = run(turn(gen("", input_tokens=9000, cache_read_tokens=0),
                     gen("done", input_tokens=9000, cache_read_tokens=0)))
    assert "NO_CACHE_HIT" in codes(found)


def test_unreported_cache_is_not_a_miss():
    """Unknown is not zero. Some instrumentations emit no cache attribute at all, and
    scoring that as a miss makes every large turn from them a false positive."""
    found = run(turn(gen("", input_tokens=9000), gen("done", input_tokens=9000)))
    assert "NO_CACHE_HIT" not in codes(found)
    assert Turn(id="x", steps=(gen("d", input_tokens=9000),)).cache_read_tokens == 0


def test_cache_read_clears_it():
    found = run(turn(gen("", input_tokens=9000, cache_read_tokens=0),
                     gen("done", input_tokens=9000, cache_read_tokens=8000)))
    assert "NO_CACHE_HIT" not in codes(found)


def test_single_generation_never_flags():
    """The first call in a turn has nothing to read from."""
    assert "NO_CACHE_HIT" not in codes(run(turn(gen("done", input_tokens=90000))))


def test_many_small_prompts_do_not_sum_past_the_floor():
    """Caching is per call. Summing is the arithmetic that fabricates this finding."""
    found = run(turn(*[gen("d", input_tokens=1500, cache_read_tokens=0)
                       for _ in range(27)]))
    assert "NO_CACHE_HIT" not in codes(found)


def test_cache_floor_is_model_aware():
    """1500 tokens is cacheable on Opus and not on Haiku."""
    steps = (gen("", input_tokens=1500, model="claude-opus-5", cache_read_tokens=0),
             gen("done", input_tokens=1500, model="claude-opus-5", cache_read_tokens=0))
    assert "NO_CACHE_HIT" in codes(run(turn(*steps)))


# --- reply --------------------------------------------------------------------

def test_empty_reply_flags_on_a_conversational_kind():
    cfg = Config(conversational_kinds=frozenset({"chat.turn"}))
    found = run(turn(gen("")), cfg)
    assert "EMPTY_REPLY" in codes(found)
    assert found[0].severity is Severity.FAULT


def test_empty_reply_is_info_until_surfaces_are_declared():
    """Unconfigured, postflight cannot tell a silent 1:1 channel from a batch job that
    returns a document — so it counts them rather than calling them all faults."""
    found = run(turn(gen("")))
    assert codes(found) == {"EMPTY_REPLY"}
    assert found[0].severity is Severity.INFO
    assert faults(found) == []


def test_quiet_kind_with_no_work_is_info_not_fault():
    cfg = Config(quiet_kinds=frozenset({"group.turn"}))
    found = run(turn(gen(""), kind="group.turn"), cfg)
    assert codes(found) == {"GATE_FILTERED"}
    assert found[0].severity is Severity.INFO
    assert faults(found) == []


def test_quiet_kind_works_without_being_listed_as_conversational_too():
    """Declaring a kind quiet is the statement; requiring it twice made the config
    silently inert and dropped GATE_FILTERED entirely."""
    cfg = Config(conversational_kinds=frozenset({"chat.turn"}),
                 quiet_kinds=frozenset({"group.turn"}))
    assert codes(run(turn(gen(""), kind="group.turn"), cfg)) == {"GATE_FILTERED"}
    worked = run(turn(tool("get_thing", result={"ok": True}), gen(""),
                      kind="group.turn"), cfg)
    assert "EMPTY_REPLY" in codes(worked)


def test_quiet_kind_that_did_work_and_said_nothing_is_a_fault():
    cfg = Config(quiet_kinds=frozenset({"group.turn"}))
    found = run(turn(tool("get_thing", result={"ok": True}), gen(""),
                     kind="group.turn"), cfg)
    assert "EMPTY_REPLY" in codes(found)


def test_non_conversational_kind_owes_nothing():
    cfg = Config(conversational_kinds=frozenset({"chat.turn"}))
    assert "EMPTY_REPLY" not in codes(run(turn(gen(""), kind="cron.turn"), cfg))


# --- model ----------------------------------------------------------------------

def test_total_tokens_counts_cached_input():
    """`input_tokens` is the UNCACHED prompt alone, so input+output omits everything the
    cache served — the direction that hides spend rather than exaggerating it."""
    g = gen("hi", input_tokens=200, output_tokens=50, cache_read_tokens=18_000)
    assert g.input_tokens_total == 18_200
    assert Turn(id="x", steps=(g,)).total_tokens == 18_250


def test_reply_is_the_last_generation_not_a_join():
    """Intermediate generations are the model talking to itself."""
    assert turn(gen("thinking about it"), gen("here you go")).reply == "here you go"


def test_duration_falls_back_to_step_stamps():
    step = Generation(text="x", started_at=T0, ended_at=T0 + timedelta(seconds=12))
    assert Turn(id="t", steps=(step,)).duration_s == 12.0
