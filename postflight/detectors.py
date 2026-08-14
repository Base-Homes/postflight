"""The taxonomy.

Each detector is a pure `(Turn, Config) -> Iterable[Finding]`. No I/O, no vendor, no
domain. Adding one means appending to `DETECTORS`; the code string it emits is the
stable identifier callers filter and chart on, so treat a rename as a breaking change.

These are not generic observability metrics. Each names a way a tool-calling agent
fails that its own logs do not make obvious. Where a rule looks oddly narrow, the
comment says which healthy case it must not fire on: the narrowing is most of the work,
and a detector that cries wolf is one people learn to ignore.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .config import Config
from .model import Finding, Outcome, Severity, ToolCall, Turn

Detector = Callable[[Turn, Config], Iterable[Finding]]


def tool_outcome(call: ToolCall, cfg: Config) -> Outcome:
    """Did this call do the thing?

    Three answers, not two. A tool that RAISED and a tool that ran fine and DECLINED are
    different failures with different fixes, and the second is the dangerous one: every
    guard keyed on "did the tool run" is satisfied by a decline, which is how a false
    confirmation reaches a user.

    REFUSED is only detectable where the tool SAYS SO in a shape config knows about —
    a boolean success flag by default, plus whatever `refusal_predicates` adds. A tool
    that signals failure only in prose is indistinguishable from one that succeeded, and
    postflight will call it OK. That is a limit of the data, not a bug to tune around:
    inferring refusal from an empty result would flag every search that legitimately
    found nothing.
    """
    if call.is_error:
        return Outcome.ERRORED
    result = call.result
    if isinstance(result, str) and result.startswith(cfg.error_text_prefixes):
        return Outcome.ERRORED
    if isinstance(result, dict) and result.get("error"):
        return Outcome.ERRORED
    # Exemptions run first and win outright: a shape declared healthy stays healthy
    # even if a later rule would flag it.
    if any(exempt(result) for exempt in cfg.refusal_exemptions):
        return Outcome.OK
    if any(is_refusal(result) for is_refusal in cfg.refusal_predicates):
        return Outcome.REFUSED
    if isinstance(result, dict) and any(
        result.get(k) is False for k in cfg.success_flags
    ):
        return Outcome.REFUSED
    return Outcome.OK


def succeeded_tools(turn: Turn, cfg: Config) -> frozenset[str]:
    return frozenset(
        c.name for c in turn.tool_calls if tool_outcome(c, cfg) is Outcome.OK
    )


def _asserts(reply: str, rule, cfg: Config) -> bool:
    """True only for an AFFIRMATIVE, FIRST-PARTY match.

    Both checks read the clause the match sits in — the ~40 characters before it, cut
    at the nearest comma or period — rather than the whole reply, because a negation
    or a subject two clauses back is unrelated to this one.

    Negation, because "the follow-up was not sent" is the agent being honest about not
    acting. Subject, because "the owner emailed you" is the agent relaying what someone
    ELSE did; the verb-object pair is identical and only one of the two is a claim.
    """
    for match in rule.pattern.finditer(reply or ""):
        window = reply[max(0, match.start() - 40) : match.start()]
        tail = window.rsplit(".", 1)[-1].rsplit(",", 1)[-1]
        if cfg.negation.search(tail) or cfg.third_party_subject.search(tail):
            continue
        return True
    return False


def _summary(call: ToolCall) -> dict[str, Any]:
    return {"tool": call.name, "result": str(call.result)[:200]}


def detect_tool_error(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """A tool raised and the framework wrapped it."""
    hits = [c for c in turn.tool_calls if tool_outcome(c, cfg) is Outcome.ERRORED]
    if hits:
        yield Finding(
            code="TOOL_ERROR",
            turn_id=turn.id,
            message=f"{len(hits)} tool call(s) raised",
            detail={"calls": [_summary(c) for c in hits]},
        )


def detect_tool_refusal(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """A tool ran and declined inside its own result body, with no error flag.

    The shape that lets a false confirmation reach a user, because every guard that
    asks "did the tool run" is satisfied.
    """
    hits = [c for c in turn.tool_calls if tool_outcome(c, cfg) is Outcome.REFUSED]
    if hits:
        yield Finding(
            code="TOOL_REFUSAL",
            turn_id=turn.id,
            message=f"{len(hits)} tool call(s) declined in-body",
            detail={"calls": [_summary(c) for c in hits]},
        )


def detect_unverified_claim(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """The reply claims a write no successful tool backs up.

    The highest-signal detector here: it means a user was told something happened that
    did not happen. Skipped only on `narrating_kinds` — see the deny-list rationale on
    that field, which is the difference between a false positive and a missed lie.
    """
    if turn.kind in cfg.narrating_kinds:
        return
    reply = turn.reply
    if not reply.strip():
        return
    succeeded = succeeded_tools(turn, cfg)
    claimed = [
        r.name
        for r in cfg.claim_rules
        if _asserts(reply, r, cfg) and not r.satisfied(succeeded)
    ]
    if claimed:
        yield Finding(
            code="UNVERIFIED_CLAIM",
            turn_id=turn.id,
            message=f"reply claims {', '.join(claimed)} with no successful tool",
            detail={
                "claims": claimed,
                "reply_preview": reply[:300],
                "succeeded_tools": sorted(succeeded),
            },
        )


def detect_repeated_tool(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """The same tool called N+ times in one turn.

    Usually the model searching for an argument it was never given — a context gap,
    not a model failure. Fix the prompt, not the temperature.
    """
    repeats = {
        name: count
        for name, count in Counter(c.name for c in turn.tool_calls).items()
        if count >= cfg.repeated_tool
    }
    if repeats:
        yield Finding(
            code="REPEATED_TOOL",
            turn_id=turn.id,
            message=f"repeated calls: {repeats}",
            detail={"repeats": repeats},
        )


def detect_tool_storm(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """Many tool calls in one turn. Same cause as REPEATED_TOOL, worse."""
    count = len(turn.tool_calls)
    if count >= cfg.tool_storm:
        yield Finding(
            code="TOOL_STORM",
            turn_id=turn.id,
            message=f"{count} tool calls in one turn",
            detail={"tool_calls": count, "tools": [c.name for c in turn.tool_calls]},
        )


def detect_slow_turn(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """Wall clock over the threshold — usually a storm with a human waiting."""
    seconds = turn.duration_s
    if seconds >= cfg.slow_turn_s:
        yield Finding(
            code="SLOW_TURN",
            turn_id=turn.id,
            message=f"{seconds:.1f}s",
            detail={
                "duration_s": round(seconds, 1),
                "tool_calls": len(turn.tool_calls),
            },
        )


def detect_no_cache_hit(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """A prompt big enough to cache that read nothing from cache.

    Caching is a PREFIX match, so one volatile byte early in the system prompt silently
    drops the discount on every turn. Measured against the LARGEST SINGLE generation,
    never the turn's sum — caching applies per call, and a fan-out of many small prompts
    sums past any floor while no individual prompt is remotely cacheable. That
    arithmetic is the whole difference between a real finding and a fabricated one.

    Requires more than one generation: the first call in a turn has nothing to read.
    """
    gens = turn.generations
    if len(gens) < 2:
        return
    largest = max(gens, key=lambda g: g.input_tokens)
    if largest.cache_read_tokens is None:
        # The producer reports no cache usage at all, which is not the same as reporting
        # none. Firing here made every sufficiently large turn from such an
        # instrumentation a false positive — confirmed against real OpenInference spans,
        # which carry no cache attribute whatsoever.
        return
    floor = cfg.cache_floor_for(largest.model)
    if largest.input_tokens > floor and turn.cache_read_tokens == 0:
        yield Finding(
            code="NO_CACHE_HIT",
            turn_id=turn.id,
            message=f"{largest.input_tokens} input tokens, 0 read from cache",
            detail={
                "largest_input_tokens": largest.input_tokens,
                "cache_floor": floor,
                "model": largest.model,
                "generations": len(gens),
            },
        )


def detect_empty_reply(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """A turn that produced no text where somebody was owed one.

    On a `quiet_kind` — a surface fronted by a relevance gate — silence is the product
    working, and flagging it drowns the one class this detector exists for, because
    most traffic there is dropped. A quiet kind is only suspicious when the gate PASSED it
    and the agent did work: tools ran, or the loop went more than one generation, and
    then the room got nothing. Otherwise it reports as GATE_FILTERED at INFO, so the
    count stays visible for a gate that has started swallowing real traffic.
    """
    if not turn.generations or turn.reply.strip():
        return
    if not cfg.owes_reply(turn.kind):
        return
    did_work = bool(turn.tool_calls) or len(turn.generations) > 1
    if turn.kind in cfg.quiet_kinds and not did_work:
        yield Finding(
            code="GATE_FILTERED",
            turn_id=turn.id,
            severity=Severity.INFO,
            message="silent by design — gate dropped the turn without work",
        )
        return
    yield Finding(
        code="EMPTY_REPLY",
        turn_id=turn.id,
        # Until someone declares which surfaces speak, postflight cannot tell a silent
        # 1:1 channel from a batch job that returns a document, so this reports as
        # INFO. Set `conversational_kinds` and it becomes a fault.
        severity=(
            Severity.FAULT if cfg.reply_expectation_configured else Severity.INFO
        ),
        message="no reply text where one was owed",
        detail={
            "tool_calls": len(turn.tool_calls),
            "generations": len(turn.generations),
            "reply_expectation_configured": cfg.reply_expectation_configured,
        },
    )


DETECTORS: tuple[Detector, ...] = (
    detect_unverified_claim,  # first: the only one a user experiences as a lie
    detect_tool_error,
    detect_tool_refusal,
    detect_repeated_tool,
    detect_tool_storm,
    detect_empty_reply,
    detect_slow_turn,
    detect_no_cache_hit,
)


def run(
    turn: Turn, cfg: Config | None = None, detectors: Iterable[Detector] | None = None
) -> list[Finding]:
    """Every finding for one turn, in `DETECTORS` order (most user-visible first)."""
    cfg = cfg or Config()
    return [f for detector in (detectors or DETECTORS) for f in detector(turn, cfg)]


def run_all(
    turns: Iterable[Turn],
    cfg: Config | None = None,
    detectors: Iterable[Detector] | None = None,
) -> dict[str, list[Finding]]:
    cfg = cfg or Config()
    chosen = tuple(detectors or DETECTORS)
    return {turn.id: run(turn, cfg, chosen) for turn in turns}


def faults(findings: Iterable[Finding]) -> list[Finding]:
    """Findings that are something to FIX.

    Reporting must cut on this rather than on the raw finding count, or a turn counted
    as flagged for behaving correctly cries as loudly as a real one — and on a surface
    that is silent by design, those dominate the count.
    """
    return [f for f in findings if f.severity is Severity.FAULT]
