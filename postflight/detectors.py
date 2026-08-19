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


def _canonical(value: Any) -> Any:
    """Hashable, key-order-insensitive form of an argument or result payload.

    Dict key order is a serialisation artifact, so two calls differing only in key
    order have to produce the same key. Sorts on the key alone: sorting on the pair
    compares canonicalised values whenever two keys tie, and those are not always
    mutually comparable.
    """
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((str(k), _canonical(v)) for k, v in value.items()),
                key=lambda kv: kv[0],
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(repr(_canonical(v)) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _is_empty(result: Any) -> bool:
    """True for every shape of nothing: None, or an empty string or container."""
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    if isinstance(result, (bytes, list, tuple, dict, set, frozenset)):
        return len(result) == 0
    return False


def _argument_key(call: ToolCall) -> Any:
    """The grouping key for one call's arguments.

    `arguments is None` means the adapter maps none, not that the call had none. Every
    call of a tool then shares one key, which is name-only keying: the detector degrades
    to what it did before rather than going silent.
    """
    return None if call.arguments is None else _canonical(call.arguments)


def _worth_nothing(call: ToolCall, cfg: Config) -> bool:
    """True when a call yielded no payload: it errored, declined, or came back empty."""
    return tool_outcome(call, cfg) is not Outcome.OK or _is_empty(call.result)


# `{}`, `[]`, `""` and `None` are one answer in different clothes, so they share a key.
_NOTHING = object()


def _result_key(result: Any) -> Any:
    return _NOTHING if _is_empty(result) else _canonical(result)


def detect_repeated_tool(turn: Turn, cfg: Config) -> Iterator[Finding]:
    """The same tool called N+ times for the same thing.

    Usually the model searching for an argument it was never given — a context gap,
    not a model failure. Fix the prompt, not the temperature.

    Two keys, because a name-only count cannot separate a thrash from fan-out over N
    ids the input supplied.

    `arguments` groups calls that asked for the same thing, falling back to name-only
    where the adapter maps no arguments.

    `results` groups calls that got the same nothing, whatever they asked for: a thrash
    usually varies one id per attempt, so arguments alone would miss it. Restricted to
    calls that errored, declined or came back empty, because N identical success bodies
    are a bulk write rather than a thrash. The cost of that arm is N searches in one
    turn that legitimately found nothing, which reads the same from a trace.
    """
    calls = turn.tool_calls
    if not calls:
        return

    repeats: dict[str, int] = {}
    basis: dict[str, list[str]] = {}

    def record(name: str, count: int, why: str) -> None:
        repeats[name] = max(repeats.get(name, 0), count)
        if why not in basis.setdefault(name, []):
            basis[name].append(why)

    for (name, _), count in Counter((c.name, _argument_key(c)) for c in calls).items():
        if count >= cfg.repeated_tool:
            record(name, count, "arguments")

    for (name, _), count in Counter(
        (c.name, _result_key(c.result)) for c in calls if _worth_nothing(c, cfg)
    ).items():
        if count >= cfg.repeated_tool:
            record(name, count, "results")

    if repeats:
        yield Finding(
            code="REPEATED_TOOL",
            turn_id=turn.id,
            message=f"repeated calls: {repeats}",
            detail={"repeats": repeats, "basis": basis},
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

    Two declarations carve out designed silence, and they answer different questions.

    `reply_optional_kinds` answers "is acting without answering a designed outcome
    here?" — that turn reports as ACTED_SILENTLY at INFO. Checked before the
    reply-expectation gate, so declaring the kind is the whole statement; it need not
    also be listed as conversational, and often is, since a surface can hold people who
    sometimes get an answer and still let the agent act without broadcasting.

    `quiet_kinds` answers "does a relevance gate drop most traffic here?" — silence
    WITHOUT work is that gate working and reports as GATE_FILTERED at INFO, keeping the
    count visible for a gate that has started swallowing real traffic.

    Everything left is a fault: tools ran, or a gate passed a turn, and a waiting person
    got nothing back.
    """
    if not turn.generations or turn.reply.strip():
        return
    did_work = bool(turn.tool_calls) or len(turn.generations) > 1
    if turn.kind in cfg.reply_optional_kinds and did_work:
        yield Finding(
            code="ACTED_SILENTLY",
            turn_id=turn.id,
            severity=Severity.INFO,
            message="acted without replying — declared reply-optional surface",
            detail={
                "tool_calls": len(turn.tool_calls),
                "generations": len(turn.generations),
            },
        )
        return
    if not cfg.owes_reply(turn.kind):
        return
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
