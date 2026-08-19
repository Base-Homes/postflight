"""Every knob a detector reads, in one value.

The split this module enforces is the reason the package can be published at all:
a detector is MECHANISM and belongs in code; the thresholds, tool-name conventions
and claim vocabulary it reads are TUNING and belong to whoever runs it. The defaults
here are a starting vocabulary, not a claim of completeness — they are what a
tool-calling agent looks like before you have watched yours fail.

Nothing here is required. `Config()` runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

UNKNOWN_KIND = "unknown"


@dataclass(frozen=True)
class ClaimRule:
    """A thing a reply can assert, and the tools that would make it true.

    `satisfied_by` names tools exactly; `satisfied_by_prefix` matches a naming
    convention, which is what makes the shipped defaults useful before you have
    listed your own toolset. A rule with neither can never be satisfied, so it would
    flag every match — `Config` rejects that at construction rather than filling a
    report with findings no tool could ever clear.
    """

    name: str
    pattern: re.Pattern[str]
    satisfied_by: frozenset[str] = frozenset()
    satisfied_by_prefix: tuple[str, ...] = ()

    def satisfied(self, succeeded: frozenset[str]) -> bool:
        if succeeded & self.satisfied_by:
            return True
        return (
            any(name.startswith(self.satisfied_by_prefix) for name in succeeded)
            if self.satisfied_by_prefix
            else False
        )


# Verb-then-object and object-then-verb, because a model writes both ("added the
# lead" / "the lead was added"). The bounded gap keeps the pair inside one clause;
# an unbounded `.*` matches a verb in one sentence against an object two sentences
# later, which reads as a claim and is not one.
def _claim(verbs: str, objects: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?:{verbs})\b[^.\n]{{0,40}}\b(?:{objects})\b"
        rf"|\b(?:{objects})\b[^.\n]{{0,40}}\b(?:{verbs})\b",
        re.IGNORECASE,
    )


DEFAULT_CLAIM_RULES: tuple[ClaimRule, ...] = (
    ClaimRule(
        name="message_sent",
        pattern=_claim(
            "sent|texted|emailed|messaged|notified|replied\\s+to|reached\\s+out\\s+to",
            # No catch-all object here. `the\s+\w+` made every noun an object, so any
            # sentence RELAYING someone else's action ("The owner emailed you about
            # the invoice") read as the agent claiming to have sent something — a false
            # positive on the highest-signal detector, in a very ordinary sentence.
            "them|him|her|you|message|email|text|note|reminder|confirmation|details",
        ),
        satisfied_by_prefix=("send_", "email_", "notify_", "sms_", "message_", "post_"),
    ),
    ClaimRule(
        name="record_created",
        pattern=_claim(
            "created|added|saved|registered|captured|filed",
            "record|entry|ticket|issue|item|account|profile|task",
        ),
        satisfied_by_prefix=("create_", "add_", "insert_", "new_", "register_"),
    ),
    ClaimRule(
        name="record_updated",
        pattern=_claim(
            "updated|changed|moved|advanced|marked|set|closed|resolved",
            "record|entry|ticket|issue|item|status|stage|state",
        ),
        satisfied_by_prefix=(
            "update_",
            "set_",
            "edit_",
            "advance_",
            "mark_",
            "close_",
            "patch_",
            "resolve_",
        ),
    ),
    ClaimRule(
        name="scheduled",
        pattern=_claim(
            "scheduled|booked|arranged|set\\s+up",
            "meeting|call|appointment|event|visit|reminder|follow[-\\s]?up",
        ),
        satisfied_by_prefix=("schedule_", "book_", "create_calendar", "create_event"),
    ),
)

# A claim regex matches words, not polarity. "The follow-up was not sent" and "I sent
# the follow-up" both contain the same pair, and only one is a claim — the first is the
# agent being honest about NOT acting, which is the behaviour you want. Without this the
# detector punishes exactly the turns it should approve of.
DEFAULT_NEGATION = re.compile(
    r"\b(?:not|never|n't|no|without|couldn't|cannot|can't|unable\s+to|failed\s+to|"
    r"did\s+not|was\s+not|were\s+not|haven't|hasn't|wasn't|weren't)\b",
    re.IGNORECASE,
)

# A claim regex matches an action, not an ACTOR. "The owner emailed you" and "I emailed
# you" contain the same verb-object pair, and only the second is the agent claiming
# anything — the first is the agent correctly relaying what somebody else did, which is
# ordinary in any reply that summarises a thread.
#
# Anchored to `\s+$`: the subject must sit IMMEDIATELY before the verb. Allowing a word
# between them blocked "After the call I emailed them", where "the call" is an object
# two clauses back and the real subject is first-person. First-person subjects (I, we)
# are deliberately absent — those are the claims this detector exists to catch.
DEFAULT_THIRD_PARTY_SUBJECT = re.compile(
    r"\b(?:they|he|she|it|the\s+\w+|an?\s+\w+|who|which)\s+$", re.IGNORECASE
)

# Keys a tool sets to False when it ran fine and DECLINED. `is False` matters at the
# read site: a bulk create returns an integer count, and `0 == False` is True.
DEFAULT_SUCCESS_FLAGS: tuple[str, ...] = (
    "ok",
    "success",
    "updated",
    "created",
    "saved",
    "sent",
    "found",
    "deleted",
    "linked",
    "scheduled",
    "captured",
    "completed",
)

# Anthropic's minimum cacheable prefix, by model-name substring. Below it, caching is
# moot and a zero cache read means nothing. Longest match wins, so a specific version
# beats a family name.
DEFAULT_CACHE_FLOORS: dict[str, int] = {
    "opus": 512,
    "sonnet": 1024,
    "haiku": 4096,
}


def _queued_not_sent(result: Any) -> bool:
    """A handoff to a queue is not a refusal.

    `{"sent": false, "queued": true}` is a sender that cannot deliver inline and has
    handed the message to a relay. The queue working is not a refusal, and a detector
    that fires on the healthy path is one people learn to ignore. Whether the queue
    actually DRAINS is a fair question, but not one a trace can answer.
    """
    return (
        isinstance(result, dict)
        and bool(result.get("queued"))
        and result.get("sent") is False
    )


@dataclass(frozen=True)
class Config:
    # --- thresholds -------------------------------------------------------------
    slow_turn_s: float = 60.0
    tool_storm: int = 8
    repeated_tool: int = 3
    # Used when a generation's model matches nothing in `cache_floors`, and when a
    # model is unknown. Deliberately the LARGEST common floor: guessing low invents
    # NO_CACHE_HIT findings on prompts that were never cacheable.
    cache_floor_tokens: int = 4096
    cache_floors: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_CACHE_FLOORS)
    )

    # --- tool outcome classification --------------------------------------------
    success_flags: tuple[str, ...] = DEFAULT_SUCCESS_FLAGS
    error_text_prefixes: tuple[str, ...] = ("Error executing tool",)
    # Predicates that rescue a result the flag scan would otherwise call a refusal.
    refusal_exemptions: tuple[Callable[[Any], bool], ...] = (_queued_not_sent,)
    # The other direction: extra ways a result can BE a refusal, for tools that do not
    # use boolean flags. Empty by default on purpose — a status-style convention cannot
    # be guessed safely, because `{"status": "failed"}` from a `get_job_status` tool
    # describes the JOB, not the call, and scoring it as a refusal is the cry-wolf
    # failure this package exists to avoid. You know which of your tools report on
    # themselves; postflight does not.
    #
    #     Config(refusal_predicates=(
    #         lambda r: isinstance(r, dict) and r.get("status") in {"failed", "declined"},
    #     ))
    refusal_predicates: tuple[Callable[[Any], bool], ...] = ()

    # --- honesty ----------------------------------------------------------------
    claim_rules: tuple[ClaimRule, ...] = DEFAULT_CLAIM_RULES
    negation: re.Pattern[str] = DEFAULT_NEGATION
    third_party_subject: re.Pattern[str] = DEFAULT_THIRD_PARTY_SUBJECT
    # Kinds whose output NARRATES rather than speaks — a summary of someone's history
    # uses the same words a claim does ("the record was marked lost in June"), with no
    # user and no write anywhere in the turn.
    #
    # A DENY-list, deliberately, not an allow-list of conversational kinds. `kind`
    # falls back to UNKNOWN_KIND whenever the adapter cannot resolve it (a root span
    # that failed to open, sampling that dropped it), and an allow-list would silently
    # exempt those real turns — and every FUTURE kind until someone edits the set. A
    # new narrating kind going unflagged is a false positive; a new conversational kind
    # going unflagged is a missed lie.
    narrating_kinds: frozenset[str] = frozenset()

    # --- reply expectations -------------------------------------------------------
    # Kinds that owe a human a reply. Empty means "every kind" — a detector that fires
    # nowhere until configured is one nobody discovers — but while it is empty the
    # finding is reported at INFO rather than FAULT, because postflight has not been
    # told which of your surfaces actually speak to anyone. A batch job that returns a
    # document legitimately has no reply, and flagging every one of them as a fault on
    # first run is exactly how a detector gets tuned away.
    conversational_kinds: frozenset[str] = frozenset()
    # Kinds that are silent BY DESIGN — a relevance gate drops most traffic without
    # replying. A quiet kind is only suspicious when it did real WORK and still said
    # nothing: the gate passed it, the agent acted, and nobody got an answer. Otherwise
    # it reports as GATE_FILTERED, which is INFO, not a fault.
    quiet_kinds: frozenset[str] = frozenset()
    # Kinds that decide whether to ACT and whether to ANSWER separately, so a turn that
    # acts and says nothing is a designed outcome. `quiet_kinds` cannot express this: it
    # describes silence BEFORE anything happens and asserts that silence after work is
    # suspicious. Work-then-silence reports as ACTED_SILENTLY here and stays EMPTY_REPLY
    # everywhere else, so declare only the surface that genuinely acts without speaking.
    #
    # Orthogonal to `conversational_kinds`, and a kind is often both: a shared channel
    # has people in it who sometimes get an answer, AND lets the agent file work without
    # broadcasting. This field governs the turns that DID work; conversational_kinds
    # still governs the rest, so a turn here that did nothing and said nothing stays an
    # EMPTY_REPLY.
    reply_optional_kinds: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unsatisfiable = [
            r.name
            for r in self.claim_rules
            if not r.satisfied_by and not r.satisfied_by_prefix
        ]
        if unsatisfiable:
            raise ValueError(
                "claim rules with no satisfying tool would flag every match: "
                + ", ".join(unsatisfiable)
            )

    @property
    def reply_expectation_configured(self) -> bool:
        return bool(self.conversational_kinds)

    def owes_reply(self, kind: str) -> bool:
        """A quiet kind ALWAYS owes a reply.

        `quiet_kinds` describes a conversational surface sitting behind a relevance
        gate, so listing one without also listing it in `conversational_kinds` used to
        make it invisible: the reply-expectation check ran first and returned early,
        and GATE_FILTERED — the whole reason to declare a kind quiet — was never
        emitted. Nothing said the config was inert. Treat the declaration as the
        statement it obviously is instead of requiring it twice.
        """
        if kind in self.quiet_kinds:
            return True
        return not self.conversational_kinds or kind in self.conversational_kinds

    def cache_floor_for(self, model: str | None) -> int:
        if not model:
            return self.cache_floor_tokens
        lowered = model.lower()
        hits = [
            (len(key), floor)
            for key, floor in self.cache_floors.items()
            if key.lower() in lowered
        ]
        return max(hits)[1] if hits else self.cache_floor_tokens
