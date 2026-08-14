"""The normalized shape every detector reads.

A turn is an ORDERED SEQUENCE of steps, and that is the whole point of this package.
The failures postflight looks for happen BETWEEN observations rather than inside one:
a tool that declined three steps before the reply that contradicts it, the same read
issued eight times, a cache that never warmed across the turn's generations. Every
tracing platform's evaluator runtime today hands an evaluator a single observation and
explicitly refuses to load its siblings, which is why none of them can express these.

Adapters build `Turn`s; detectors read them. Nothing in this module knows about a
vendor, a transport, or an application domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


def _span_seconds(started_at: datetime | None, ended_at: datetime | None) -> float:
    if not started_at or not ended_at:
        return 0.0
    return max(0.0, (ended_at - started_at).total_seconds())


@dataclass(frozen=True)
class Generation:
    """One model call inside a turn.

    Token fields are separate rather than a single `usage` blob because
    `cache_read_tokens` carries a detector of its own and adapters disagree about
    where it lives — normalizing it here is the adapter's job, not the detector's.
    """

    model: str | None = None
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # None means the producer does not REPORT cache usage; 0 means it reported none.
    # Collapsing those two made every trace from an instrumentation that omits the
    # field look like a cache miss — see the NO_CACHE_HIT detector.
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def input_tokens_total(self) -> int:
        """Every input token this call consumed, cached or not.

        `input_tokens` is the UNCACHED prompt alone — providers report cache reads and
        cache writes as separate counters, so on a well-cached turn most of the real
        prompt sits outside it. Summing only `input + output` understates a cached
        agent by an order of magnitude, which is the direction that makes it look free.
        """
        return (
            self.input_tokens
            + (self.cache_read_tokens or 0)
            + (self.cache_write_tokens or 0)
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens_total + self.output_tokens

    @property
    def latency_s(self) -> float:
        return _span_seconds(self.started_at, self.ended_at)


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation and what came back.

    `is_error` is the TRANSPORT-level flag (MCP's `isError`, an exception the framework
    caught) — the thing a caller can see without reading the body. It is deliberately
    separate from a tool that ran fine and DECLINED inside its own result, which is a
    different failure with a different fix and is classified from `result` by config.
    """

    name: str
    arguments: dict[str, Any] | None = None
    result: Any = None
    is_error: bool = False
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @property
    def latency_s(self) -> float:
        return _span_seconds(self.started_at, self.ended_at)


Step = Generation | ToolCall


@dataclass(frozen=True)
class Turn:
    """One agent turn: everything that happened between an input and a reply.

    `kind` is the surface label — which agent, which channel. Detectors branch on it
    (a digest that narrates someone's history is held to different honesty rules than
    a reply to a person), so an adapter that cannot determine it must pass
    `UNKNOWN_KIND` rather than guess. Every kind-keyed rule in this package is written
    as a DENY-list against known-exempt kinds, so an unknown kind fails closed.
    """

    id: str
    kind: str = "unknown"
    steps: tuple[Step, ...] = ()
    user_id: str | None = None
    session_id: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # The source payload the adapter built this from. For renderers and drill-down
    # output ONLY — a detector that reaches in here has coupled itself to a vendor and
    # defeats the point of the normalized shape.
    raw: Any = None

    @property
    def generations(self) -> tuple[Generation, ...]:
        return tuple(s for s in self.steps if isinstance(s, Generation))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(s for s in self.steps if isinstance(s, ToolCall))

    @property
    def reply(self) -> str:
        """The text the turn ended with, which is what a human actually received.

        The LAST generation, not a join of all of them: intermediate generations in a
        tool-calling loop are the model talking to itself, and folding them into the
        reply makes every honesty detector match on reasoning the user never saw.
        """
        gens = self.generations
        return gens[-1].text if gens else ""

    @property
    def duration_s(self) -> float:
        if self.started_at and self.ended_at:
            return _span_seconds(self.started_at, self.ended_at)
        stamps = [s.started_at for s in self.steps if s.started_at]
        ends = [s.ended_at or s.started_at for s in self.steps if s.started_at]
        return _span_seconds(min(stamps), max(ends)) if stamps else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(g.total_tokens for g in self.generations)

    @property
    def cache_read_tokens(self) -> int:
        """Cache reads across the turn, counting UNREPORTED as zero.

        Fine for a total; useless for asking "did this turn miss the cache", because
        an unreported zero and a real zero add up the same. A detector wanting that
        distinction must read  per call.
        """
        return sum(g.cache_read_tokens or 0 for g in self.generations)

    @property
    def cache_write_tokens(self) -> int:
        return sum(g.cache_write_tokens or 0 for g in self.generations)


class Severity(StrEnum):
    """FAULT is something to fix. INFO is a count worth watching.

    The distinction is load-bearing in reporting, not decoration: a detector that fires
    on correct behaviour still belongs in the table (a suppression gate that starts
    swallowing real traffic shows up as a rising count), but counting it as a fault
    makes the headline cry wolf and is how the real rows get ignored.
    """

    FAULT = "fault"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    code: str
    turn_id: str
    severity: Severity = Severity.FAULT
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class Outcome(StrEnum):
    OK = "ok"
    REFUSED = "refused"
    ERRORED = "errored"
