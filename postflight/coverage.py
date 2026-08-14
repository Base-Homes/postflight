"""Which detectors can actually fire on your data, and which are silently inert.

Every detector reads some input an adapter has to supply. When that input is missing,
the detector does not error — it simply never fires, and an empty column reads exactly
like a clean agent. That is the worst failure mode this package has: the report looks
best when it is working least.

`coverage()` answers the question the report cannot. Run it once when wiring up a new
adapter or config, and again if a detector's count goes to zero and stays there.

    for row in coverage(turns, cfg):
        if not row.live:
            print(f"{row.code}: INERT — {row.reason}")

It reports **structural** inertness only: an input the detector needs is absent from
every turn, so no possible trace could trip it. It deliberately does NOT infer inertness
from a zero count — a tool that never errored is a healthy agent, not a broken detector,
and conflating those would recreate the problem one level up.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .config import Config
from .model import Turn


@dataclass(frozen=True)
class Coverage:
    code: str
    live: bool
    reason: str
    # True when the detector CAN fire but is mis-wired in a way that produces false
    # positives rather than silence — louder, but just as wrong.
    misleading: bool = False

    def __str__(self) -> str:
        state = "MISLEADING" if self.misleading else ("live" if self.live else "INERT")
        return f"{self.code}: {state} - {self.reason}"


def coverage(turns: Iterable[Turn], cfg: Config | None = None) -> list[Coverage]:
    """One row per detector, in the order a reader should worry about them."""
    cfg = cfg or Config()
    turns = list(turns)
    if not turns:
        return [Coverage("*", False, "no turns supplied")]

    generations = [g for t in turns for g in t.generations]
    tool_calls = [c for t in turns for c in t.tool_calls]
    replies = [t for t in turns if t.reply.strip()]
    # ALL tool names, not just the ones that succeeded here. The question this answers
    # is whether your rules and your tool vocabulary line up — a satisfying tool that
    # happened to fail in this sample says nothing about the wiring.
    seen_tools = {c.name for c in tool_calls}

    rows: list[Coverage] = []
    n = len(turns)

    # --- UNVERIFIED_CLAIM ----------------------------------------------------------
    if not cfg.claim_rules:
        rows.append(Coverage("UNVERIFIED_CLAIM", False, "no claim_rules configured"))
    elif not replies:
        rows.append(
            Coverage(
                "UNVERIFIED_CLAIM",
                False,
                "no turn has reply text — check the adapter extracts the final message. "
                "This also makes EMPTY_REPLY fire on everything",
            )
        )
    elif tool_calls and not any(r.satisfied(seen_tools) for r in cfg.claim_rules):
        rows.append(
            Coverage(
                "UNVERIFIED_CLAIM",
                True,
                "no tool name in this data satisfies any claim rule (saw "
                f"{len(seen_tools)} distinct tools). A genuine action will read as an "
                "unbacked claim. Check satisfied_by / satisfied_by_prefix against your "
                "tool names",
                misleading=True,
            )
        )
    else:
        rows.append(
            Coverage(
                "UNVERIFIED_CLAIM",
                True,
                f"{len(replies)}/{n} turns have reply text; "
                f"{len(cfg.claim_rules)} claim rule(s) matched to your tools",
            )
        )

    # --- the tool detectors --------------------------------------------------------
    if not tool_calls:
        for code in ("TOOL_ERROR", "TOOL_REFUSAL", "REPEATED_TOOL", "TOOL_STORM"):
            rows.append(
                Coverage(
                    code,
                    False,
                    "no tool calls in any turn — check the adapter maps tool spans",
                )
            )
    else:
        rows.append(
            Coverage("TOOL_ERROR", True, f"{len(tool_calls)} tool call(s) visible")
        )
        structured = any(isinstance(c.result, dict) for c in tool_calls)
        if structured or cfg.refusal_predicates:
            rows.append(
                Coverage(
                    "TOOL_REFUSAL",
                    True,
                    "tool results are structured; success_flags apply",
                )
            )
        else:
            rows.append(
                Coverage(
                    "TOOL_REFUSAL",
                    False,
                    "no tool result is a dict, so success_flags can never match. If your "
                    "tools signal failure another way, add a refusal_predicate",
                )
            )
        for code in ("REPEATED_TOOL", "TOOL_STORM"):
            rows.append(Coverage(code, True, f"{len(tool_calls)} tool call(s) visible"))

    # --- EMPTY_REPLY / GATE_FILTERED -----------------------------------------------
    if not generations:
        rows.append(Coverage("EMPTY_REPLY", False, "no generations in any turn"))
    elif not replies:
        rows.append(
            Coverage(
                "EMPTY_REPLY",
                True,
                "NO turn has reply text, so this will fire on all of them — that is far "
                "more likely to be an adapter that does not extract the reply than an "
                "agent that never speaks",
                misleading=True,
            )
        )
    else:
        rows.append(
            Coverage(
                "EMPTY_REPLY",
                True,
                "reply text present; "
                + (
                    "scored as a fault"
                    if cfg.reply_expectation_configured
                    else "reported at INFO until conversational_kinds is set"
                ),
            )
        )
    rows.append(
        Coverage(
            "GATE_FILTERED",
            bool(cfg.quiet_kinds),
            "quiet_kinds configured"
            if cfg.quiet_kinds
            else "no quiet_kinds configured, so nothing is silent by design",
        )
    )

    # --- SLOW_TURN -----------------------------------------------------------------
    timed = [t for t in turns if t.duration_s > 0]
    rows.append(
        Coverage(
            "SLOW_TURN",
            bool(timed),
            f"{len(timed)}/{n} turns have a duration"
            if timed
            else "no turn has usable timestamps — check the adapter maps "
            "start/end times",
        )
    )

    # --- NO_CACHE_HIT ---------------------------------------------------------------
    sized = [g for g in generations if g.input_tokens > 0]
    reported = [g for g in generations if g.cache_read_tokens is not None]
    if not sized:
        rows.append(
            Coverage("NO_CACHE_HIT", False, "no generation reports input tokens")
        )
    elif not reported:
        rows.append(
            Coverage(
                "NO_CACHE_HIT",
                False,
                "no generation reports cache usage, and unknown is not treated as zero",
            )
        )
    else:
        rows.append(
            Coverage(
                "NO_CACHE_HIT",
                True,
                f"{len(reported)}/{len(generations)} generations report cache",
            )
        )

    return rows
