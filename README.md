# postflight

**Turn-level failure detection for tool-calling agents.**

Your evals score what the model *said*. These are the failures that happen in the gaps
between what it said and what it did — a tool that declined three steps before the reply
that contradicts it, the same read issued eight times, a cache that never warmed. They
are invisible to an evaluator scoped to one observation, which is what every tracing
platform's evaluator runtime gives you today.

## The taxonomy

| Code | What it means | Why it matters |
|---|---|---|
| `UNVERIFIED_CLAIM` | The reply asserts a write that no successful tool backs up. | The only one a user experiences directly as a lie. They were told something happened that did not happen. |
| `TOOL_ERROR` | A tool raised; the framework wrapped it. | The visible half of tool failure. Usually already in your dashboards. |
| `TOOL_REFUSAL` | A tool ran fine and **declined in its own result body** — `{"updated": false}` — with no error flag. | The dangerous half. Every guard that asks "did the tool run" is satisfied, so a false confirmation sails through. |
| `REPEATED_TOOL` | The same tool called 3+ times in one turn. | The model is searching for an argument it was never given. A context gap, not a model failure — fix the prompt. |
| `TOOL_STORM` | 8+ tool calls in one turn. | Same cause, worse. Cost and latency both. |
| `EMPTY_REPLY` | The turn produced no text where somebody was owed one. | On a 1:1 channel this is the "it just didn't respond" bug. Reports at INFO until you set `conversational_kinds` — unconfigured, postflight can't tell a silent channel from a batch job that returns a document. |
| `GATE_FILTERED` | A turn a relevance gate dropped without doing work. | **Information, not a fault.** Silence is the design. Watch the count for a gate that has started swallowing real traffic. |
| `SLOW_TURN` | Wall clock over the threshold. | Usually a storm with a human waiting. |
| `NO_CACHE_HIT` | A prompt big enough to cache that read nothing from cache. | Caching is a prefix match, so one volatile byte early in the system prompt silently drops the discount on *every* turn. |

The codes are the stable interface. Filter on them, chart them, page on them.

## Use

```python
from postflight import Config, faults, run_all
from postflight.adapters.langfuse import LangfuseAdapter, LangfuseClient

client = LangfuseClient(host, public_key, secret_key)
turns = LangfuseAdapter().turns(client.observations(hours=24))

for turn_id, findings in run_all(turns, Config()).items():
    for finding in faults(findings):
        print(turn_id, finding.code, finding.message)
```

No dependencies. Python 3.11+.

## Tuning it to your agent

The detectors are mechanism; the vocabulary is yours. Everything below is a `Config`
field, and the shipped defaults are a starting point, not a claim of completeness —
they are what a tool-calling agent looks like before you have watched *yours* fail.

```python
Config(
    slow_turn_s=30.0,
    tool_storm=6,
    # Tools that decline in-body, by the key they set to False.
    success_flags=("ok", "updated", "sent", "created"),
    # Surfaces that owe a human a reply. Setting this is what promotes EMPTY_REPLY from
    # INFO to a fault — leave it empty and postflight cannot tell a silent channel from
    # a batch job that returns a document, so it counts them instead of blaming them.
    conversational_kinds=frozenset({"chat.turn", "inbound.turn", "group.turn"}),
    # Surfaces that narrate rather than speak. A digest summarising someone's history
    # uses the same words a claim does, with no user and no write in the turn.
    narrating_kinds=frozenset({"digest.turn"}),
    # Surfaces fronted by a relevance gate, where silence is correct. A quiet kind is
    # conversational by definition — you do not have to list it in both.
    quiet_kinds=frozenset({"group.turn"}),
)
```

The one worth real attention is `claim_rules`, which drives `UNVERIFIED_CLAIM`. A rule
pairs a regex against the tools that would make the claim true:

```python
from postflight import ClaimRule, Config
import re

Config(claim_rules=(
    ClaimRule(
        name="ticket_filed",
        pattern=re.compile(r"\b(?:filed|opened|created)\b[^.\n]{0,40}\bticket\b",
                           re.IGNORECASE),
        satisfied_by=frozenset({"create_ticket", "escalate_to_support"}),
    ),
))
```

Matching reads the clause the match sits in, and skips it on two conditions:

- **Negation.** "The follow-up was **not** sent" is the agent being honest about not
  acting, and scoring that as a lie punishes exactly the behaviour you want.
- **A third-party subject.** "**The owner** emailed you" is the agent relaying what
  someone else did. The verb-object pair is identical to a real claim; only the subject
  differs, and relaying is ordinary in any reply that summarises a thread.

Both are `Config` regexes (`negation`, `third_party_subject`) if your replies read
differently.

## Two design rules worth knowing before you extend it

**Kind-keyed exemptions are deny-lists, never allow-lists.** `Turn.kind` falls back to
`"unknown"` whenever the adapter cannot resolve it — a root span that failed to open,
sampling that dropped it. An allow-list of "surfaces we check" silently exempts those
real turns, and every future surface until someone remembers to edit the set. A new
narrating surface going unflagged is a false positive; a new conversational surface
going unflagged is a missed lie.

**Report on faults, not on findings.** `GATE_FILTERED` is `Severity.INFO` because it
fires on correct behaviour. Counting it as a fault makes the headline cry wolf, and a
detector that cries wolf on the healthy case is how the real rows get ignored. Use
`faults()` for anything a human reads first.

## Status

Alpha, and private while it earns its keep. The taxonomy is the product, so detector
codes are treated as a breaking change to rename; thresholds and vocabulary are not.

**Adapters: Langfuse and OpenTelemetry / OpenInference.** The detectors never see a
vendor — they read `postflight.model.Turn`, so an adapter is just a function from your
trace format to an ordered list of `Generation` and `ToolCall` steps.

Portability is now demonstrated rather than asserted: the OTel adapter was written
against real spans from an Anthropic agent instrumented with OpenInference — a producer
that shares nothing with the first one — and the shipped defaults caught a `TOOL_REFUSAL`
in it while correctly declining to flag the model's own negated sentence. `Turn` needed
no change to accept it.

It did surface one real modelling bug, which is the point of trying: OpenInference emits
no cache attribute at all, and scoring that absence as `0` made every large-prompt turn a
false `NO_CACHE_HIT`. `Generation.cache_read_tokens` is now `int | None` — **unknown is
not zero** — and an adapter that cannot report cache usage says so. Worth knowing if you
write the third adapter.

Issues and PRs welcome; no response SLA.

Apache 2.0 — see [LICENSE](LICENSE).
