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

## What it actually prints

Run it against the trace shipped in `tests/fixtures/` — a real OpenInference capture of a
support agent whose notification tool declined:

```
0x4069cd95…  TOOL_REFUSAL  1 tool call(s) declined in-body
```

The `detail` dict is the part you act on:

```json
{"calls": [{"tool": "send_notification",
            "result": "{'sent': False, 'reason': 'channel unavailable'}"}]}
```

Note what did **not** fire. The agent's reply said *"I wasn't able to send the
notification"* — an honest report of a failure, not a claim — so `UNVERIFIED_CLAIM`
stayed quiet. Had it said "I've let the customer know", that same turn would have
produced the finding you actually want to be paged about.

Across a day of one production deployment (114 turns), the shape looks like this:

| pattern | turns | |
|---|---|---|
| `SLOW_TURN` | 22 | |
| `GATE_FILTERED` | 19 | *not a fault — silence by design* |
| `NO_CACHE_HIT` | 7 | |
| `TOOL_ERROR` | 3 | |
| `TOOL_STORM` | 2 | |
| `TOOL_REFUSAL` | 1 | |
| `REPEATED_TOOL` | 1 | |

The long tail is the point. `TOOL_REFUSAL` fires once in a day and is worth more than the
twenty-two slow turns, because a slow turn is visible to whoever waited and a silent
in-body decline is not.

## Writing an adapter

Detectors never see your trace format. They read `Turn`, so an adapter is a function from
whatever you have to an ordered sequence of steps:

```python
from postflight.model import Generation, ToolCall, Turn

Turn(
    id="…",                 # your trace/turn identifier
    kind="chat.turn",       # the SURFACE — which agent, which channel
    steps=(                 # ORDERED. the order is the signal
        Generation(text="", input_tokens=900, model="…"),
        ToolCall(name="search", result={"count": 0}),
        Generation(text="I couldn't find it.", input_tokens=1200, model="…"),
    ),
)
```

That is the whole contract. Four notes, each of which cost something to learn:

- **Order matters more than nesting.** A turn is a sequence, not a tree. That is what
  lets a detector see a tool that declined three steps before the reply contradicting it
   — the thing a per-observation evaluator structurally cannot do. Flatten your tree.
- **`ToolCall.is_error` is the TRANSPORT flag only** — an exception, an `isError`, an
  `ERROR` span status. A tool that ran fine and declined in its own body is *not* an
  error; leave `is_error=False`, put the body in `result`, and `success_flags` will
  classify it. Conflating the two hides the more dangerous failure.
- **Unknown is not zero.** If your producer does not report cache usage, leave
  `cache_read_tokens=None`. Passing `0` asserts a cache miss, and `NO_CACHE_HIT` will
  believe you. (This is exactly how the OTel adapter got it wrong first.)
- **`kind` should fall back to `"unknown"`, never to a guess.** Every kind-keyed rule in
  this package is a deny-list so that unknown fails *closed*; a plausible-looking default
  would quietly exempt the turns you most want checked.

`postflight/adapters/otel.py` is ~150 lines and is the one to copy — it deals with
flattened attributes, a span tree, and two timestamp encodings, so most of the awkward
cases are already worked out there.

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
