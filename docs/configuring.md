# Configuring postflight

The detectors are mechanism. The vocabulary, the thresholds and the surface names
are yours, and they live here. Read this when you adopt, not before.

## Writing an adapter

The `Turn` contract is in the [README](../README.md#the-turn-contract). Three notes on
filling it in, each of which cost something to learn:

- **`ToolCall.is_error` is the TRANSPORT flag only**: an exception, an `isError`, an
  `ERROR` span status. A tool that ran fine and declined in its own body is *not* an
  error. Leave `is_error=False`, put the body in `result`, and let `success_flags`
  classify it. Conflating the two hides the more dangerous failure.
- **Unknown is not zero.** If your producer does not report cache usage, leave
  `cache_read_tokens=None`. Passing `0` asserts a cache miss and `NO_CACHE_HIT` will
  believe you.
- **`kind` falls back to `"unknown"`, never to a guess.** See the deny-list rule below
  for why a plausible-looking default is the dangerous option.

Adapters must not add a dependency. Parse exported JSON rather than importing a vendor
SDK: the producer needs it, the reader does not.

## Tuning it to your agent

The detectors are mechanism; the vocabulary is yours. Everything below is a `Config`
field, and the shipped defaults are a starting point, not a claim of completeness.
they are what a tool-calling agent looks like before you have watched *yours* fail.

```python
Config(
    slow_turn_s=30.0,
    tool_storm=6,
    # Tools that decline in-body, by the key they set to False.
    success_flags=("ok", "updated", "sent", "created"),
    # Surfaces that owe a human a reply. Setting this is what promotes EMPTY_REPLY from
    # INFO to a fault. Leave it empty and postflight cannot tell a silent channel from
    # a batch job that returns a document, so it counts them instead of blaming them.
    conversational_kinds=frozenset({"chat.turn", "inbound.turn", "group.turn"}),
    # Surfaces that narrate rather than speak. A digest summarising someone's history
    # uses the same words a claim does, with no user and no write in the turn.
    narrating_kinds=frozenset({"digest.turn"}),
    # Surfaces fronted by a relevance gate, where silence is correct. A quiet kind is
    # conversational by definition, so you need not list it in both.
    quiet_kinds=frozenset({"group.turn"}),
)
```

### What `TOOL_REFUSAL` can and cannot see

This detector does **not** assume your tools return `{"updated": false}`. It assumes you
tell it how your tools say no. Out of the box it recognises four shapes:

| shape | verdict |
|---|---|
| the call raised (`is_error` set by the adapter) | `TOOL_ERROR` |
| the framework's error string (`Error executing tool …`) | `TOOL_ERROR` |
| a truthy `error` key in the result | `TOOL_ERROR` |
| a key from `success_flags` set to `false` | `TOOL_REFUSAL` |

Anything else reads as success. If your tools signal failure some other way, whether
a `status` field, an enum or an HTTP-ish code, then **`TOOL_REFUSAL` will never fire and your
report will look clean**. Add your convention:

```python
Config(
    refusal_predicates=(
        lambda r: isinstance(r, dict) and r.get("status") in {"failed", "declined"},
    )
)
```

There is no default for that, deliberately. `{"status": "failed"}` returned by a
`get_job_status` tool describes the *job*, not the call, and guessing would make every
healthy status read into a refusal, which is precisely the cry-wolf failure this package
exists to avoid. You know which of your tools report on themselves; postflight doesn't.

Two things it will never infer, by design: an **empty result set** (a search that found
nothing is not a decline) and **prose** (`"No matching orders found."` is
indistinguishable from success without reading it). If a tool of yours only fails in
prose, the durable fix is in the tool, not here.

#### Reading a refusal

A `TOOL_REFUSAL` is not automatically a bug in the tool. Most in-body declines are
expected outcomes rather than defects, and the reason is often information the agent
needs in order to do something sensible next.

What is worth checking is whether the decline was *handled*. A refusal on the same turn
as an `UNVERIFIED_CLAIM` is the pairing that matters: the tool said no, and the reply
said yes, which means a user was told something untrue. A refusal that the reply reports
honestly is the system working, and the fixture in this repository is exactly that case.

The one shape worth fixing at the source is a tool swallowing a genuine failure into a
result body, a 500 returned as `{"ok": false}`. That is an error wearing a decline's
clothes, and it belongs in `TOOL_ERROR` where your existing alerting can see it.

`refusal_exemptions` is the other direction: shapes that look like refusals and are
not. The shipped one is `{"sent": false, "queued": true}`: a send handed off to a relay.
Exemptions outrank `refusal_predicates`, so widening your detection cannot silently
re-flag a path you already excused.

### What the other detectors depend on

Same class of problem, and the reason `coverage()` exists: a detector whose input is
missing does not error, it just never fires, and an empty column reads exactly like a
clean agent.

| detector | goes quiet if | goes *wrong* if |
|---|---|---|
| `UNVERIFIED_CLAIM` | the adapter supplies no reply text, or your replies are not in the vocabulary `claim_rules` knows (they are English by default) | your tool names don't match `satisfied_by` / `satisfied_by_prefix`, and a genuine action then reads as an unbacked claim |
| `TOOL_ERROR` · `TOOL_REFUSAL` · `REPEATED_TOOL` · `TOOL_STORM` | the adapter maps no tool spans | |
| `SLOW_TURN` | the adapter supplies no timestamps | |
| `NO_CACHE_HIT` | no token counts, or the producer reports no cache usage | |
| `EMPTY_REPLY` | there are no generations | the adapter fails to extract reply text, and it then fires on **every** turn |
| `GATE_FILTERED` | `quiet_kinds` is unset (the default) | |

Note the coupling: a broken reply mapping silences `UNVERIFIED_CLAIM` *and* makes
`EMPTY_REPLY` fire on everything. One wrong field, two wrong columns, in opposite
directions.

So check rather than assume:

```python
from postflight import coverage

for row in coverage(turns, cfg):
    print(row)  # e.g. "NO_CACHE_HIT: INERT - no generation reports cache usage"
```

It reports structural inertness only, meaning an input absent from every turn. It will not tell
you a detector is broken because its count is zero, because a tool that never errored is
a healthy agent, and conflating those would just move the problem up a level.

### Claim rules

The one worth real attention is `claim_rules`, which drives `UNVERIFIED_CLAIM`. A rule
pairs a regex against the tools that would make the claim true:

```python
from postflight import ClaimRule, Config
import re

Config(
    claim_rules=(
        ClaimRule(
            name="ticket_filed",
            pattern=re.compile(
                r"\b(?:filed|opened|created)\b[^.\n]{0,40}\bticket\b", re.IGNORECASE
            ),
            satisfied_by=frozenset({"create_ticket", "escalate_to_support"}),
        ),
    )
)
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
`"unknown"` whenever the adapter cannot resolve it: a root span that failed to open,
sampling that dropped it. An allow-list of "surfaces we check" silently exempts those
real turns, and every future surface until someone remembers to edit the set. A new
narrating surface going unflagged is a false positive; a new conversational surface
going unflagged is a missed lie.

**Report on faults, not on findings.** `GATE_FILTERED` is `Severity.INFO` because it
fires on correct behaviour. Counting it as a fault makes the headline cry wolf, and a
detector that cries wolf on the healthy case is how the real rows get ignored. Use
`faults()` for anything a human reads first.
