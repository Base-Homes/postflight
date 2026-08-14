# postflight

[![tests](https://github.com/Base-Homes/postflight/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Base-Homes/postflight/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/postflight)](https://pypi.org/project/postflight/)
[![Python versions](https://img.shields.io/pypi/pyversions/postflight)](https://pypi.org/project/postflight/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Turn-level failure detection for tool-calling agents.**

Your evals score what the model *said*. These are the failures that happen in the gaps
between what it said and what it did: a tool that declined three steps before the reply
contradicting it, the same read issued eight times, a cache that never warmed. They are
invisible to an evaluator scoped to a single observation, which is what every tracing
platform's evaluator runtime gives you.

## The taxonomy

| Code | What it means | Why it matters |
|---|---|---|
| `UNVERIFIED_CLAIM` | The reply asserts a write that no successful tool backs up. | The only one a user experiences directly as a lie. They were told something happened that did not happen. |
| `TOOL_ERROR` | A tool raised; the framework wrapped it. | The visible half of tool failure. Usually already in your dashboards. |
| `TOOL_REFUSAL` [^1] | A tool ran fine and **declined in its own result body**, with no error flag. | The dangerous half. Every guard that asks "did the tool run" is satisfied, so a false confirmation sails through. |
| `REPEATED_TOOL` | The same tool called 3+ times in one turn. | The model is searching for an argument it was never given. A context gap, not a model failure. |
| `TOOL_STORM` | 8+ tool calls in one turn. | Same cause, worse. Cost and latency both. |
| `EMPTY_REPLY` | The turn produced no text where somebody was owed one. | On a 1:1 channel this is the "it just didn't respond" bug. |
| `GATE_FILTERED` | A turn a relevance gate dropped without doing work. | **Information, not a fault.** Silence is the design. Watch the count for a gate that has started swallowing real traffic. |
| `SLOW_TURN` | Wall clock over the threshold. | Usually a storm with a human waiting. |
| `NO_CACHE_HIT` | A prompt big enough to cache that read nothing from cache. | Caching is a prefix match, so one volatile byte early in the system prompt drops the discount on *every* turn. |

The codes are the stable interface. Filter on them, chart them, page on them.

[^1]: Detects an in-body decline in a shape you have told it about. The default is a
success flag set to `false`. If your tools say no some other way, see
[configuring](docs/configuring.md#what-tool_refusal-can-and-cannot-see).

## Install

```bash
pip install postflight
```

No dependencies. Python 3.11+.

## Use

From the command line, which is what a cron job or a CI step wants. Exit status is 1
when something faulted, 0 otherwise:

```bash
python -m postflight --langfuse --hours 24
python -m postflight --otel spans.jsonl
```

Or from Python:

```python
from postflight import Config, faults, run_all
from postflight.adapters.langfuse import LangfuseAdapter, LangfuseClient

client = LangfuseClient(host, public_key, secret_key)
turns = LangfuseAdapter().turns(client.observations(hours=24))

for turn_id, findings in run_all(turns, Config()).items():
    for finding in faults(findings):
        print(turn_id, finding.code, finding.message)
```

## What it prints

Against the trace in `tests/fixtures/`, a throwaway support agent whose notification
tool declined (an invented scenario, captured for real through OpenInference):

```
$ python -m postflight --otel tests/fixtures/openinference_support_turn.jsonl
  0x4069cd953e   TOOL_REFUSAL       1 tool call(s) declined in-body

1 turns, 1 flagged
  TOOL_REFUSAL       1

Not all detectors are live on this data:
  NO_CACHE_HIT: INERT - no generation reports cache usage
```

Three things to read there.

The finding's `detail` is what you act on: `{"tool": "send_notification", "result":
"{'sent': False, 'reason': 'channel unavailable'}"}`.

Nothing fired for `UNVERIFIED_CLAIM`, correctly. The agent's reply said *"I wasn't able
to send the notification"*, an honest report rather than a claim. Had it said "I've let
the customer know", that same turn produces the finding you want to be paged about.

The last block matters as much as the findings. A detector whose input is missing does
not error, it just never fires, and an empty column reads exactly like a clean agent.
`coverage()` reports which ones could not have fired, so a zero can be trusted.

Expect the counts to be lopsided, and expect that to be the useful part. `SLOW_TURN` and
`GATE_FILTERED` dominate any real window; one is already visible to whoever waited, and
the other is a surface behaving correctly. The rare rows carry the weight. Sort by
severity, not by count.

## The `Turn` contract

Detectors never see your trace format. They read `Turn`, so an adapter is a function
from whatever you have to an ordered sequence of steps:

```python
from postflight.model import Generation, ToolCall, Turn

Turn(
    id="…",
    # the surface: which agent, which channel
    kind="chat.turn",
    # ordered, because the sequence is the signal
    steps=(
        Generation(text="", input_tokens=900, model="…"),
        ToolCall(name="search", result={"count": 0}),
        Generation(text="I couldn't find it.", input_tokens=1200, model="…"),
    ),
)
```

A turn is a sequence, not a tree. That is what lets a detector see a tool that declined
several steps before the reply contradicting it. Flatten yours.

Adapters ship for Langfuse and for OpenTelemetry / OpenInference.
`postflight/adapters/otel.py` is the one to copy: it handles flattened attributes, a span
tree and two timestamp encodings, so the awkward cases are worked out there. Read
[writing an adapter](docs/configuring.md#writing-an-adapter) first. Its four notes are
each a mistake already made once.

## Configuring

Thresholds, claim vocabulary, which surfaces owe a reply, and what each detector needs
in order to fire at all: **[docs/configuring.md](docs/configuring.md)**.

## Status

Alpha. The taxonomy is the product: detector codes are the public interface, and
renaming one is a breaking change. Thresholds, default vocabularies and added detectors
are not.

Versioning follows SemVer, with the 0.x convention the spec leaves undefined made
explicit: while the major is 0, a **minor** bump may break the API and a **patch** may
not. Release notes are on the
[Releases page](https://github.com/Base-Homes/postflight/releases).

Issues and PRs welcome. No response SLA.

Apache 2.0. See [LICENSE](LICENSE).
