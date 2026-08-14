"""OpenTelemetry spans (OpenInference conventions) → `Turn`.

Reads spans as exported JSON — `ReadableSpan.to_json()`, an OTLP-JSON dump, or anything
that hands back the same dicts — so postflight still needs no OpenTelemetry SDK and keeps
its zero-dependency promise. The producer needs the SDK; the reader does not.

Three differences from the Langfuse adapter, all of them the reason this module exists
rather than a flag on that one:

  - **The turn is a SPAN TREE, not a flat list keyed by trace id.** Grouping is by
    `context.trace_id`, and the surface name comes from the root — the span with no
    `parent_id`, or the one marked `openinference.span.kind == "AGENT"`.
  - **Attributes are FLATTENED with indices.** A reply is not a field; it is spread over
    `llm.output_messages.<i>.message.contents.<j>.message_content.text`, and
    reconstructing it means sorting on those indices. Read the last output message, since
    earlier ones are the model talking to itself.
  - **A tool result is an opaque string** in `output.value`, not structured JSON, so the
    refusal check parses it here and hands `tool_outcome` the same shape it gets from
    every other adapter.

Cache tokens map to **None, not 0**: OpenInference does not emit a cache attribute at
all. The absence is real rather than a mapping gap, and the distinction is load-bearing
— scored as zero, any turn whose prompt exceeds the model's cacheable floor reads as a
cache miss.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from ..config import UNKNOWN_KIND
from ..model import Generation, Step, ToolCall, Turn

_KIND = "openinference.span.kind"
# `llm.output_messages.0.message.contents.1.message_content.text` → (0, 1)
_TEXT_RE = re.compile(
    r"^llm\.output_messages\.(\d+)\.message\.contents\.(\d+)\.message_content\.text$")
# The unflattened variant some instrumentations emit instead.
_CONTENT_RE = re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$")


def _int(attrs: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _opt_int(attrs: dict[str, Any], *keys: str) -> int | None:
    """The reported number, or None when this producer does not report it at all.

    Distinct from `_int`, which floors a missing key to 0. For cache counters that
    difference IS the signal: OpenInference emits no cache attribute at all, and calling
    that zero asserts a miss the trace gives no evidence for.
    """
    for key in keys:
        if attrs.get(key) not in (None, ""):
            try:
                return int(attrs[key])
            except (TypeError, ValueError):
                continue
    return None


def _ts(value: Any) -> datetime | None:
    """Spans carry ISO strings from `to_json()` and integer nanos over OTLP."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def reply_text(attrs: dict[str, Any]) -> str:
    """The LAST output message's text, reassembled from flattened attributes.

    Indices are sorted NUMERICALLY. Sorting the attribute keys as strings puts
    `contents.10` before `contents.2`, which silently scrambles any reply long enough to
    span ten content blocks — and reads as the model producing word salad.
    """
    parts: dict[int, dict[int, str]] = defaultdict(dict)
    for key, value in attrs.items():
        match = _TEXT_RE.match(key)
        if match:
            parts[int(match.group(1))][int(match.group(2))] = str(value)
            continue
        match = _CONTENT_RE.match(key)
        if match:
            parts[int(match.group(1))].setdefault(-1, str(value))
    if not parts:
        return ""
    last = parts[max(parts)]
    return "".join(last[i] for i in sorted(last)).strip()


def _parse(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return None


def _generation(span: dict[str, Any]) -> Generation:
    attrs = span.get("attributes") or {}
    return Generation(
        model=attrs.get("llm.model_name"),
        text=reply_text(attrs),
        input_tokens=_int(attrs, "llm.token_count.prompt"),
        output_tokens=_int(attrs, "llm.token_count.completion"),
        # OpenInference's prompt_details convention, where the instrumentation reports
        # it at all. None when it does not — see the module docstring.
        cache_read_tokens=_opt_int(attrs, "llm.token_count.prompt_details.cache_read"),
        cache_write_tokens=_opt_int(attrs, "llm.token_count.prompt_details.cache_write"),
        started_at=_ts(span.get("start_time")),
        ended_at=_ts(span.get("end_time")),
    )


def _tool_call(span: dict[str, Any]) -> ToolCall:
    attrs = span.get("attributes") or {}
    raw_out = attrs.get("output.value")
    parsed = _parse(raw_out)
    arguments = _parse(attrs.get("input.value"))
    status = (span.get("status") or {}).get("status_code")
    return ToolCall(
        # `tool.name` where the instrumentation sets it, else the span name with the
        # conventional `tool.` prefix stripped.
        name=str(attrs.get("tool.name") or span.get("name") or "").removeprefix("tool."),
        arguments=arguments if isinstance(arguments, dict) else None,
        result=parsed if parsed is not None else raw_out,
        is_error=status == "ERROR",
        started_at=_ts(span.get("start_time")),
        ended_at=_ts(span.get("end_time")),
    )


def _trace_id(span: dict[str, Any]) -> str | None:
    context = span.get("context") or {}
    return context.get("trace_id") or span.get("traceId") or span.get("trace_id")


def turn(trace_id: str, spans: list[dict[str, Any]]) -> Turn:
    ordered = sorted(spans, key=lambda s: str(s.get("start_time") or ""))
    steps: list[Step] = []
    root: dict[str, Any] | None = None
    for span in ordered:
        kind = (span.get("attributes") or {}).get(_KIND)
        if kind == "LLM":
            steps.append(_generation(span))
        elif kind == "TOOL":
            steps.append(_tool_call(span))
        elif kind == "AGENT" or not span.get("parent_id"):
            # First one wins: a nested AGENT span is a sub-agent, and the OUTERMOST span
            # is the one whose name describes the surface.
            root = root or span
    starts = [_ts(s.get("start_time")) for s in ordered]
    ends = [_ts(s.get("end_time")) for s in ordered]
    return Turn(
        id=str(trace_id),
        kind=str((root or {}).get("name") or UNKNOWN_KIND),
        steps=tuple(steps),
        started_at=min([s for s in starts if s], default=None),
        ended_at=max([e for e in ends if e], default=None),
        metadata={"spans": len(ordered), "root": (root or {}).get("name")},
        raw=ordered,
    )


def turns(spans: Iterable[dict[str, Any]]) -> list[Turn]:
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        trace_id = _trace_id(span)
        if trace_id:
            by_trace[str(trace_id)].append(span)
    built = [turn(tid, rows) for tid, rows in by_trace.items()]
    built.sort(key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc),
               reverse=True)
    return built


def turns_from_jsonl(path: str) -> list[Turn]:
    """One exported span per line — what `ReadableSpan.to_json(indent=None)` writes."""
    with open(path, encoding="utf-8") as handle:
        return turns(json.loads(line) for line in handle if line.strip())
