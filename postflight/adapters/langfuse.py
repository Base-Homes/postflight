"""Langfuse observations → `Turn`.

Two things this endpoint does NOT let you do, both learned the hard way and both the
reason the grouping happens locally instead of server-side:

  - `traceName` is set on the ROOT observation only, so a server-side filter on it
    returns roots WITHOUT their children — a turn with no generations and no tool
    spans, which every detector reads as empty. (It is also why Langfuse's own metrics
    API reports $0 for a per-surface cost grouping: the name is on the root, the cost
    is on the children.) Kind filtering therefore stays client-side, and it is free.
  - `parseIoAsJson` is rejected with a 400, so `input`/`output` come back as JSON
    STRINGS. A reader that assumes parsed objects silently sees JSON scaffolding as
    the reply text, and every claim regex then matches punctuation instead of prose.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from ..config import UNKNOWN_KIND
from ..model import Generation, Step, ToolCall, Turn

# 1000 is the endpoint's max page size. Asking for less only costs more requests, and
# requests — not bytes — are what the rate limiter counts. `fields` is a projection:
# every group named here feeds a detector, and dropping one to save bytes silently
# blanks whatever reads it.
_PAGE = 1000
_FIELDS = "core,basic,time,io,usage,model,metadata,trace_context"


@dataclass(frozen=True)
class LangfuseAdapter:
    """How this deployment names its spans.

    Defaults match the common convention (`tool.<name>` children under a `<kind>.turn`
    root). If yours differs, this is the only place that needs to know.
    """

    tool_span_prefix: str = "tool."
    root_span_suffix: str = ".turn"

    def turn(self, trace_id: str, observations: list[dict[str, Any]]) -> Turn:
        rows = sorted(observations, key=lambda o: o.get("startTime") or "")
        steps: list[Step] = []
        for row in rows:
            if row.get("type") == "GENERATION":
                steps.append(self._generation(row))
            elif str(row.get("name") or "").startswith(self.tool_span_prefix):
                steps.append(self._tool_call(row))

        root = next((o for o in rows if o.get("isRootObservation")), None)
        kind = next((str(o.get("name")) for o in rows
                     if str(o.get("name") or "").endswith(self.root_span_suffix)),
                    None)
        starts = [_ts(o.get("startTime")) for o in rows if o.get("startTime")]
        ends = [_ts(o.get("endTime") or o.get("startTime")) for o in rows
                if o.get("startTime")]
        return Turn(
            id=trace_id,
            kind=kind or UNKNOWN_KIND,
            steps=tuple(steps),
            # userId/sessionId ride every observation but only where the caller set
            # them — take the first non-empty rather than the root's, since a root span
            # opened before the user is resolved carries "".
            user_id=next((o.get("userId") for o in rows if o.get("userId")), None),
            session_id=next((o.get("sessionId") for o in rows if o.get("sessionId")), None),
            started_at=min([s for s in starts if s], default=None),
            ended_at=max([e for e in ends if e], default=None),
            metadata={"observations": len(rows), "root": (root or {}).get("name")},
            raw=rows,
        )

    def turns(self, observations: Iterable[dict[str, Any]]) -> list[Turn]:
        by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            trace_id = row.get("traceId")
            if trace_id:
                by_trace[trace_id].append(row)
        built = [self.turn(tid, rows) for tid, rows in by_trace.items()]
        built.sort(key=lambda t: t.started_at or datetime.min.replace(tzinfo=timezone.utc),
                   reverse=True)
        return built

    def _generation(self, row: dict[str, Any]) -> Generation:
        # `usage` exists on the per-trace endpoint only; the observations endpoint
        # carries `usageDetails` alone. Read both, preferring whichever is populated —
        # keyed on `usage` alone, every turn reports 0 tokens and the cache detector
        # never fires, because a zero prompt is never "big".
        usage = row.get("usage") or {}
        details = row.get("usageDetails") or {}
        pick = lambda *keys: next(
            (int(src.get(key) or 0) for src in (usage, details) for key in keys
             if src.get(key)), 0)
        return Generation(
            model=row.get("model"),
            text=text_of(row.get("output")),
            input_tokens=pick("input", "input_tokens"),
            output_tokens=pick("output", "output_tokens"),
            # Present-but-zero and absent are DIFFERENT: measured over 906 production
            # generations, 833 carried the key (798 of them zero) and 73 omitted it.
            # Mapping the missing 73 to zero told the cache detector they missed a cache
            # they were never observed to have.
            cache_read_tokens=_opt_int(details, "cache_read_input_tokens"),
            cache_write_tokens=_opt_int(details, "cache_creation_input_tokens"),
            started_at=_ts(row.get("startTime")),
            ended_at=_ts(row.get("endTime")),
        )

    def _tool_call(self, row: dict[str, Any]) -> ToolCall:
        result = parsed(row.get("output"))
        arguments = parsed(row.get("input"))
        # `metadata` is arbitrary JSON — a string and a list are both legal and both
        # occur. Reading `.get` off one raised AttributeError inside the eager
        # `turns()` build, so a single oddly-shaped span took down a whole day's pull
        # rather than one turn.
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        return ToolCall(
            name=str(row.get("name"))[len(self.tool_span_prefix):],
            arguments=arguments if isinstance(arguments, dict) else None,
            # Keep the raw string when it did not parse — the error-prefix check reads it.
            result=result if result is not None else row.get("output"),
            is_error=bool(metadata.get("isError") or row.get("level") == "ERROR"),
            started_at=_ts(row.get("startTime")),
            ended_at=_ts(row.get("endTime")),
        )


def _opt_int(details: dict[str, Any], key: str) -> int | None:
    """The reported number, or None when this producer does not report it at all."""
    if key not in details or details[key] is None:
        return None
    try:
        return int(details[key])
    except (TypeError, ValueError):
        return None


def parsed(value: Any) -> Any:
    """A span's payload is an object, a JSON string, or neither."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return None


def text_of(output: Any) -> str:
    """Anthropic returns content blocks; a fallback provider returns a bare string.

    A third shape arrives from the observations endpoint, which serialises `output` and
    hands back the JSON as a STRING. Parse first, then fall through to the plain-string
    case for a bare reply or anything that parses as a scalar.
    """
    if isinstance(output, str):
        inner = parsed(output)
        return text_of(inner) if isinstance(inner, (list, dict)) else output
    if isinstance(output, list):
        return " ".join(block.get("text", "") for block in output
                        if isinstance(block, dict) and block.get("type") == "text")
    if isinstance(output, dict):
        return output.get("text") or ""
    return ""


def _ts(value: str | None) -> datetime | None:
    """Always tz-aware, or None.

    A timestamp that arrives without an offset parses NAIVE, and one naive value among
    aware ones raises `TypeError: can't compare offset-naive and offset-aware` the
    moment anything sorts or takes a min/max over them — which `turns()` and
    `Turn.duration_s` both do, so a single such row took down the whole batch. Assume
    UTC for a bare timestamp: this endpoint reports in UTC, and a wrong-by-a-timezone
    ordering is a far better failure than a crash.
    """
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


class LangfuseClient:
    """Minimal read client. Stdlib only, on purpose — this package has no dependencies.

    Backs off on 429: Langfuse answers with a per-MINUTE `Retry-After`, and a naive
    loop that gives up partway reports on a BIASED subset (the pages it happened to
    get), which is worse than a slow run.
    """

    def __init__(self, host: str, public_key: str, secret_key: str, pause: float = 0.0):
        self.host = host.rstrip("/")
        self._auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.pause = pause

    def get(self, path: str, retries: int = 5) -> Any:
        delay = 2.0
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                self.host + path, headers={"Authorization": "Basic " + self._auth})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if self.pause:
                        time.sleep(self.pause)
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                    raise
                time.sleep(float(exc.headers.get("Retry-After") or 0) or delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def observations(self, hours: int = 24, environment: str | None = None,
                     on_page: Any = None) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows: list[dict[str, Any]] = []
        cursor, pages = None, 0
        while True:
            params: dict[str, Any] = {"fromStartTime": since, "limit": _PAGE,
                                      "fields": _FIELDS}
            if environment:
                params["environment"] = environment
            if cursor:
                params["cursor"] = cursor
            data = self.get("/api/public/v2/observations?"
                            + urllib.parse.urlencode(params))
            rows.extend(data.get("data") or [])
            pages += 1
            if on_page:
                on_page(pages, len(rows))
            cursor = (data.get("meta") or {}).get("cursor")
            if not cursor:
                return rows
