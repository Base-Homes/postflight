"""Command line entry point, so a cron job or CI step does not need a wrapper script.

    python -m postflight --otel spans.jsonl
    python -m postflight --langfuse --hours 24
    python -m postflight --otel spans.jsonl --coverage

Exit status is the useful part in CI: 0 when nothing faulted, 1 when something did.
INFO findings never affect it, because a surface that is silent by design should not
fail anyone's build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from .config import Config
from .coverage import coverage
from .detectors import faults, run
from .model import Turn


def _kinds(value: str | None) -> frozenset[str]:
    return frozenset(k.strip() for k in (value or "").split(",") if k.strip())


def _build_config(args: argparse.Namespace) -> Config:
    return Config(
        slow_turn_s=args.slow_turn_s,
        tool_storm=args.tool_storm,
        repeated_tool=args.repeated_tool,
        conversational_kinds=_kinds(args.conversational_kinds),
        quiet_kinds=_kinds(args.quiet_kinds),
        narrating_kinds=_kinds(args.narrating_kinds),
    )


def _load(args: argparse.Namespace) -> list[Turn]:
    if args.otel:
        from .adapters.otel import turns_from_jsonl

        # A typo'd path and a file that is not JSONL are both ordinary mistakes, and a
        # traceback answers neither of them. SystemExit prints the message and sets a
        # non-zero status without pretending the tool crashed.
        try:
            return turns_from_jsonl(args.otel)
        except OSError as exc:
            raise SystemExit(f"cannot read {args.otel}: {exc.strerror}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"{args.otel} is not newline-delimited JSON: {exc} "
                "(expected one exported span object per line)"
            ) from exc

    from .adapters.langfuse import LangfuseAdapter, LangfuseClient

    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public or not secret:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set for --langfuse"
        )
    host = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    client = LangfuseClient(host, public, secret)
    observations = client.observations(hours=args.hours, environment=args.environment)
    return LangfuseAdapter().turns(observations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="postflight",
        description="Turn-level failure detection for tool-calling agents.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--otel",
        metavar="FILE",
        help="exported OpenTelemetry spans, one JSON object per line",
    )
    source.add_argument(
        "--langfuse",
        action="store_true",
        help="pull from Langfuse (credentials from the environment)",
    )

    parser.add_argument(
        "--hours", type=int, default=24, help="window for --langfuse (default: 24)"
    )
    parser.add_argument(
        "--environment", default=None, help="Langfuse environment filter"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="report which detectors can fire on this data, then exit",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="machine-readable output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="summary only, no per-turn lines"
    )

    parser.add_argument("--slow-turn-s", type=float, default=Config().slow_turn_s)
    parser.add_argument("--tool-storm", type=int, default=Config().tool_storm)
    parser.add_argument("--repeated-tool", type=int, default=Config().repeated_tool)
    parser.add_argument(
        "--conversational-kinds", help="comma separated surfaces that owe a reply"
    )
    parser.add_argument(
        "--quiet-kinds", help="comma separated surfaces that are silent by design"
    )
    parser.add_argument(
        "--narrating-kinds",
        help="comma separated surfaces that narrate rather than speak",
    )

    args = parser.parse_args(argv)
    cfg = _build_config(args)
    turns = _load(args)

    if args.coverage:
        rows = coverage(turns, cfg)
        if args.as_json:
            print(json.dumps([r.__dict__ for r in rows], indent=2))
        else:
            for row in rows:
                print(row)
        return 0

    results = {t.id: run(t, cfg) for t in turns}
    all_findings = [f for fs in results.values() for f in fs]
    flagged = {tid: fs for tid, fs in results.items() if faults(fs)}

    if args.as_json:
        print(
            json.dumps(
                {
                    "turns": len(turns),
                    "flagged": len(flagged),
                    "findings": [
                        {
                            "turn_id": f.turn_id,
                            "code": f.code,
                            "severity": f.severity.value,
                            "message": f.message,
                            "detail": f.detail,
                        }
                        for f in all_findings
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 1 if flagged else 0

    if not args.quiet:
        for turn_id, found in results.items():
            for finding in found:
                mark = " " if finding.severity.value == "fault" else "i"
                print(f"{mark} {turn_id[:12]:14} {finding.code:18} {finding.message}")

    counts = Counter(f.code for f in all_findings)
    print(f"\n{len(turns)} turns, {len(flagged)} flagged")
    for code, count in counts.most_common():
        print(f"  {code:18} {count}")

    # An inert detector and a clean agent look identical in the output above, so say
    # which ones could not have fired rather than leaving a zero to be misread.
    inert = [r for r in coverage(turns, cfg) if not r.live or r.misleading]
    if inert:
        print("\nNot all detectors are live on this data:")
        for row in inert:
            print(f"  {row}")

    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
