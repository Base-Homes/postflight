"""The CLI is how this gets wired into a cron job or a CI step, so its exit status is
part of the contract: 0 when nothing faulted, 1 when something did."""

import pathlib

import pytest

from postflight.__main__ import main

FIXTURE = str(
    pathlib.Path(__file__).parent / "fixtures" / "openinference_support_turn.jsonl"
)


def test_exits_1_when_a_fault_is_found(capsys):
    assert main(["--otel", FIXTURE]) == 1
    assert "TOOL_REFUSAL" in capsys.readouterr().out


def test_exits_0_when_nothing_faults(tmp_path, capsys):
    empty = tmp_path / "none.jsonl"
    empty.write_text(
        '{"name": "a.turn", "context": {"trace_id": "t"}, "parent_id": null,'
        ' "attributes": {"openinference.span.kind": "AGENT"}}\n'
    )
    assert main(["--otel", str(empty)]) == 0


def test_inert_detectors_are_named_in_the_output(capsys):
    """A zero next to a detector that could not have fired is the failure this whole
    package is trying not to have, so the CLI says so unprompted."""
    main(["--otel", FIXTURE])
    assert "Not all detectors are live" in capsys.readouterr().out


def test_coverage_mode_exits_0_and_reports_every_detector(capsys):
    assert main(["--otel", FIXTURE, "--coverage"]) == 0
    out = capsys.readouterr().out
    for code in ("UNVERIFIED_CLAIM", "TOOL_REFUSAL", "SLOW_TURN", "NO_CACHE_HIT"):
        assert code in out


def test_json_output_is_parseable_and_keeps_the_exit_code(capsys):
    import json

    assert main(["--otel", FIXTURE, "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["turns"] == 1 and payload["flagged"] == 1
    assert payload["findings"][0]["code"] == "TOOL_REFUSAL"


def test_langfuse_without_credentials_fails_loudly(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit, match="LANGFUSE_PUBLIC_KEY"):
        main(["--langfuse"])


def test_a_source_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_a_missing_file_is_a_message_not_a_traceback():
    """A typo'd path is an ordinary mistake. Answering it with a stack trace tells the
    user the tool crashed, which is both unhelpful and untrue."""
    with pytest.raises(SystemExit, match="cannot read"):
        main(["--otel", "/nonexistent/spans.jsonl"])


def test_a_file_that_is_not_jsonl_says_so(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("these are my notes, not spans\n")
    with pytest.raises(SystemExit, match="not newline-delimited JSON"):
        main(["--otel", str(bad)])
