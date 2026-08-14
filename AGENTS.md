# AGENTS.md

Instructions for coding agents working in this repository.
Human contributors want [CONTRIBUTING.md](CONTRIBUTING.md); this file only covers the
constraints that are invisible from the code and expensive to violate.

## Checks

```bash
pip install pytest && python -m pytest tests/ -q
ruff check . && ruff format --check .
```

Lint runs over the whole repository, markdown included, because ruff formats fenced
Python inside it. Scoping a local run to `postflight/` and `tests/` is how local and CI
end up disagreeing.

## Constraints

**No dependencies. Ever.** `dependencies = []` is the reason this package can be added to
anything without a version negotiation, and a test enforces it. The HTTP clients use
`urllib` rather than `httpx` on purpose. An adapter parses exported JSON rather than
importing a vendor SDK: the producer needs the SDK, the reader does not.

**Detector codes are the public API.** `UNVERIFIED_CLAIM`, `TOOL_REFUSAL` and the rest
are what callers filter, chart and page on. Renaming one is a breaking change. Adding a
detector is not.

**Unknown is not zero.** `cache_read_tokens=None` means the producer does not report
cache usage; `0` means it reported none. Detectors must skip the first rather than treat
absence as evidence. The same shape recurs elsewhere: never infer a problem from a
missing signal.

**Kind-keyed rules are deny-lists, never allow-lists.** `Turn.kind` falls back to
`"unknown"` when an adapter cannot resolve it, so an allow-list silently exempts real
turns and every future surface. Failing closed is the point.

**Detectors are mechanism; vocabulary is configuration.** Tool names, claim phrasings,
thresholds and surface names belong in `Config`, supplied by the caller. If a change
teaches the package something about a particular product or domain, it is in the wrong
place.

**A detector that fires on healthy behaviour is worse than one that misses.** Every
narrowing in `detectors.py` exists because something cried wolf. Before widening a rule,
add the case it must NOT fire on. Both directions get a test.

## Writing

Comments explain **why the code is the way it is**, not what it does and not the story of
how the problem was found. Incident narratives, measured percentages and production
counts do not belong here.

Public copy (README, `docs/`, `CONTRIBUTING`, `SECURITY`, and anything the CLI prints)
uses no em-dashes. Rewrite the sentence rather than swapping in a comma, which produces
splices.

Release notes live on the Releases page and are generated from merged PR titles. There is
no changelog file; do not add one.

## Do not touch

`docs/img/*.svg` are generated, with text converted to outlines. They are path data, not
editable markup, and the generator is not in this repository. Leave them alone.
