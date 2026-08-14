# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**Versioning.** This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
with the 0.x convention the spec leaves undefined made explicit: while the major version
is 0, a **minor** bump may break the API and a **patch** bump may not. Detector codes
(`UNVERIFIED_CLAIM`, `TOOL_REFUSAL`, …) are the public interface — renaming one is a
breaking change. Thresholds, default vocabularies and added detectors are not.

## [Unreleased]

## [0.1.0] - 2026-08-14

### Added
- Nine turn-level failure detectors for tool-calling agents: `UNVERIFIED_CLAIM`,
  `TOOL_ERROR`, `TOOL_REFUSAL`, `REPEATED_TOOL`, `TOOL_STORM`, `EMPTY_REPLY`,
  `GATE_FILTERED`, `SLOW_TURN`, `NO_CACHE_HIT`.
- `Turn` — an ordered sequence of `Generation` and `ToolCall` steps. Modelling the
  sequence rather than a bag is what lets a detector see a tool that declined several
  steps before the reply contradicting it, which an evaluator scoped to a single
  observation structurally cannot do.
- `Config` — every threshold, tool-name convention and claim vocabulary as a value.
  Detectors are mechanism and ship; tuning belongs to whoever runs them.
- Langfuse adapter, with a stdlib read client.
- OpenTelemetry / OpenInference adapter, written against real spans from a producer
  sharing nothing with the first. `Turn` needed no change to accept it.
- `py.typed` (PEP 561).

### Notes
- Claim matching is negation- and subject-aware: "the follow-up was **not** sent" is an
  honest report, and "**the owner** emailed you" is the agent relaying someone else's
  action. Both are identical to a real claim at the verb-object level.
- `Generation.cache_read_tokens` is `int | None` — **unknown is not zero**. An adapter
  whose producer does not report cache usage says so, and `NO_CACHE_HIT` skips it rather
  than inventing a miss.

[Unreleased]: https://github.com/Base-Homes/postflight/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Base-Homes/postflight/releases/tag/v0.1.0
