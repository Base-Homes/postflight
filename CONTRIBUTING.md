# Contributing

## Running it

```bash
git clone https://github.com/Base-Homes/postflight
cd postflight
pip install pytest        # the only thing the tests need
python -m pytest tests/ -q
```

There is no install step and no dependency file to sync. If `pip install pytest` is not
enough to run the suite, that is a bug in this project, not in your setup — the package
is zero-dependency and CI proves it by installing nothing else.

## What a good change looks like

**A new detector** needs a name that describes the *behaviour*, not a metric, and a test
for both directions — the case it must catch and the case it must not. The second is the
one that matters: a detector that fires on healthy traffic is how the real findings get
ignored, and every detector here was narrowed at least once because of that.

**A new adapter** is a function from your trace format to `Turn`. Read "Writing an
adapter" in the README first; the four notes there are each a mistake already made once.
Adapters must not add a dependency — parse exported JSON rather than importing a vendor
SDK.

**Detector codes are the public interface.** Renaming one is a breaking change, so it
lands as a minor bump with a changelog entry, not quietly.

## Tuning vs mechanism

The line this project is organised around: detectors are mechanism and live in code;
thresholds, tool-name conventions and claim vocabulary are tuning and live in `Config`.
If a change makes postflight know something about *your* domain, it probably belongs in
your `Config`, not in here. Say so in the PR if you think it's the exception.

## Before you open a PR

- `python -m pytest tests/ -q` passes
- New behaviour has a test
- A comment explains *why the code is the way it is*, where that is not obvious. Not what
  it does, and not the story of how it was found
- The PR title reads as a release note — it becomes one verbatim, since release notes are
  generated from merged PRs at tag time

Issues and PRs are welcome. There is no response SLA.
