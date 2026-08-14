# Security policy

## Supported versions

Only the latest released version receives fixes. This is a 0.x library; there are no
maintenance branches.

## Reporting a vulnerability

Report privately via
[GitHub Security Advisories](https://github.com/Base-Homes/postflight/security/advisories/new).
Please do not open a public issue.

Expect an acknowledgement within 7 days. This is a small project with no dedicated
security staffing — that window is what can actually be met, not an aspiration.

## Scope

postflight reads trace data and returns findings. It makes no network calls of its own
except through the optional `LangfuseClient`, has no runtime dependencies, and executes
nothing from the traces it reads. The realistic surface is therefore:

- Parsing untrusted trace payloads (malformed JSON, hostile regex input, deeply nested
  or enormous attribute sets).
- `LangfuseClient` and the credentials a caller hands it.

Findings can quote trace content — including reply text — in `Finding.detail`. If your
traces carry personal data, treat postflight's output as carrying it too.
