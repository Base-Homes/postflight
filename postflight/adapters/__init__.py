"""Source-specific readers that build the normalized `Turn`.

An adapter is the only place in the package allowed to know a vendor's field names.
Everything downstream reads `postflight.model`, which is what lets the same detector
run against Langfuse today and an OpenTelemetry span tree tomorrow.
"""
