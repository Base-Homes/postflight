"""postflight — turn-level failure detection for tool-calling agents.

The failures here happen BETWEEN observations rather than inside one, which is why an
evaluator runtime scoped to a single generation cannot express them. Point postflight
at your traces, get named findings back.

    from postflight import Config, run_all
    from postflight.adapters.langfuse import LangfuseAdapter, LangfuseClient

    client = LangfuseClient(host, public_key, secret_key)
    turns = LangfuseAdapter().turns(client.observations(hours=24))
    findings = run_all(turns, Config())
"""

from .config import UNKNOWN_KIND, ClaimRule, Config
from .coverage import Coverage, coverage
from .detectors import (
    DETECTORS,
    Detector,
    faults,
    run,
    run_all,
    succeeded_tools,
    tool_outcome,
)
from .model import Finding, Generation, Outcome, Severity, Step, ToolCall, Turn

__version__ = "0.1.0"

__all__ = [
    "DETECTORS",
    "UNKNOWN_KIND",
    "ClaimRule",
    "Config",
    "Coverage",
    "Detector",
    "Finding",
    "Generation",
    "Outcome",
    "Severity",
    "Step",
    "ToolCall",
    "Turn",
    "__version__",
    "coverage",
    "faults",
    "run",
    "run_all",
    "succeeded_tools",
    "tool_outcome",
]
