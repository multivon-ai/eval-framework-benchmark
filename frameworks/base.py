"""
Common contract for every framework adapter.

Each adapter exposes a single ``run(case) -> FrameworkResult`` method. The
benchmark orchestrator calls it once per (case, run_index) pair so we can
compute run-to-run variance.

Frameworks differ in what they expose: some emit raw 0–1 scores, some
emit pass/fail, some expose tokens, some don't. We capture what we can
and leave optional fields as ``None`` so downstream analysis can be
honest about missing data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any

from data.loader import Case


@dataclass
class FrameworkResult:
    framework: str        # "multivon-eval" | "deepeval" | "ragas"
    case_id: str
    score: float          # 0..1, higher = more faithful
    flagged_hallucinated: bool  # framework's own pass/fail at default threshold
    threshold: float      # the threshold the framework used for the verdict
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: Any = None       # framework-specific raw output for debugging
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't serialize raw objects that may not be json-clean.
        if self.raw is not None and not isinstance(self.raw, (str, int, float, bool, list, dict)):
            d["raw"] = repr(self.raw)
        return d


class FrameworkRunner(ABC):
    """Synchronous contract; concurrency is handled by the orchestrator."""

    name: str = ""

    @abstractmethod
    def run(self, case: Case) -> FrameworkResult:
        ...
