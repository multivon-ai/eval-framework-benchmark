"""DeepEval FaithfulnessMetric adapter.

DeepEval scores in [0, 1]; ``passed`` is true when score >= threshold.
Default threshold is 0.5. We hold the same OpenAI judge as the other two
frameworks for parity.
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import FrameworkResult, FrameworkRunner


class DeepEvalFaithfulness(FrameworkRunner):
    name = "deepeval"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported so the harness doesn't blow up if deepeval is missing.
        from deepeval.metrics import FaithfulnessMetric  # noqa
        self._FaithfulnessMetric: Any = FaithfulnessMetric
        from deepeval.test_case import LLMTestCase  # noqa
        self._LLMTestCase: Any = LLMTestCase
        self._judge_model = judge_model
        self._threshold = threshold

    def run(self, case: Case) -> FrameworkResult:
        t0 = time.perf_counter()
        try:
            metric = self._FaithfulnessMetric(
                threshold=self._threshold,
                model=self._judge_model,
                include_reason=False,
                async_mode=False,
                strict_mode=False,
            )
            test_case = self._LLMTestCase(
                input=case.question or "Provide a faithful summary of the document.",
                actual_output=case.answer,
                retrieval_context=[case.context],
            )
            metric.measure(test_case)
            score = float(metric.score) if metric.score is not None else 0.0
            passed = bool(metric.success)
        except Exception as exc:
            return FrameworkResult(
                framework=self.name,
                case_id=case.id,
                score=0.0,
                flagged_hallucinated=False,
                threshold=self._threshold,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        return FrameworkResult(
            framework=self.name,
            case_id=case.id,
            score=score,
            flagged_hallucinated=not passed,
            threshold=self._threshold,
            latency_ms=latency_ms,
            raw={"reason": getattr(metric, "reason", None)},
        )
