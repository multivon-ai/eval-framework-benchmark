"""DeepEval FaithfulnessMetric adapter.

DeepEval scores in [0, 1]; ``passed`` is true when score >= threshold.
Default threshold is 0.5. We hold the same judge as the other frameworks
for parity.

Judge plumbing: gpt-* ids pass through as strings (DeepEval's default
GPTModel/OpenAI path). claude-* ids are wrapped in DeepEval's own
``deepeval.models.AnthropicModel`` (native Anthropic SDK) — passing the
bare string routes it to the OpenAI endpoint, which 404s (observed in the
pilot's claude-haiku-4-5 column). Both paths are DeepEval's shipped model
classes; no judge-prompt or parser configuration is touched.
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import STATIC_SUMMARIZATION_INPUT, FrameworkResult, FrameworkRunner


class DeepEvalFaithfulness(FrameworkRunner):
    name = "deepeval"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported so the harness doesn't blow up if deepeval is missing.
        from deepeval.metrics import FaithfulnessMetric  # noqa
        self._FaithfulnessMetric: Any = FaithfulnessMetric
        from deepeval.test_case import LLMTestCase  # noqa
        self._LLMTestCase: Any = LLMTestCase
        if judge_model.lower().startswith(("claude-", "anthropic/")):
            from deepeval.models import AnthropicModel  # noqa
            self._judge_model: Any = AnthropicModel(
                model=judge_model.removeprefix("anthropic/"), temperature=0)
        else:
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
                input=case.question or STATIC_SUMMARIZATION_INPUT,
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
