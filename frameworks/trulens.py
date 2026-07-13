"""TruLens Groundedness adapter.

Metric: the canonical ``groundedness_measure_with_cot_reasons`` feedback
(TruLens ≥2.x, ``trulens.feedback.llm_provider.LLMProvider``). It takes
exactly two fields — ``source`` (context) and ``statement`` (response) —
so the field mapping is strictly 1:1 and no question/query field is
needed (the Unavoidable Configuration Rule is not triggered for TruLens).
Returns a score in [0, 1], higher = more grounded/faithful, which matches
``FrameworkResult.score`` direction directly.

Threshold provenance (investigated 2026-07-13, trulens 2.8.1):
TruLens ships NO default binarization threshold for groundedness.

- The feedback/metric API returns only a float; there is no pass/fail
  anywhere in the scoring path.
- The guardrail decorators (``trulens.core.guardrails.base.context_filter``
  / ``block_input`` / ``block_output``) take ``threshold`` as a REQUIRED
  argument with no default.
- The dashboard styling (``trulens/dashboard/ux/styles.py``, ``CATEGORY``)
  uses tri-state display cutoffs (fail <0.6, warning 0.6–0.8, pass ≥0.8)
  for higher-is-better feedbacks — a UI color convention, not a binary
  verdict, so it does not qualify as a shipped binarization default.

Per the preregistered metric-mapping rule we therefore binarize at 0.5,
pass at score >= 0.5. 0.5 is the library-documentation convention: it is
the threshold used in TruLens's own guardrails documentation examples
(``@context_filter(feedback, 0.5, ...)`` and
``WithFeedbackFilterDocuments.of_retriever(..., threshold=0.5)``), see
https://www.trulens.org/component_guides/runtime_evaluation/guardrails/

Judge plumbing: OpenAI-style model ids (gpt-*) go through the
``trulens.providers.openai.OpenAI`` provider; Anthropic model ids
(claude-*) go through ``trulens.providers.litellm.LiteLLM`` with an
``anthropic/`` prefix, since TruLens has no native Anthropic provider.
Temperature 0 is passed explicitly to the feedback call (the parameter
TruLens exposes on ``groundedness_measure_with_cot_reasons``).
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import FrameworkResult, FrameworkRunner


class TruLensGroundedness(FrameworkRunner):
    name = "trulens"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported so the harness doesn't blow up if trulens is missing.
        m = judge_model.lower()
        if m.startswith(("claude-", "anthropic/")):
            from trulens.providers.litellm import LiteLLM  # noqa
            litellm_id = (
                judge_model if "/" in judge_model else f"anthropic/{judge_model}"
            )
            self._provider: Any = LiteLLM(model_engine=litellm_id)
        else:
            from trulens.providers.openai import OpenAI  # noqa
            self._provider = OpenAI(model_engine=judge_model)
        self._judge_model = judge_model
        self._threshold = threshold

    def run(self, case: Case) -> FrameworkResult:
        t0 = time.perf_counter()
        try:
            # Canonical groundedness feedback: source = context,
            # statement = the response under evaluation. No question field
            # exists in this metric, so none is synthesized.
            score, reasons = self._provider.groundedness_measure_with_cot_reasons(
                source=case.context,
                statement=case.answer,
                temperature=0.0,
            )
            score = float(score) if score is not None else 0.0
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
            # Pass at score >= 0.5 (documentation convention, see module
            # docstring); flag as hallucinated below it.
            flagged_hallucinated=score < self._threshold,
            threshold=self._threshold,
            latency_ms=latency_ms,
            raw={"reasons": str(reasons)},
        )
