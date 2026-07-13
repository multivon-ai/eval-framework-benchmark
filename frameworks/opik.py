"""Opik Hallucination adapter.

Metric: ``opik.evaluation.metrics.Hallucination`` (Comet Opik's canonical
LLM-judged hallucination metric for input+output+context). Native fields:
``input`` (question), ``output`` (response), ``context`` (list of context
strings) — a strictly 1:1 mapping to our Case schema. ``input`` is a
required argument, so for summarization cases (empty ``Case.question``)
we supply the identical static string "Summarize the text." per the
preregistered Unavoidable Configuration Rule.

Score direction and threshold provenance (investigated 2026-07-13,
opik 2.1.22):

- Opik's Hallucination returns a score in [0, 1] where HIGHER = MORE
  hallucinated — the inverse of ``FrameworkResult.score`` (higher = more
  faithful). We store ``score = 1.0 - hallucination_score`` and keep the
  native value in ``raw``.
- The documented convention is binary: "The hallucination score is
  either 0 or 1. A score of 0 indicates that no hallucinations were
  detected, a score of 1 indicates that hallucinations were detected."
  (https://www.comet.com/docs/opik/evaluation/metrics/hallucination)
  The shipped judge prompt nevertheless permits fractional scores
  ("assign a hallucination score between 0 and 1"), and the parser
  accepts any float in [0, 1]; Opik ships no numeric cutoff of its own.
- We therefore binarize at the midpoint of the documented 0/1 convention:
  flagged_hallucinated iff native hallucination score >= 0.5
  (equivalently, faithfulness score < 0.5 with ties flagged).
  ``FrameworkResult.threshold`` records 0.5 on the faithfulness scale.

Judge plumbing: Opik judges route through LiteLLM
(``LiteLLMChatModel``) by passing a model-name string; gpt-* ids work
as-is, claude-* ids get an ``anthropic/`` prefix. Temperature 0 is passed
via the metric's own ``temperature`` constructor argument. ``track=False``
keeps the metric from attempting to log to an Opik/Comet backend.
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import FrameworkResult, FrameworkRunner

# Unavoidable Configuration Rule (preregistered): identical static string
# supplied to any framework that requires a question field on tasks where
# the dataset has none (summarization).
STATIC_SUMMARIZATION_INPUT = "Summarize the text."


class OpikHallucination(FrameworkRunner):
    name = "opik"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported so the harness doesn't blow up if opik is missing.
        from opik.evaluation.metrics import Hallucination  # noqa
        m = judge_model.lower()
        litellm_id = judge_model
        if m.startswith("claude-") and "/" not in judge_model:
            litellm_id = f"anthropic/{judge_model}"
        self._metric: Any = Hallucination(
            model=litellm_id,
            temperature=0.0,
            track=False,
        )
        self._judge_model = judge_model
        self._threshold = threshold

    def run(self, case: Case) -> FrameworkResult:
        t0 = time.perf_counter()
        try:
            result = self._metric.score(
                input=case.question or STATIC_SUMMARIZATION_INPUT,
                output=case.answer,
                context=[case.context],
            )
            halluc = float(result.value)
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
            # Invert to the shared direction: higher = more faithful.
            score=1.0 - halluc,
            # Native convention: >= 0.5 hallucination score = detected.
            flagged_hallucinated=halluc >= self._threshold,
            threshold=self._threshold,
            latency_ms=latency_ms,
            raw={"hallucination_score": halluc,
                 "reason": str(getattr(result, "reason", None))},
        )
