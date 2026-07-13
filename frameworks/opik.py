"""Opik Hallucination adapter.

Metric: ``opik.evaluation.metrics.Hallucination`` (Comet Opik's canonical
LLM-judged hallucination metric for input+output+context). Native fields:
``input`` (question), ``output`` (response), ``context`` (list of context
strings) — a strictly 1:1 mapping to our Case schema. ``input`` is a
required argument, so for summarization cases (empty ``Case.question``)
we supply the identical harness-wide static string per the preregistered
Unavoidable Configuration Rule (see below).

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

Judge plumbing (PREREG_ADDENDUM.md resolution): Opik 2.1.22's
``models_factory`` routes claude-*/anthropic-prefixed model names to its
NATIVE Anthropic SDK adapter (``AnthropicChatModel``) whenever the
``anthropic`` package is importable — and ``anthropic==0.116.0`` is
hash-pinned in ``study/requirements.lock``, so the native path is the
deterministic outcome of the lockfile, not a runtime accident. We pin
this explicitly: the constructor asserts that claude judges resolved to
the native Anthropic adapter (guarding against a silent LiteLLM fallback
if the environment changes). gpt-* ids route through ``LiteLLMChatModel``
as before. Temperature 0 is passed via the metric's own ``temperature``
constructor argument. ``track=False`` keeps the metric from attempting to
log to an Opik/Comet backend.

Static summarization input: the identical harness-wide string from
``frameworks.base.STATIC_SUMMARIZATION_INPUT`` (see PREREG_ADDENDUM.md —
the plan's example string "Summarize the text." was superseded by the
harness's existing pilot-era string for cross-framework identity and
pilot comparability).
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import STATIC_SUMMARIZATION_INPUT, FrameworkResult, FrameworkRunner


class OpikHallucination(FrameworkRunner):
    name = "opik"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported so the harness doesn't blow up if opik is missing.
        from opik.evaluation.metrics import Hallucination  # noqa
        self._metric: Any = Hallucination(
            model=judge_model,
            temperature=0.0,
            track=False,
        )
        # Pin the native Anthropic SDK path for claude judges (addendum
        # resolution): fail fast rather than silently degrade to LiteLLM.
        if judge_model.lower().startswith(("claude-", "anthropic/")):
            resolved = type(self._metric._model).__name__
            if resolved != "AnthropicChatModel":
                raise RuntimeError(
                    f"Opik resolved {judge_model!r} to {resolved}, not the "
                    "native AnthropicChatModel — is the anthropic package "
                    "missing from the environment?")
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
