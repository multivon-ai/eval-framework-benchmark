"""multivon-eval Faithfulness adapter (QAG scoring).

We use the calibrated threshold for the resolved judge model when one
exists in the shipped ``_calibration_data/v1.json``; otherwise fall back
to multivon-eval's library default of 0.7.
"""
from __future__ import annotations

import time

from data.loader import Case
from multivon_eval import EvalCase, JudgeConfig
from multivon_eval.calibration import calibrated_threshold
from multivon_eval.evaluators.llm_judge import Faithfulness

from .base import FrameworkResult, FrameworkRunner


def _infer_provider(judge_model: str) -> str:
    """Auto-detect provider from a model id so callers can pass --judge
    claude-haiku-4-5 without also specifying --provider anthropic.

    Falls through to ``openai`` for unknown prefixes so OpenAI-compatible
    local endpoints (vLLM/Ollama serving llama-*, qwen-*, ...) keep
    working when paired with ``--judge-base-url``.
    """
    m = judge_model.lower()
    if m.startswith(("claude-", "anthropic/")):
        return "anthropic"
    if m.startswith(("gpt-", "openai/")):
        return "openai"
    if m.startswith(("gemini-", "google/")):
        return "litellm"  # via LiteLLM
    return "openai"


class MultivonFaithfulness(FrameworkRunner):
    name = "multivon-eval"

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        provider = _infer_provider(judge_model)
        self._judge = JudgeConfig(
            provider=provider, model=judge_model, temperature=0.0,
        ).resolve()
        self._threshold = calibrated_threshold("faithfulness", self._judge)
        self._evaluator = Faithfulness(threshold=self._threshold, judge=self._judge)

    def run(self, case: Case) -> FrameworkResult:
        ev_case = EvalCase(
            input=case.question or f"Summarize this document.\n\n{case.context}",
            context=case.context,
        )
        t0 = time.perf_counter()
        try:
            result = self._evaluator.evaluate(ev_case, case.answer)
            err = None
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
            score=float(result.score),
            # multivon Faithfulness returns score in [0,1] where higher = more faithful.
            # "flagged_hallucinated" is the framework's pass/fail inverted.
            flagged_hallucinated=not result.passed,
            threshold=self._threshold,
            latency_ms=latency_ms,
            raw={"reason": result.reason},
            error=err,
        )
