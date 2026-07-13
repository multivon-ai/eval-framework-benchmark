"""RAGAS faithfulness adapter.

RAGAS ``faithfulness`` returns a score in [0, 1] where higher = more
faithful. RAGAS does not produce a pass/fail; we apply the standard 0.5
threshold so the verdict comparison is meaningful.

Judge plumbing: gpt-* ids use ``LangchainLLMWrapper(ChatOpenAI(...))``
(unchanged from the pilot). claude-* ids use RAGAS's own canonical
``ragas.llms.llm_factory`` with ``provider="anthropic"`` and a native
Anthropic SDK client — ``PydanticPrompt.generate`` accepts these
instructor-based LLMs natively, so the metric's shipped prompts and
parser are untouched. (The pilot's OpenAI-only wiring simply cannot run
claude judges; this is transport plumbing, not configuration.)
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import STATIC_SUMMARIZATION_INPUT, FrameworkResult, FrameworkRunner


class RagasFaithfulness(FrameworkRunner):
    name = "ragas"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported.
        from ragas.metrics import Faithfulness  # noqa
        self._Faithfulness: Any = Faithfulness
        self._judge_model = judge_model
        self._threshold = threshold
        if judge_model.lower().startswith(("claude-", "anthropic/")):
            self._make_llm = self._make_anthropic_llm
        else:
            self._make_llm = self._make_openai_llm

    def _make_openai_llm(self) -> Any:
        from ragas.llms import LangchainLLMWrapper  # noqa
        from langchain_openai import ChatOpenAI  # noqa
        return LangchainLLMWrapper(ChatOpenAI(
            model=self._judge_model, temperature=0.0,
        ))

    def _make_anthropic_llm(self) -> Any:
        import anthropic  # noqa
        from ragas.llms import llm_factory  # noqa
        return llm_factory(
            self._judge_model.removeprefix("anthropic/"),
            provider="anthropic",
            client=anthropic.AsyncAnthropic(),
            temperature=0.0,
        )

    def run(self, case: Case) -> FrameworkResult:
        t0 = time.perf_counter()
        try:
            # RAGAS faithfulness wants a SingleTurnSample with user_input,
            # response, retrieved_contexts.
            from ragas.dataset_schema import SingleTurnSample  # noqa
            metric = self._Faithfulness(llm=self._make_llm())
            sample = SingleTurnSample(
                user_input=case.question or STATIC_SUMMARIZATION_INPUT,
                response=case.answer,
                retrieved_contexts=[case.context],
            )
            # ragas exposes both single_turn_score (preferred) and ascore (async).
            import asyncio
            score = asyncio.run(metric.single_turn_ascore(sample))
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
            flagged_hallucinated=score < self._threshold,
            threshold=self._threshold,
            latency_ms=latency_ms,
        )
