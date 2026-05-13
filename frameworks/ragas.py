"""RAGAS faithfulness adapter.

RAGAS ``faithfulness`` returns a score in [0, 1] where higher = more
faithful. RAGAS does not produce a pass/fail; we apply the standard 0.5
threshold so the verdict comparison is meaningful.
"""
from __future__ import annotations

import time
from typing import Any

from data.loader import Case
from .base import FrameworkResult, FrameworkRunner


class RagasFaithfulness(FrameworkRunner):
    name = "ragas"

    def __init__(self, judge_model: str = "gpt-4o-mini", threshold: float = 0.5):
        # Lazy-imported.
        from ragas.metrics import Faithfulness  # noqa
        from ragas.llms import LangchainLLMWrapper  # noqa
        from langchain_openai import ChatOpenAI  # noqa
        self._Faithfulness: Any = Faithfulness
        self._LangchainLLMWrapper: Any = LangchainLLMWrapper
        self._ChatOpenAI: Any = ChatOpenAI
        self._judge_model = judge_model
        self._threshold = threshold

    def run(self, case: Case) -> FrameworkResult:
        t0 = time.perf_counter()
        try:
            # RAGAS faithfulness wants a SingleTurnSample with user_input,
            # response, retrieved_contexts.
            from ragas.dataset_schema import SingleTurnSample  # noqa
            llm = self._LangchainLLMWrapper(self._ChatOpenAI(
                model=self._judge_model, temperature=0.0,
            ))
            metric = self._Faithfulness(llm=llm)
            sample = SingleTurnSample(
                user_input=case.question or "Provide a faithful summary of the document.",
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
