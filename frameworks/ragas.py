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
        m = self._judge_model.lower()
        if m.startswith("gpt-5") and "chat" not in m:
            # (import the real class — ragas.llms.LangchainLLMWrapper is a
            # DeprecationHelper shim that cannot be subclassed)
            from ragas.llms.base import LangchainLLMWrapper  # noqa
            # Reasoning-tier OpenAI judges (gpt-5.x) accept ONLY their
            # default temperature (1): explicit 0/0.01 is a 400
            # invalid_request_error. RAGAS hardwires a call-time judge
            # temperature via BaseRagasLLM.get_temperature (0.01 for n=1),
            # which bypasses langchain's own init-time gpt-5 temperature
            # drop. Override get_temperature to return the provider's sole
            # accepted value — the shipped prompts and parser are untouched
            # (transport plumbing, same class of fix as the Anthropic top_p
            # drop below; PREREG_ADDENDUM.md §11).
            class _DefaultTemperatureWrapper(LangchainLLMWrapper):
                def get_temperature(self, n: int) -> float:
                    return 1.0
            return _DefaultTemperatureWrapper(ChatOpenAI(model=self._judge_model))
        return LangchainLLMWrapper(ChatOpenAI(
            model=self._judge_model, temperature=0.0,
        ))

    def _make_anthropic_llm(self) -> Any:
        import anthropic  # noqa
        from ragas.llms import llm_factory  # noqa
        llm = llm_factory(
            self._judge_model.removeprefix("anthropic/"),
            provider="anthropic",
            client=anthropic.AsyncAnthropic(),
            temperature=0.0,
        )
        # RAGAS's InstructorModelArgs defaults top_p=0.1 alongside
        # temperature, and its Anthropic param mapping is pass-through.
        # claude-haiku-4-5 rejects requests specifying BOTH temperature and
        # top_p (400 invalid_request_error). Drop top_p so the Anthropic
        # path sends temperature=0.0 only — exactly what the OpenAI path
        # (ChatOpenAI(temperature=0.0), no top_p) already sends. Transport
        # plumbing, not metric configuration (PREREG_ADDENDUM.md §9).
        llm.model_args.pop("top_p", None)
        return llm

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
