# Commentary

> This file is hand-written; `RESULTS.md` is regenerated mechanically by
> `analyze.py`. Read both together.

## What this benchmark is and isn't

Three open-source faithfulness/hallucination evaluators, scored against
the human labels in HaluEval QA and HaluEval Summarization, using
**the same OpenAI judge** (gpt-4o-mini at temperature 0) for all three.
This isolates the evaluator's prompt + parsing + scoring contribution
from the underlying judge.

What we are *not* claiming:

- That multivon-eval is best on every dataset. The QA results in
  particular are interesting precisely because they expose a
  shared weakness of claim-decomposition-style faithfulness scoring on
  short-form QA — and we report it as a finding, not hide it.
- That this is a 1,000-case benchmark. It isn't. 100 cases per task is
  a *pilot*. Wilson 95% CI on F1 at n=100 is roughly ±5–8pp depending
  on F1; treat sub-2pp differences as noise.

## Headline findings (regenerate from RESULTS.md)

<!-- Filled in after each pilot run. Keep these to 3–5 punchy sentences. -->

1. _Faithfulness/hallucination evaluators agree on `<κ>` of cases — and
   disagree on the rest. The disagreements are not random; they cluster
   on `<pattern>`._
2. _Run-to-run score std on the same input is `<X>` for multivon-eval,
   `<Y>` for DeepEval, `<Z>` for RAGAS. This validates the original
   NAACL-2025 finding (single runs are not reliable point estimates)
   for all three frameworks, not just one._
3. _At default thresholds, F1 against human labels: `<table>`. The
   ranking is `<order>` on QA and `<order>` on Summarization._

## What QA results tell you

HaluEval QA pairs each context with a short answer (often 1–3 tokens
— a name, a date, a single fact). Claim-decomposition style faithfulness
evaluators (multivon-eval Faithfulness, DeepEval FaithfulnessMetric)
extract individual claims from the answer and verify each against the
context. **Short answers decompose to zero or one claims**, which
collapses the score distribution.

This isn't a multivon-eval problem — it's a methodology mismatch for
the task. We document it because pretending it doesn't happen would be
exactly the kind of overclaiming the framework critics were right to
flag.

A more meaningful QA evaluator is one designed to compare *answer
semantics against ground-truth*. multivon-eval ships `AnswerAccuracy`
for this; DeepEval ships `AnswerRelevancyMetric`. A follow-up benchmark
will compare those.

## What Summarization results tell you

Summarization is the natural home for claim-decomposition faithfulness:
candidate summaries decompose into multiple verifiable claims, and the
human labels in HaluEval Summarization were produced specifically to
test factual consistency at the claim level.

The summarization numbers are the headline.

## Honest limitations of multivon-eval shown by this pilot

- _List the cases where multivon-eval loses (lower F1, higher variance,
  or higher cost) here. Don't hide them. The first reviewer to spot
  a hidden weakness will discredit the whole benchmark._

## Honest limitations of DeepEval shown by this pilot

- _Same — keep it specific and cite case ids from `results/raw/`._

## Honest limitations of RAGAS shown by this pilot

- _Same._

## Open questions

- Does each framework's accuracy hold up at n=1000? (Run scheduled for
  next iteration.)
- Do these results hold for a different judge (claude-haiku-4-5, gpt-4o,
  llama-3.3-70b)? (Multi-judge sweep on the roadmap.)
- How does cost scale per million eval cases? Project the pilot cost.
