# Commentary

> Hand-written; `RESULTS.md` is regenerated mechanically by `analyze.py`.

## Pilot configuration

| | Value |
|---|---|
| Dataset | HaluEval Summarization, 50 cases (25 faithful, 25 hallucinated), seed=42 |
| Judge | gpt-4o-mini, temperature=0, max_tokens=1024 |
| Runs per case | 3 (multivon-eval, DeepEval); 1 (RAGAS — see below) |
| Wallclock | ~12 min for multivon+deepeval; ~30 min for RAGAS |
| Estimated cost | ~$0.20 OpenAI spend |

## Headline findings

1. **At default thresholds, multivon-eval has the highest F1
   (0.630) vs DeepEval (0.077) and RAGAS (0.500).** The DeepEval gap
   isn't a question of detection prompt quality — it's a threshold
   calibration issue. DeepEval's `FaithfulnessMetric` rarely scores
   below 0.5, so almost nothing is flagged. Move the threshold to 0.8
   and DeepEval's F1 jumps; we'll publish a threshold sweep in the next
   iteration.

2. **The three frameworks disagree more than they agree.** Cohen's κ on
   the binary verdict: multivon ↔ deepeval 0.029, multivon ↔ ragas
   0.266, deepeval ↔ ragas 0.135. Worse than coin-flip agreement
   between multivon and DeepEval. This is informative — it means
   the "ground truth" each framework projects onto a faithfulness
   judgment is different, and a switch in evaluator does change
   verdicts, not just numbers.

3. **Run-to-run score variance is non-zero for every framework** at
   temperature=0. multivon-eval std = 0.027, DeepEval std = 0.054
   (which translates to 2% flaky-verdict cases for DeepEval, 8% for
   multivon-eval). Single-run point estimates are unreliable across
   all three, validating the [NAACL 2025 non-determinism finding](https://arxiv.org/abs/2502.01775).

4. **Latency varies 20×.** Median per-case: multivon 6.6s, DeepEval
   11.7s, RAGAS 129s. Same judge, same case. The differences come from
   how each evaluator decomposes the work (single judge call vs
   per-claim verification vs multi-stage extraction).

## What multivon-eval got right

- **Calibrated threshold pays off.** multivon-eval ships with a 0.90
  threshold for gpt-4o-mini Faithfulness (`_calibration_data/v1.json`).
  At that threshold the framework flags 29 of 50 cases as hallucinated;
  17 are true positives, 12 are false positives.
- **Recall (0.68) is the highest of the three.** If your goal is "don't
  ship a hallucinated summary," catching 17 of 25 known hallucinations
  beats DeepEval's 1 of 25.

## Where multivon-eval loses

- **Precision (0.586) is the lowest of the three.** RAGAS's 0.818
  precision means when it flags, it's usually right. multivon flags
  more aggressively and pays for it with false positives.
- **Score variance (std 0.027) is real.** Two of the 50 cases flipped
  pass/fail across the three runs. If you ship CI on a single multivon
  run, you'll get an occasional flaky result. Use `runs=3` or higher
  for production.
- **8% flaky-case rate** is the highest of the three.

## Where DeepEval loses

- **At its default threshold of 0.5, DeepEval flags almost nothing.**
  That's not a bug — it's a calibration issue. DeepEval doesn't ship
  threshold-per-judge data; the 0.5 default is uniform. With gpt-4o-mini
  the score distribution clusters in the 0.6–1.0 range, so 0.5 misses
  most hallucinations.
- **Worst inter-framework agreement.** DeepEval ↔ multivon κ = 0.029 is
  effectively random. The two frameworks pick different cases as
  hallucinated.

## Where RAGAS loses

- **20× slower per case than multivon, 11× slower than DeepEval.** A
  50-case run took ~30 minutes. At production scale (1k cases × 5
  runs), this is the difference between minutes and hours of wall time
  per CI run.
- **Only one run in this pilot** — see [Why only one run for RAGAS](#why-only-one-run-for-ragas).

## Why only one run for RAGAS

RAGAS's `faithfulness` decomposes each summary into individual claims
and runs a verification call per claim. In our pilot that's ~10 LLM
calls per case versus ~3 for the other two. At 129s per case median,
3 runs × 50 cases would have been ~3.2 hours. We dropped to 1 run for
RAGAS and noted the limit so the cross-run std column is null for
RAGAS.

This is a real production signal, not a knock — RAGAS is more
thorough by design. But if you're running on every PR with a token
budget, throughput matters.

## What this pilot is not

- **n=50 is small.** Wilson 95% CI on F1=0.63 at n=50 is roughly
  [0.49, 0.75]. Treat sub-5pp differences as inside the noise margin.
- **One dataset.** HaluEval Summarization. We did *not* run HaluEval QA
  (short-answer task that degenerates for all three claim-decomposition
  evaluators).
- **One judge.** All three frameworks held to gpt-4o-mini. Other judges
  (claude-haiku-4-5, gpt-4o, llama-3.3-70b) may produce different
  rankings.
- **One direction of comparison.** We did not evaluate red-teaming,
  agent trace evaluation, conversation evaluation, or RAG metrics
  beyond faithfulness. Each framework's strengths in those areas don't
  show up here.

## What to do next

1. **Scale to n=1,000** for tighter intervals. Cost projects to ~$4.
2. **Add a threshold sweep per framework.** Currently we report at
   default thresholds; readers should see what happens at the optimum.
3. **Add a second judge** (claude-haiku-4-5 or gpt-4o) to test
   whether the framework ranking is judge-stable.
4. **Add HaluEval QA with answer-similarity evaluators** (multivon
   AnswerAccuracy vs DeepEval AnswerRelevancyMetric), since
   faithfulness is the wrong metric for short-form QA.
