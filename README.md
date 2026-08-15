# eval-framework-benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Last run](https://img.shields.io/badge/last%20run-2026--06--26-emerald)](results/RESULTS.md)
[![multivon-eval (tagged run)](https://img.shields.io/badge/multivon--eval%20(tagged%20run)-0.15.1-emerald)](https://github.com/multivon-ai/multivon-eval)
[![current release](https://img.shields.io/badge/current%20release-0.16.0-blue)](https://github.com/multivon-ai/multivon-eval/releases)

**[Live results page](https://multivon.ai/benchmark)** · [RESULTS.md](results/RESULTS.md) · [multivon-eval (engine)](https://github.com/multivon-ai/multivon-eval)

Reproducible head-to-head benchmark of open-source LLM evaluation frameworks
on hallucination detection.

## Headline result: the frameworks barely agree with each other (ragtruth-sum n=100, judge: gpt-4o-mini)

Given the **same items, same judge (gpt-4o-mini, temp 0), same seed**, the
three frameworks' binary hallucinated/faithful verdicts agree only marginally
above chance. Cohen's κ:

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ DeepEval | 33/100 (33%) | 0.038 |
| multivon-eval ↔ RAGAS | 33/100 (33%) | 0.038 |
| DeepEval ↔ RAGAS | 2/100 (2%) | -0.010 |

κ ≈ 0 means "no better than chance agreement" (κ = 1 is perfect, κ = 0 is
random). So *which framework you pick changes which cases get flagged*, not
just the headline number. This is the finding that doesn't depend on trusting
any one framework's adapter — it holds no matter whose scoring you believe,
because it's a comparison *between* them on identical inputs. If you're
choosing an eval framework, this is the result that should worry you: they are
not interchangeable. (The DeepEval ↔ RAGAS pair agrees more only because at
their default 0.5 threshold both flag almost nothing — agreement on "nothing
is a hallucination" is cheap.)

Full agreement tables, including the HaluEval Sum run, are in
[`results/RESULTS.md`](results/RESULTS.md).

### And when you *do* score against human labels, calibrated defaults matter

Only after the agreement question do the accuracy numbers become useful — and
here the story is about **default thresholds**, not about any framework
"winning." multivon-eval is one of the frameworks under test (we maintain it),
so read this table as "calibrated defaults beat uncalibrated ones," not as an
endorsement:

| Framework | Threshold | F1 | Precision | Recall | Errors |
|---|---|---|---|---|---|
| **multivon-eval** 0.15.1 | 0.90 | **0.729** | 0.912 | 0.608 | 0 |
| DeepEval 4.0.2 | 0.50 (default) | 0.038 | 1.000 | 0.020 | 0 |
| RAGAS 0.4.3 | 0.50 (default) | 0.038 | 1.000 | 0.020 | 4 |

At best-tuned thresholds (0.95), multivon-eval F1 reaches **0.837** vs DeepEval **0.609** vs RAGAS **0.812** — so the gap at defaults is mostly a calibration gap, not a fundamental accuracy gap. At their default 0.5 threshold both DeepEval and RAGAS flag almost nothing (recall 0.02): F1 0.04, not a literal zero, but effectively no signal until you tune the threshold. multivon-eval ships a calibrated default (0.90) and needs no tuning. All numbers are reproducible from the code in this repo. n=100, single run; Wilson 95% CI on F1 at this size is ≈ ±10pp.

Snapshot: multivon-eval 0.15.1 / DeepEval 4.0.2 / RAGAS 0.4.3, run 2026-06-26 (git tag `results-0.15.1-2026-06-26`). RAGAS errored on every case in the prior 0.9.8 harness; with ragas 0.4.3 it now completes (4/100 cases still error, and it runs ~15x slower than the others).

**Why this exists:** every framework (DeepEval, RAGAS, multivon-eval included) claims accuracy on faithfulness/hallucination detection, and none publishes a side-by-side comparison with the same judge, same dataset, same seed. This repo is that comparison.

**Measures (published today):**
- F1 vs human labels at each framework's default threshold *and* at swept thresholds (so you see best-case for each framework)
- Inter-framework disagreement (Cohen's κ)
- Median per-case latency

**Designed but not yet in the published results** (the harness supports them via `--runs`; the 0.15.1 snapshot is single-run, and cost tracking is not implemented yet):
- Run-to-run variance over 5 repeated runs (same input, same seed)
- Cost per 100 cases

**Run it yourself:** all you need is an OpenAI key and either the Colab notebook or a `pip install`.

**Fastest reproduction of just our number** (skips DeepEval + RAGAS install, ~30s + ~$0.01 for 10 cases):

```bash
pip install -r requirements.txt   # or just: pip install multivon-eval datasets pandas scikit-learn
export OPENAI_API_KEY=sk-...
python run.py --task ragtruth-sum --n 10 --only multivon-eval --judge gpt-4o-mini --out /tmp/repro
python analyze.py --results-dir /tmp/repro --task ragtruth-sum --n 10
```

For the full head-to-head you need DeepEval + RAGAS too (heavier installs with their own dependency trees); drop `--only multivon-eval` from the command above.

> Maintained by [Multivon](https://multivon.ai). multivon-eval is one of
> the frameworks tested. We tried to make the comparison fair; see
> [methodology](#methodology) for the calls we made. Where multivon-eval
> loses, it's documented in [`results/COMMENTARY.md`](results/COMMENTARY.md).

---

## Frameworks compared

| Framework | Metric used | Default threshold | Version |
|---|---|---|---|
| [multivon-eval](https://github.com/multivon-ai/multivon-eval) | `Faithfulness` (QAG) | 0.90 (gpt-4o-mini, calibrated) | 0.15.1 |
| [DeepEval](https://github.com/confident-ai/deepeval) | `FaithfulnessMetric` | 0.5 (default) | 4.0.2 |
| [RAGAS](https://github.com/explodinggradients/ragas) | `faithfulness` | 0.5 (default — pass at >=0.5) | 0.4.3 |

Judge: `gpt-4o-mini`, temperature=0, max_tokens=1024.

---

## Datasets

- **RAGTruth Summarization** (headline dataset) — 100-case sample
  (51 hallucinated, 49 faithful) from the Summary-task test split of
  [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al.,
  ACL 2024; Apache 2.0). Each case: a source document, a model-generated
  summary, and human span-level hallucination annotations collapsed to a
  binary label (`hallucinated` = any annotated span, `faithful` = none).
  Drawn deterministically with seed=42 by `data/ragtruth_loader.py`; the
  exact sample is committed as `data/ragtruth_sum_pilot_100.json`.
  RAGTruth is the **cross-dataset test**: multivon-eval's 0.90 threshold
  was calibrated on HaluEval Sum, never on RAGTruth, which removes the
  v1 calibration-circularity caveat.

- **HaluEval QA** — 100-case stratified sample (50 hallucinated, 50 faithful)
  from [HaluEval](https://github.com/RUCAIBox/HaluEval). Each case: a
  Wikipedia knowledge snippet, a question, a candidate answer, and a
  binary human label (`hallucinated` / `right`).

- **HaluEval Summarization** — 100-case stratified sample. Each case: a
  source document, a candidate summary, and a binary human label.

Subsets are deterministic (seed=42). Cached locally; first run downloads
the full dataset to `data/`.

---

## What we report

For each (framework × dataset):

| Metric | Definition | In 0.15.1 results? |
|---|---|---|
| **F1** | F1 of "framework says hallucinated" vs human label, at the framework's default threshold | Yes |
| **Precision / Recall** | At the same threshold | Yes |
| **Median latency (ms)** | Per case | Yes |
| **Std of score across repeated runs** | Same input, same seed, same temperature; quantifies judge stochasticity | No — snapshot is single-run; the v1 pilot (`results/COMMENTARY.md`) has 3-run variance |
| **Flaky case rate** | Fraction of cases where the binary verdict (pass/fail) changed across runs | No — single-run |
| **Cost ($)** | Total OpenAI spend per benchmark run | No — not yet instrumented (token counts are not captured) |

We also report **pairwise agreement** between frameworks (Cohen's κ) so
readers can see whether different scores actually disagree on which cases
are hallucinated, or just call them by different names.

---

## Methodology

The fair-comparison calls we made:

1. **Same judge for all three frameworks.** All wrap `gpt-4o-mini` with
   `temperature=0`. We do *not* let each framework use its own default
   judge (Anthropic for one, GPT-4o for another, …) because that
   conflates framework accuracy with judge accuracy.
2. **Same prompt formatting where possible.** Each framework's
   evaluator builds its own internal prompt. We don't override these —
   that *is* the framework's contribution. We document what each one
   sends to the judge in `frameworks/`.
3. Repeated runs where budget allows. Hosted-API judges have
   measurable variance even at temperature=0 (Atil et al., ACL 2025).
   The harness defaults to 5 runs per case (`--runs 5`); the published
   0.15.1 snapshot is 1 run per case to keep cost down, and the v1
   pilot used 3 runs.
4. **Default thresholds.** Each framework is scored at its own default.
   We also publish a threshold sweep so readers can see what F1 looks
   like at the optimum for each framework. multivon-eval ships with
   calibrated thresholds per judge (`0.90` for gpt-4o-mini on
   HaluEval Sum); the other two do not. That threshold was calibrated
   on HaluEval and the headline is measured on RAGTruth — a
   cross-dataset test, not an in-distribution one.
5. Same dataset split: all three frameworks see the same 100 cases
   in the same order, drawn deterministically with seed=42.
6. Cost, once instrumented, will include all retries (none of
   these frameworks expose retry costs separately). Cost tracking is
   not yet implemented — see [What we report](#what-we-report).

### Caveats and what this benchmark does NOT measure

- This is not a benchmark of judge model accuracy. We're comparing the
  *evaluator's prompt + parsing + scoring logic*, holding the judge
  constant. Some frameworks may perform better with their default
  judge; that's a different study.
- Nor is it a survey of every metric each framework offers. DeepEval has
  40+ evaluators and we're testing one (faithfulness); RAGAS has more
  than faithfulness too. This is the most directly comparable triple.
- 100 cases is pilot scale. Wilson 95% CI on F1=0.80 at n=100 is
  roughly [0.71, 0.87]. Useful for direction, not for sub-2pp claims.
  A 1,000-case run is on the roadmap.
- Agent, multi-turn, and structured-output evaluation are out of scope.

### Reproducibility

Every cell in `colab.ipynb` produces a deterministic intermediate
artifact. Raw judge responses are saved to `results/raw/`. The raw
files behind the headline ragtruth-sum result are committed in this
repo (`results/raw/{gpt-4o-mini,claude-haiku-4-5}/*_ragtruth-sum_run0.jsonl`,
~240 KB total); the run snapshot itself is marked by the git tag
`results-0.15.1-2026-06-26`. Other raw outputs from local re-runs are
gitignored to keep the repo small.

If you re-run and get different numbers, please open an issue with your
versions of openai, deepeval, ragas, multivon-eval, your OS, and the
exact diff against `results/`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run.py --smoke                                  # fast sanity check: 4 cases × 1 run
python run.py --task ragtruth-sum --n 100 --runs 1     # the headline configuration
python analyze.py --task ragtruth-sum --n 100
```

Or open `colab.ipynb` in Colab and run all cells.

---

## Results (pilot, n=100)

See [`results/RESULTS.md`](results/RESULTS.md). Updated whenever a
fresh run is committed.

## The Multivon ecosystem

This benchmark is one piece of a broader open-source AI evaluation stack:

| Repo | What it is |
|---|---|
| [multivon-eval](https://github.com/multivon-ai/multivon-eval) | The framework being benchmarked here. 44 evaluators + `bootstrap` CLI + `multivon_eval.auto`. |
| [pdfhell](https://github.com/multivon-ai/pdfhell) | Sibling adversarial benchmark — for PDFs, not text |
| [multivon-mcp](https://github.com/multivon-ai/multivon-mcp) | MCP server — call multivon-eval from inside Claude / Cursor |
| [eval-action](https://github.com/multivon-ai/eval-action) | GitHub Action — run multivon-eval on every PR |
| **eval-framework-benchmark** (you are here) | Head-to-head vs DeepEval + RAGAS |
| multivon-guard *(early access)* | Local proxy that catches LLM coding agents leaking secrets / PII |

## License

Apache 2.0.
