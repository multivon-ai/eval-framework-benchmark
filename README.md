# eval-framework-benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Last run](https://img.shields.io/badge/last%20run-2026--05--17-emerald)](results/RESULTS.md)
[![multivon-eval](https://img.shields.io/badge/multivon--eval-0.8.2-emerald)](https://github.com/multivon-ai/multivon-eval)

**[Live results page](https://multivon.ai/benchmark)** · [RESULTS.md](results/RESULTS.md) · [multivon-eval (engine)](https://github.com/multivon-ai/multivon-eval)

Reproducible head-to-head benchmark of open-source LLM evaluation frameworks
on hallucination detection.

## Headline result (ragtruth-sum n=100, judge: gpt-4o-mini)

| Framework | Threshold | F1 | Precision | Recall | Errors |
|---|---|---|---|---|---|
| **multivon-eval** 0.8.2 | 0.90 | **0.787** | 0.921 | 0.686 | 0 |
| DeepEval (latest) | 0.50 (default) | 0.000 | 0.000 | 0.000 | 0 |
| RAGAS (latest) | — | _no runs_ | — | — | — |

At best-tuned thresholds (0.95), multivon-eval F1 reaches **0.854** vs DeepEval **0.587**. RAGAS errored on every run with the current test harness; documented in [`results/RESULTS.md`](results/RESULTS.md). All numbers reproducible — code in this repo.

**Why this exists:** Every framework — DeepEval, RAGAS, multivon-eval — claims accuracy on faithfulness/hallucination detection. None publish a side-by-side comparison with the same judge, same dataset, same seed. This repo does.

**Measures:**
- F1 vs human labels at each framework's default threshold *and* at swept thresholds (so you see best-case for each framework)
- Run-to-run variance over 5 repeated runs (same input, same seed)
- Cost per 100 cases
- Inter-framework disagreement (Cohen's κ)

**Run it yourself:** one Colab notebook, one `pip install`, one OpenAI key.

> Maintained by [Multivon](https://multivon.ai). multivon-eval is one of
> the frameworks tested. We tried to make the comparison fair; see
> [methodology](#methodology) for the calls we made. Where multivon-eval
> loses, it's documented in [`results/COMMENTARY.md`](results/COMMENTARY.md).

---

## Frameworks compared

| Framework | Metric used | Default threshold | Version |
|---|---|---|---|
| [multivon-eval](https://github.com/multivon-ai/multivon-eval) | `Faithfulness` (QAG) | 0.90 (gpt-4o-mini, calibrated) | 0.8.2 |
| [DeepEval](https://github.com/confident-ai/deepeval) | `FaithfulnessMetric` | 0.5 (default) | latest at 2026-05-17 |
| [RAGAS](https://github.com/explodinggradients/ragas) | `faithfulness` | 0.5 (default — pass at >=0.5) | latest at 2026-05-17 |

Judge: `gpt-4o-mini`, temperature=0, max_tokens=1024.

---

## Datasets

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

| Metric | Definition |
|---|---|
| **F1** | F1 of "framework says hallucinated" vs human label, at the framework's default threshold |
| **Precision / Recall** | At the same threshold |
| **Std of score across 5 runs** | Same input, same seed, same temperature; quantifies judge stochasticity |
| **Flaky case rate** | Fraction of cases where the binary verdict (pass/fail) changed across the 5 runs |
| **Cost ($)** | Total OpenAI spend for 100 cases × 5 runs |
| **Median latency (ms)** | Per case, single-threaded |

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
3. **5 runs per case.** Hosted-API judges have measurable variance even
   at temperature=0 (Atil et al., ACL 2025). 5 runs is a balance between
   signal and cost.
4. **Default thresholds.** Each framework is scored at its own default.
   We also publish a threshold sweep so readers can see what F1 looks
   like at the optimum for each framework. multivon-eval ships with
   calibrated thresholds per judge (`0.90` for gpt-4o-mini on
   HaluEval Sum); the other two do not.
5. **Same dataset split.** All three frameworks see the same 100 cases
   in the same order, drawn deterministically with seed=42.
6. **Cost includes all retries** (none of these frameworks expose
   retry costs separately).

### Caveats and what this benchmark does NOT measure

- **Not a benchmark of judge model accuracy.** We're comparing the
  *evaluator's prompt + parsing + scoring logic*, holding the judge
  constant. Some frameworks may perform better with their default
  judge — that's a different study.
- **Not all the metrics each framework offers.** DeepEval has 40+
  evaluators; we're testing one (faithfulness). RAGAS has more than
  faithfulness too. This is the most directly comparable triple.
- **100 cases is a pilot scale.** Wilson 95% CI on F1=0.80 at n=100 is
  roughly [0.71, 0.87]. Useful for direction, not for sub-2pp claims.
  A 1,000-case run is on the roadmap.
- **No agent / multi-turn / structured-output coverage.** Out of scope.

### Reproducibility

Every cell in `colab.ipynb` produces a deterministic intermediate
artifact. Raw judge responses are saved to `results/raw/` (gitignored to
keep the public repo small; included in archive releases).

If you re-run and get different numbers, please open an issue with your
versions of openai, deepeval, ragas, multivon-eval, your OS, and the
exact diff against `results/`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run.py --pilot
python analyze.py
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
