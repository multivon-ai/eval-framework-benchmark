# Results

> Snapshot: multivon-eval 0.15.1 / DeepEval 4.0.2 / RAGAS 0.4.3, run 2026-06-26
> (git tag `results-0.15.1-2026-06-26`). Single run (`run0`) per framework
> x task, temperature 0 -- so the cross-run std and flaky-rate columns are
> empty. Raw judge responses are under `results/raw/{judge}/`.
>
> Regenerate: `python analyze.py --task ragtruth-sum --n 100 --judges gpt-4o-mini claude-haiku-4-5`
> (headline) and `python analyze.py --task sum --n 50 --judges gpt-4o-mini`
> (the HaluEval Sum / inter-framework-agreement section).

**RAGAS now completes.** It errored on every case in the 0.9.8 / 2026-06-05
harness; with ragas 0.4.3 it runs (gpt-4o-mini). A few cases still error
(4/100 on ragtruth-sum, 8/50 on sum) and it is ~13x slower than the others
(~130 s/case). At its default 0.5 threshold it flags almost nothing on
ragtruth-sum (F1 0.038), the same failure mode as DeepEval; swept to 0.95 it
reaches F1 0.812.

**Judge scope.** `gpt-4o-mini` is the headline judge and carries the full
three-way comparison. `claude-haiku-4-5` is reported for multivon-eval only:
DeepEval still errors on every case with a non-OpenAI judge in this harness
(its 65/100 "disagreement" row is an artifact of errored cases scored as
not-hallucinated -- ignore it), and RAGAS on haiku is not run (it is too slow
at ~130 s/case to be worth it for a secondary judge).

## ragtruth-sum (n=100)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.7294 | 0.9118 | 0.6078 | None | 0.0 | 10341.2 | 0 |
| deepeval | 0.5 | 0.0385 | 1.0 | 0.0196 | None | 0.0 | 12719.5 | 0 |
| ragas | 0.5 | 0.0385 | 1.0 | 0.0196 | None | 0.0 | 136122.2 | 4 |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 33/100 (33%) | 0.0385 |
| multivon-eval ↔ ragas | 33/100 (33%) | 0.0385 |
| deepeval ↔ ragas | 2/100 (2%) | -0.0101 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.075 | 1.000 | 0.039 |
| 0.70 | 0.210 | 1.000 | 0.118 |
| 0.80 | 0.418 | 0.875 | 0.275 |
| 0.90 | 0.729 | 0.912 | 0.608 |
| 0.95 | 0.837 | 0.872 | 0.804 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.038 | 1.000 | 0.020 |
| 0.60 | 0.107 | 0.600 | 0.059 |
| 0.70 | 0.194 | 0.545 | 0.118 |
| 0.80 | 0.384 | 0.636 | 0.275 |
| 0.90 | 0.569 | 0.569 | 0.569 |
| 0.95 | 0.609 | 0.547 | 0.686 |

**ragas**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.041 | 1.000 | 0.021 |
| 0.60 | 0.041 | 1.000 | 0.021 |
| 0.70 | 0.316 | 1.000 | 0.188 |
| 0.80 | 0.500 | 0.850 | 0.354 |
| 0.90 | 0.736 | 0.821 | 0.667 |
| 0.95 | 0.812 | 0.774 | 0.854 |

## ragtruth-sum (n=100)  ·  judge: `claude-haiku-4-5`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.6897 | 0.6154 | 0.7843 | None | 0.0 | 10065.5 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | None | 100 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 65/100 (65%) | 0.0 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.038 | 0.500 | 0.020 |
| 0.60 | 0.191 | 0.500 | 0.118 |
| 0.70 | 0.351 | 0.565 | 0.255 |
| 0.80 | 0.587 | 0.658 | 0.529 |
| 0.90 | 0.690 | 0.615 | 0.784 |
| 0.95 | 0.703 | 0.584 | 0.882 |

## sum (n=50)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.6296 | 0.5862 | 0.68 | None | 0.0 | 6798.7 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | 11338.5 | 0 |
| ragas | 0.5 | 0.4242 | 0.875 | 0.28 | None | 0.0 | 130039.2 | 8 |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 29/50 (58%) | -0.0 |
| multivon-eval ↔ ragas | 23/50 (46%) | 0.1703 |
| deepeval ↔ ragas | 8/50 (16%) | 0.0 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.214 | 1.000 | 0.120 |
| 0.50 | 0.267 | 0.800 | 0.160 |
| 0.60 | 0.471 | 0.889 | 0.320 |
| 0.70 | 0.513 | 0.714 | 0.400 |
| 0.80 | 0.558 | 0.667 | 0.480 |
| 0.90 | 0.630 | 0.586 | 0.680 |
| 0.95 | 0.630 | 0.586 | 0.680 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.364 | 0.750 | 0.240 |
| 0.70 | 0.565 | 0.619 | 0.520 |
| 0.80 | 0.566 | 0.536 | 0.600 |
| 0.90 | 0.623 | 0.528 | 0.760 |
| 0.95 | 0.623 | 0.528 | 0.760 |

**ragas**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.250 | 1.000 | 0.143 |
| 0.50 | 0.483 | 0.875 | 0.333 |
| 0.60 | 0.571 | 0.714 | 0.476 |
| 0.70 | 0.619 | 0.619 | 0.619 |
| 0.80 | 0.609 | 0.560 | 0.667 |
| 0.90 | 0.640 | 0.552 | 0.762 |
| 0.95 | 0.640 | 0.552 | 0.762 |
