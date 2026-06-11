# Results

> Snapshot: multivon-eval 0.9.8 / DeepEval 4.0.5, run 2026-06-05
> (git tag `results-0.9.8-2026-06-05`). Single run (`run0`) per
> framework × judge — the cross-run std and flaky-rate columns are
> therefore empty. Raw judge responses for the ragtruth-sum runs are
> committed under `results/raw/{judge}/`.
>
> Regenerate the ragtruth-sum sections with
> `python analyze.py --task ragtruth-sum --n 100`; the HaluEval Sum
> section with `python analyze.py --task sum --n 50 --judges gpt-4o-mini`.

**RAGAS:** errored on every attempted case in the 2026-06-05 harness
(its `faithfulness` pipeline raised before producing a score with the
pinned dependency set), so no RAGAS raw files were persisted and it
appears as _no runs_ below. The older 0.8.2 archive
(`results/_archive_0.8.2_2026-05-17/`) and `results/COMMENTARY.md`
cover the v1 pilot where RAGAS did complete.

**DeepEval × claude-haiku-4-5:** all 100 cases errored (Errors = 100);
its F1 row and the 62/100 disagreement row for that judge are artifacts
of errored cases being scored as "not hallucinated" — ignore them.

## ragtruth-sum (n=100)  ·  judge: `claude-haiku-4-5`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.6726 | 0.6129 | 0.7451 | None | 0.0 | 9730.6 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | None | 100 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 62/100 (62%) | 0.0 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.075 | 1.000 | 0.039 |
| 0.60 | 0.254 | 0.667 | 0.157 |
| 0.70 | 0.405 | 0.652 | 0.294 |
| 0.80 | 0.545 | 0.649 | 0.471 |
| 0.90 | 0.673 | 0.613 | 0.745 |
| 0.95 | 0.710 | 0.603 | 0.863 |

## ragtruth-sum (n=100)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.7442 | 0.9143 | 0.6275 | None | 0.0 | 9026.5 | 0 |
| deepeval | 0.5 | 0.0385 | 1.0 | 0.0196 | None | 0.0 | 11972.7 | 0 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 34/100 (34%) | 0.0368 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.111 | 1.000 | 0.059 |
| 0.70 | 0.210 | 1.000 | 0.118 |
| 0.80 | 0.355 | 1.000 | 0.216 |
| 0.90 | 0.744 | 0.914 | 0.627 |
| 0.95 | 0.833 | 0.889 | 0.784 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.038 | 1.000 | 0.020 |
| 0.60 | 0.071 | 0.400 | 0.039 |
| 0.70 | 0.222 | 0.583 | 0.137 |
| 0.80 | 0.329 | 0.545 | 0.235 |
| 0.90 | 0.566 | 0.583 | 0.549 |
| 0.95 | 0.631 | 0.583 | 0.686 |


## sum (n=50)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.5926 | 0.5517 | 0.64 | None | 0.0 | 6585.1 | 0 |
| deepeval | 0.5 | 0.0741 | 0.5 | 0.04 | None | 0.0 | 10411.1 | 1 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 29/50 (58%) | -0.0112 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.148 | 1.000 | 0.080 |
| 0.50 | 0.387 | 1.000 | 0.240 |
| 0.60 | 0.471 | 0.889 | 0.320 |
| 0.70 | 0.619 | 0.765 | 0.520 |
| 0.80 | 0.605 | 0.722 | 0.520 |
| 0.90 | 0.593 | 0.552 | 0.640 |
| 0.95 | 0.593 | 0.552 | 0.640 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.074 | 0.500 | 0.040 |
| 0.50 | 0.074 | 0.500 | 0.040 |
| 0.60 | 0.250 | 0.571 | 0.160 |
| 0.70 | 0.522 | 0.571 | 0.480 |
| 0.80 | 0.604 | 0.571 | 0.640 |
| 0.90 | 0.697 | 0.561 | 0.920 |
| 0.95 | 0.697 | 0.561 | 0.920 |

