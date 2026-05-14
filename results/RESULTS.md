# Results

## ragtruth-sum (n=100)  ·  judge: `claude-haiku-4-5`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.6755 | 0.51 | 1.0 | None | 0.0 | 361.8 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | None | 100 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 100/100 (100%) | 0.0 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.675 | 0.510 | 1.000 |
| 0.50 | 0.675 | 0.510 | 1.000 |
| 0.60 | 0.675 | 0.510 | 1.000 |
| 0.70 | 0.675 | 0.510 | 1.000 |
| 0.80 | 0.675 | 0.510 | 1.000 |
| 0.90 | 0.675 | 0.510 | 1.000 |
| 0.95 | 0.675 | 0.510 | 1.000 |

## ragtruth-sum (n=100)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.7865 | 0.9211 | 0.6863 | None | 0.0 | 10744.5 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | 13520.3 | 0 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 38/100 (38%) | 0.0 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.038 | 1.000 | 0.020 |
| 0.70 | 0.179 | 1.000 | 0.098 |
| 0.80 | 0.375 | 0.923 | 0.235 |
| 0.90 | 0.786 | 0.921 | 0.686 |
| 0.95 | 0.854 | 0.911 | 0.804 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0.188 | 0.462 | 0.118 |
| 0.80 | 0.368 | 0.560 | 0.275 |
| 0.90 | 0.525 | 0.542 | 0.510 |
| 0.95 | 0.587 | 0.552 | 0.627 |

