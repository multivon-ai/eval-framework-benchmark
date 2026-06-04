# Results

## qa (n=100)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | — | _no runs_ | — | — | — | — | — | — |
| deepeval | — | _no runs_ | — | — | — | — | — | — |
| ragas | — | _no runs_ | — | — | — | — | — | — |

## sum (n=100)  ·  judge: `gpt-4o-mini`

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.0 | 0.0 | 0.0 | None | 0.0 | 6280.6 | 0 |
| deepeval | 0.5 | 0.0 | 0.0 | 0.0 | None | 0.0 | 9309.3 | 1 |
| ragas | — | _no runs_ | — | — | — | — | — | — |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 14/25 (56%) | -0.0802 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0.000 | 0.000 | 0.000 |
| 0.80 | 0.000 | 0.000 | 0.000 |
| 0.90 | 0.000 | 0.000 | 0.000 |
| 0.95 | 0.000 | 0.000 | 0.000 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.000 | 0.000 | 0.000 |
| 0.60 | 0.000 | 0.000 | 0.000 |
| 0.70 | 0.000 | 0.000 | 0.000 |
| 0.80 | 0.000 | 0.000 | 0.000 |
| 0.90 | 0.000 | 0.000 | 0.000 |
| 0.95 | 0.000 | 0.000 | 0.000 |

