# Results

## sum (n=50)

| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |
|---|---|---|---|---|---|---|---|---|
| multivon-eval | 0.9 | 0.6296 | 0.5862 | 0.68 | 0.0267 | 0.08 | 6562.0 | 0 |
| deepeval | 0.5 | 0.0769 | 1.0 | 0.04 | 0.0543 | 0.02 | 11752.1 | 1 |
| ragas | 0.5 | 0.5 | 0.8182 | 0.36 | None | 0.0 | 129049.8 | 0 |

### Inter-framework verdict disagreement

| Pair | Cases flipped | Cohen's κ |
|---|---|---|
| multivon-eval ↔ deepeval | 28/50 (56%) | 0.0291 |
| multivon-eval ↔ ragas | 20/50 (40%) | 0.2658 |
| deepeval ↔ ragas | 10/50 (20%) | 0.1349 |

### Threshold sweep

F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.

**multivon-eval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.148 | 1.000 | 0.080 |
| 0.50 | 0.267 | 0.800 | 0.160 |
| 0.60 | 0.424 | 0.875 | 0.280 |
| 0.70 | 0.564 | 0.786 | 0.440 |
| 0.80 | 0.638 | 0.682 | 0.600 |
| 0.90 | 0.630 | 0.586 | 0.680 |
| 0.95 | 0.679 | 0.613 | 0.760 |

**deepeval**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.000 | 0.000 | 0.000 |
| 0.50 | 0.077 | 1.000 | 0.040 |
| 0.60 | 0.258 | 0.667 | 0.160 |
| 0.70 | 0.465 | 0.556 | 0.400 |
| 0.80 | 0.667 | 0.621 | 0.720 |
| 0.90 | 0.706 | 0.558 | 0.960 |
| 0.95 | 0.696 | 0.545 | 0.960 |

**ragas**

| Threshold | F1 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.214 | 1.000 | 0.120 |
| 0.50 | 0.500 | 0.818 | 0.360 |
| 0.60 | 0.537 | 0.688 | 0.440 |
| 0.70 | 0.583 | 0.609 | 0.560 |
| 0.80 | 0.604 | 0.571 | 0.640 |
| 0.90 | 0.610 | 0.529 | 0.720 |
| 0.95 | 0.610 | 0.529 | 0.720 |

