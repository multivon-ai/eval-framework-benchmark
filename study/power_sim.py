"""Preregistered power gate — plan §9.

10,000-replicate simulation per grid cell, balanced labels, framework
positive rates 0.20-0.80, kappa 0.20-0.70, paired error correlation
0-0.5. The gate: proceed to test-split spend only if power >= 0.80 at
n=300 for BOTH minimum effects:

  Endpoint A  dependent-kappa difference of 0.15
              (three raters on the same items: kappa(A,B) = kappa_hi,
               kappa(A,C) = kappa_hi - 0.15; the study's kappa_self vs
               kappa_cross contrast has exactly this dependent structure)
  Endpoint B  paired-F1 gap of 0.10
              (two frameworks vs gold labels on the same items, per-class
               errors correlated at rho via a Gaussian copula)

Per-replicate inference mirrors the study's item-cluster resampling with
a delta-equivalent grouped jackknife over the joint verdict categories
(exact leave-one-item-out; verdicts are exchangeable within category, so
the item jackknife collapses to <=8 category evaluations). Two-sided
alpha = 0.05. Escalation cells at n=500 (RAGTruth-Sum n->500 is the sole
preregistered escalation) are simulated alongside. Null cells (true
effect = 0) are included as a type-I calibration check.

Deterministic: numpy default_rng(42). Output: study/power_sim_results.json.

Run:  .venv-study/bin/python study/power_sim.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

SEED = 42
REPS = 10_000
ALPHA_Z = 1.959963984540054  # two-sided 0.05

POS_RATES = [0.20, 0.35, 0.50, 0.65, 0.80]
KAPPA_HI = [0.35, 0.50, 0.70]          # kappa_lo = kappa_hi - 0.15 spans 0.20-0.55
ERR_RHO = [0.0, 0.25, 0.50]
NS = [300, 500]
KAPPA_EFFECT = 0.15
F1_EFFECT = 0.10

OUT = Path(__file__).resolve().parent / "power_sim_results.json"


# ── Copula helpers ───────────────────────────────────────────────────────────

def _p11(r: float, t: float) -> float:
    """P(Z1>t, Z2>t) under bivariate normal with correlation r."""
    if abs(r) >= 0.999999:
        r = 0.999999 if r > 0 else -0.999999
    return float(multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-t, -t]))


def solve_corr_for_p11(target_p11: float, t: float) -> float:
    lo, hi = -0.999, 0.999
    if not (_p11(lo, t) - 1e-12 <= target_p11 <= _p11(hi, t) + 1e-12):
        raise ValueError(f"target P11 {target_p11:.4f} out of range at t={t:.3f}")
    for _ in range(80):
        mid = (lo + hi) / 2
        if _p11(mid, t) < target_p11:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def corr_for_kappa(kappa: float, p: float) -> float:
    """Latent correlation giving Cohen's kappa between two raters with equal
    marginal positive rate p."""
    pe = p * p + (1 - p) * (1 - p)
    po = kappa * (1 - pe) + pe
    p11 = p - (1 - po) / 2
    t = norm.ppf(1 - p)
    return solve_corr_for_p11(p11, t)


# ── Grouped jackknife over joint categories ──────────────────────────────────

def jackknife_power(counts: np.ndarray, stat_fn) -> float:
    """counts: (reps, K) integer category counts summing to n per row.
    stat_fn(counts) -> (reps,) statistic. Returns power = fraction of
    replicates whose |stat|/SE_jack > z_{.975}. Exact leave-one-out item
    jackknife (items within a category are interchangeable)."""
    reps, K = counts.shape
    n = counts[0].sum()
    full = stat_fn(counts)
    tks = np.empty((reps, K))
    for k in range(K):
        cm = counts.copy()
        cm[:, k] -= 1
        tks[:, k] = stat_fn(np.maximum(cm, 0))
    weights = counts / n
    t_dot = (weights * tks).sum(axis=1)
    var = (n - 1) / n * (counts * (tks - t_dot[:, None]) ** 2).sum(axis=1)
    se = np.sqrt(np.maximum(var, 1e-12))
    return float(np.mean(np.abs(full) / se > ALPHA_Z))


# ── Endpoint A: dependent-kappa difference ───────────────────────────────────

def _kappa_from_pair_counts(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    po = (n11 + n00) / n
    pa = (n11 + n10) / n
    pb = (n11 + n01) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    denom = np.maximum(1 - pe, 1e-9)
    return (po - pe) / denom


def kappa_diff_stat(counts: np.ndarray) -> np.ndarray:
    """counts columns = 8 joint categories of (A,B,C) in binary order
    (A*4 + B*2 + C). Returns kappa(A,B) - kappa(A,C)."""
    c = counts.astype(float)
    A = [4, 5, 6, 7]
    ab11 = c[:, 6] + c[:, 7]
    ab10 = c[:, 4] + c[:, 5]
    ab01 = c[:, 2] + c[:, 3]
    ab00 = c[:, 0] + c[:, 1]
    ac11 = c[:, 5] + c[:, 7]
    ac10 = c[:, 4] + c[:, 6]
    ac01 = c[:, 1] + c[:, 3]
    ac00 = c[:, 0] + c[:, 2]
    _ = A
    return _kappa_from_pair_counts(ab11, ab10, ab01, ab00) - \
        _kappa_from_pair_counts(ac11, ac10, ac01, ac00)


def simulate_kappa_cell(rng: np.ndarray, n: int, p: float, k_hi: float,
                        k_lo: float) -> float:
    r_ab = corr_for_kappa(k_hi, p)
    r_ac = corr_for_kappa(k_lo, p)
    # One-factor structure: corr(B,C) = r_ab * r_ac  (PSD by construction).
    cov = np.array([[1, r_ab, r_ac],
                    [r_ab, 1, r_ab * r_ac],
                    [r_ac, r_ab * r_ac, 1]])
    L = np.linalg.cholesky(cov)
    t = norm.ppf(1 - p)
    z = rng.standard_normal((REPS, n, 3)) @ L.T
    v = (z > t)
    idx = v[..., 0] * 4 + v[..., 1] * 2 + v[..., 2] * 1
    counts = np.stack([(idx == k).sum(axis=1) for k in range(8)], axis=1)
    return jackknife_power(counts, kappa_diff_stat)


# ── Endpoint B: paired-F1 gap ────────────────────────────────────────────────

def f1_from_counts(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)


def f1_diff_stat(counts: np.ndarray) -> np.ndarray:
    """counts columns = 8 categories (y, a_flag, b_flag) in binary order
    (y*4 + a*2 + b). Returns F1_A - F1_B (positive class = hallucinated=y=1)."""
    c = counts.astype(float)
    tp_a = c[:, 6] + c[:, 7]
    fp_a = c[:, 2] + c[:, 3]
    fn_a = c[:, 4] + c[:, 5]
    tp_b = c[:, 5] + c[:, 7]
    fp_b = c[:, 1] + c[:, 3]
    fn_b = c[:, 4] + c[:, 6]
    return f1_from_counts(tp_a, fp_a, fn_a) - f1_from_counts(tp_b, fp_b, fn_b)


def _oper_point(f1: float, p: float) -> tuple[float, float]:
    """Balanced classes: F1 = 2*sens/(1+2p) with p = (sens+fpr)/2."""
    sens = f1 * (1 + 2 * p) / 2
    fpr = 2 * p - sens
    if not (0 <= sens <= 1 and 0 <= fpr <= 1):
        raise ValueError(f"infeasible operating point f1={f1}, p={p}")
    return sens, fpr


def simulate_f1_cell(rng, n: int, p: float, rho: float, gap: float) -> float:
    f1_max = 2 * min(1.0, 2 * p) / (1 + 2 * p)
    f1_a = min(0.80, f1_max - 0.03)
    f1_b = f1_a - gap
    sens_a, fpr_a = _oper_point(f1_a, p)
    sens_b, fpr_b = _oper_point(f1_b, p)
    n_pos = n // 2
    cov = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov)

    def _flags(n_items, rate_a, rate_b):
        z = rng.standard_normal((REPS, n_items, 2)) @ L.T
        return (z[..., 0] < norm.ppf(rate_a)), (z[..., 1] < norm.ppf(rate_b))

    # Positive class: flag with prob sens; negative class: flag with prob fpr.
    a_pos, b_pos = _flags(n_pos, sens_a, sens_b)
    a_neg, b_neg = _flags(n - n_pos, fpr_a, fpr_b)

    idx_pos = 4 + a_pos * 2 + b_pos * 1
    idx_neg = 0 + a_neg * 2 + b_neg * 1
    counts = np.stack(
        [(idx_pos == k).sum(axis=1) + (idx_neg == k).sum(axis=1) for k in range(8)],
        axis=1)
    return jackknife_power(counts, f1_diff_stat)


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    results: dict = {
        "seed": SEED, "replicates": REPS, "alpha": 0.05,
        "effects": {"dependent_kappa_difference": KAPPA_EFFECT, "paired_f1_gap": F1_EFFECT},
        "kappa_diff": [], "f1_gap": [], "null_calibration": [],
    }

    for n in NS:
        for p in POS_RATES:
            for k_hi in KAPPA_HI:
                pw = simulate_kappa_cell(rng, n, p, k_hi, k_hi - KAPPA_EFFECT)
                results["kappa_diff"].append(
                    {"n": n, "pos_rate": p, "kappa_hi": k_hi,
                     "kappa_lo": round(k_hi - KAPPA_EFFECT, 2), "power": round(pw, 4)})
                print(f"kappa-diff n={n} p={p} k={k_hi}->{k_hi - KAPPA_EFFECT:.2f}: "
                      f"power={pw:.3f}", file=sys.stderr)
            for rho in ERR_RHO:
                pw = simulate_f1_cell(rng, n, p, rho, F1_EFFECT)
                results["f1_gap"].append(
                    {"n": n, "pos_rate": p, "error_rho": rho, "power": round(pw, 4)})
                print(f"f1-gap    n={n} p={p} rho={rho}: power={pw:.3f}", file=sys.stderr)

    # Type-I calibration (true effect = 0): rejection rate should be ~0.05.
    null_k = simulate_kappa_cell(rng, 300, 0.5, 0.5, 0.5)
    null_f = simulate_f1_cell(rng, 300, 0.5, 0.25, 0.0)
    results["null_calibration"] = [
        {"endpoint": "kappa_diff", "n": 300, "rejection_rate": round(null_k, 4)},
        {"endpoint": "f1_gap", "n": 300, "rejection_rate": round(null_f, 4)},
    ]

    for key in ("kappa_diff", "f1_gap"):
        cells300 = [c for c in results[key] if c["n"] == 300]
        cells500 = [c for c in results[key] if c["n"] == 500]
        results[f"min_power_{key}_n300"] = min(c["power"] for c in cells300)
        results[f"min_power_{key}_n500"] = min(c["power"] for c in cells500)

    gate = (results["min_power_kappa_diff_n300"] >= 0.80
            and results["min_power_f1_gap_n300"] >= 0.80)
    results["gate_rule"] = ("proceed only if power >= 0.80 for both minimum effects "
                            "at n=300 across the full preregistered grid (plan §9)")
    results["gate_passed_n300"] = gate
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    print(f"min power (kappa diff 0.15, n=300): {results['min_power_kappa_diff_n300']:.3f}")
    print(f"min power (F1 gap 0.10,     n=300): {results['min_power_f1_gap_n300']:.3f}")
    print(f"min power (kappa diff 0.15, n=500): {results['min_power_kappa_diff_n500']:.3f}")
    print(f"min power (F1 gap 0.10,     n=500): {results['min_power_f1_gap_n500']:.3f}")
    if not gate:
        print("\n" + "!" * 78)
        print("!!  POWER GATE FAILED at n=300 (plan §9).")
        print("!!  Preregistered sole escalation: RAGTruth-Sum n -> 500.")
        print("!!  Do NOT start test-split API spend until the escalation decision")
        print("!!  is recorded in study/PREREG_ADDENDUM.md / DEVIATIONS.md.")
        print("!" * 78)
        return 1
    print("\nPOWER GATE PASSED at n=300 — cleared to proceed per plan §9.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
