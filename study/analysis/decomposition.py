"""P2 — variance decomposition on repeated cells (plan §6, H2).

Per judge×task with R ≥ 2, Condition A errors-as-failures verdicts
v_ifr ∈ {0,1}:
  π_f  = E_i E_{r<r'} 1(v_ifr ≠ v_ifr')          judge-noise floor
  δ_fg = E_i E_{r,r'} 1(v_ifr ≠ v_igr')          cross-framework
  Δ_fg = δ_fg − (π_f + π_g)/2                    BCa item-cluster CI
  B/W  (Design 2 finite-run bias-corrected): per item W_i = mean_f
  var_r(Y_ifr), B_i = var_f(mean_r Y_ifr) − W_i/R; negatives NOT truncated.

Stated model (addendum §12.6, framing only — math unchanged): per item i,
framework f's runs are iid Bernoulli(p_if) given the item; then
Δ_fg = E_i[(p_if − p_ig)²] ≥ 0, zero iff p_if = p_ig a.s. Outside that
model Δ_fg reads descriptively as cross-framework mismatch in excess of
the mean within-framework mismatch.

Normalization convention (addendum §12.7, stated not changed): all sample
variances use ddof=1 (denominators R−1 across runs, F−1 across
frameworks); under ddof=1 with independent run noise the unbiased
finite-run correction is exactly W_i/R. Pooling = unweighted mean over
items; B is dispersion among the selected configurations, not a
population variance component.

Decision rules (addendum §12.1–.2): H2 confirmation and falsification (c)
both use the strict-majority pair count (> n_pairs/2), i.e. ≥6/10 on the
full 5-framework set and ≥4/6 on the P5 kill-switch set; falsification (c)
additionally requires pooled B ≤ W.

Plus pass@k / pass^k via multivon-eval 0.16.0's passk module (success =
verdict matches gold), and per-item flip probability across runs vs across
frameworks. All of these SKIP with a reason when R = 1.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import stats
from .data import Cell


def _per_item_terms(cell: Cell, frameworks: list[str]) -> dict:
    """Per-item π_f and pairwise δ_fg contribution arrays (n,)."""
    R, n = cell.R, cell.n
    p = {fw: cell.v["A"][fw].astype(float).mean(axis=0) for fw in frameworks}
    pi = {}
    for fw in frameworks:
        c = cell.v["A"][fw].sum(axis=0).astype(float)   # flags among R runs
        # disagreeing unordered run pairs = c(R-c) of C(R,2)
        pi[fw] = c * (R - c) / (R * (R - 1) / 2.0)
    delta = {}
    for fa, fb in itertools.combinations(frameworks, 2):
        delta[(fa, fb)] = p[fa] * (1 - p[fb]) + (1 - p[fa]) * p[fb]
    return {"p": p, "pi": pi, "delta": delta}


def p2_cell(cell: Cell, frameworks: list[str], n_boot: int) -> dict:
    R = cell.R
    if R < 2:
        return {"skipped": f"repeated-cell analysis requires R >= 2 runs; "
                           f"this cell has R={R} "
                           f"({cell.split} split, {cell.judge}/{cell.task})"}
    terms = _per_item_terms(cell, frameworks)
    out = {"judge": cell.judge, "task": cell.task, "R": R,
           "n_items": cell.n, "n_boot": n_boot,
           "pi_f": {}, "pairs": {}}
    for fw in frameworks:
        est, ci = stats.percentile_boot_mean(
            terms["pi"][fw], n_boot, np.random.default_rng(
                [stats.SEED, 21, _code(cell), _fwcode(fw)]))
        out["pi_f"][fw] = {"estimate": round(est, 4),
                           "ci95": [round(x, 4) for x in ci]}
    n_pos = 0
    n_incl0 = 0
    for (fa, fb), d in terms["delta"].items():
        w = d - (terms["pi"][fa] + terms["pi"][fb]) / 2.0
        theta, ci = stats.bca_ci_mean(
            w, n_boot, np.random.default_rng(
                [stats.SEED, 22, _code(cell), _fwcode(fa), _fwcode(fb)]))
        excl0 = ci[0] > 0 or ci[1] < 0
        n_pos += int(theta > 0 and ci[0] > 0)
        n_incl0 += int(not excl0)
        out["pairs"][f"{fa} <-> {fb}"] = {
            "delta_fg": round(float(d.mean()), 4),
            "pi_mean": round(float(((terms["pi"][fa]
                                     + terms["pi"][fb]) / 2).mean()), 4),
            "Delta_fg": round(theta, 4),
            "Delta_ci95_bca": [round(x, 4) for x in ci],
            "ci_excludes_0": bool(excl0),
        }
    n_pairs = len(terms["delta"])
    # Strict majority (> n_pairs/2) = ≥6/10 full set, ≥4/6 kill-switch set
    # (addendum §12.1–.2).
    out["h2_pairs_positive_ci_excl0"] = f"{n_pos}/{n_pairs}"
    out["h2_majority_positive"] = bool(n_pos > n_pairs / 2)
    out["pooled_BW"] = _pooled_bw(cell, frameworks, n_boot)
    bw = out["pooled_BW"]
    out["h2_falsification_c"] = {
        "rule": "fires iff strict majority of Delta_fg CIs include zero "
                "(>=6/10 full set, >=4/6 kill-switch set) AND pooled "
                "B <= W (addendum §12.1)",
        "pairs_ci_including_0": f"{n_incl0}/{n_pairs}",
        "majority_include_0": bool(n_incl0 > n_pairs / 2),
        "pooled_B_le_W": bool(bw["B"] <= bw["W"]),
        "fires": bool(n_incl0 > n_pairs / 2 and bw["B"] <= bw["W"]),
    }
    out["per_item_flip"] = _flip_probs(terms, frameworks)
    return out


def _pooled_bw(cell: Cell, frameworks: list[str], n_boot: int) -> dict:
    R = cell.R
    Y = np.stack([cell.v["A"][fw].astype(float) for fw in frameworks])
    # Y: (F, R, n). Sample variances ddof=1; negatives not truncated.
    W_i = Y.var(axis=1, ddof=1).mean(axis=0)                 # (n,)
    B_i = Y.mean(axis=1).var(axis=0, ddof=1) - W_i / R       # (n,)
    rng = np.random.default_rng([stats.SEED, 23, _code(cell)])
    idx = rng.integers(0, cell.n, size=(n_boot, cell.n))
    bW, bB = W_i[idx].mean(axis=1), B_i[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bR = np.where(bW > 0, bB / bW, np.nan)
    def ci(v):
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]
    W, B = float(W_i.mean()), float(B_i.mean())
    return {"W": round(W, 4), "W_ci95": ci(bW),
            "B": round(B, 4), "B_ci95": ci(bB),
            "B_minus_W": round(B - W, 4), "B_minus_W_ci95": ci(bB - bW),
            "B_over_W": round(B / W, 4) if W > 0 else None,
            "B_over_W_ci95": ci(bR),
            "h2_B_minus_W_ci_excl0": bool(np.percentile(bB - bW, 2.5) > 0)}


def _flip_probs(terms: dict, frameworks: list[str]) -> dict:
    within = float(np.mean([terms["pi"][fw].mean() for fw in frameworks]))
    across = float(np.mean([d.mean() for d in terms["delta"].values()]))
    return {"mean_within_framework_across_runs": round(within, 4),
            "mean_across_frameworks": round(across, 4)}


def passk_cell(cell: Cell, frameworks: list[str], gold: np.ndarray) -> dict:
    """pass@k / pass^k (k=1..R) per framework via multivon-eval 0.16.0's
    passk module; success = Condition A verdict matches gold."""
    if cell.R < 2:
        return {"skipped": f"pass@k/pass^k need repeated runs; R={cell.R}"}
    from multivon_eval import passk
    out = {"judge": cell.judge, "task": cell.task, "R": cell.R,
           "success": "verdict == gold label (errors-as-failures)",
           "frameworks": {}}
    for fw in frameworks:
        correct = (cell.v["A"][fw].astype(bool) == gold[None, :])
        case_stats = [(cell.R, int(c)) for c in correct.sum(axis=0)]
        fw_out = {}
        for k in range(1, cell.R + 1):
            for metric in (passk.METRIC_PASS_AT_K, passk.METRIC_PASS_HAT_K):
                res = passk.suite_pass_k(case_stats, k, metric=metric)
                fw_out[f"{metric} k={k}"] = {
                    "value": round(res.value, 4),
                    "ci95": [round(res.ci_low, 4), round(res.ci_high, 4)]}
        out["frameworks"][fw] = fw_out
    return out


def _code(cell: Cell) -> int:
    from .data import JUDGES, TASKS
    if cell.judge not in JUDGES:        # synthetic selftest cells
        return 99
    return JUDGES.index(cell.judge) * 10 + TASKS.index(cell.task)


def _fwcode(fw: str) -> int:
    from .data import FRAMEWORKS
    return FRAMEWORKS.index(fw)
