"""P1 (pairwise κ, median-κ CI) + agreement secondaries (AC1, McNemar).

P1 (plan §7): pairwise Cohen's κ matrix + raw disagreement per
judge×task×condition on run 1 (run index 0); confirmatory statistic =
median pairwise κ with percentile item-cluster bootstrap CI. Condition B
applies the locked τ (H4 secondary). Kill-switch variants exclude
multivon-eval (P5).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import stats
from .data import Cell


def run0_matrix(cell: Cell, cond: str, frameworks: list[str]) -> np.ndarray:
    """(F, n) verdict matrix, run 0, errors-as-failures."""
    return np.stack([cell.v[cond][fw][0] for fw in frameworks])


def p1_cell(cell: Cell, cond: str, frameworks: list[str], n_boot: int,
            rng: np.random.Generator) -> dict:
    V = run0_matrix(cell, cond, frameworks)
    pairs = list(itertools.combinations(range(len(frameworks)), 2))
    kappa, disagree, ac1 = {}, {}, {}
    for f, g in pairs:
        key = f"{frameworks[f]} <-> {frameworks[g]}"
        kappa[key] = round(stats.cohen_kappa(V[f], V[g]), 4)
        disagree[key] = round(float((V[f] != V[g]).mean()), 4)
        ac1[key] = round(stats.gwet_ac1(V[f], V[g]), 4)
    med = stats.median_kappa_ci(V, n_boot, rng)
    return {
        "judge": cell.judge, "task": cell.task, "condition": cond,
        "frameworks": frameworks, "n_items": cell.n,
        "flag_rates": {fw: round(float(cell.v[cond][fw][0].mean()), 4)
                       for fw in frameworks},
        "pairwise_kappa": kappa,
        "pairwise_raw_disagreement": disagree,
        "pairwise_gwet_ac1": ac1,
        "median_gwet_ac1": round(float(np.median(list(ac1.values()))), 4),
        "median_pairwise_kappa": round(med["median_kappa"], 4),
        "median_kappa_ci95": [round(x, 4) for x in med["ci95"]],
        "n_boot": med["n_boot"],
        # H1/H4 bar (plan §1): CI upper bound < 0.40
        "h_bar_0.40_upper_lt": bool(med["ci95"][1] < 0.40),
    }


def mcnemar_family(cell: Cell, frameworks: list[str]) -> dict:
    """McNemar exact tests over all framework pairs, Holm-corrected within
    this judge×task family (plan §9). Condition A, run 0."""
    V = run0_matrix(cell, "A", frameworks)
    raw, detail = {}, {}
    for f, g in itertools.combinations(range(len(frameworks)), 2):
        key = f"{frameworks[f]} <-> {frameworks[g]}"
        b = int(((V[f] == 1) & (V[g] == 0)).sum())
        c = int(((V[f] == 0) & (V[g] == 1)).sum())
        raw[key] = stats.mcnemar_exact(b, c)
        detail[key] = {"b_only_first_flags": b, "c_only_second_flags": c}
    adj = stats.holm(raw)
    return {"judge": cell.judge, "task": cell.task,
            "family": "all framework pairs within judge x task",
            "pairs": {k: {**detail[k], "p_exact": round(raw[k], 6),
                          "p_holm": round(adj[k], 6),
                          "significant_.05": bool(adj[k] < 0.05)}
                      for k in sorted(raw)}}


def kappa_self_cross(cell: Cell, frameworks: list[str], n_boot: int,
                     rng: np.random.Generator) -> dict:
    """κ_self(f) = mean κ over within-framework run pairs; κ_cross(f,g) over
    the R×R between-framework run pairs. Contrast min_f κ_self −
    max_{f,g} κ_cross with item-cluster bootstrap CI (plan §6). Repeated
    cells only. Exploratory per addendum §7 demotion."""
    R = cell.R
    if R < 2:
        return {"skipped": f"requires repeated runs (R={R})"}
    V = {fw: cell.v["A"][fw].astype(float) for fw in frameworks}

    def contrast(idx=None):
        def col(fw, r):
            x = V[fw][r]
            return x if idx is None else x[idx]        # (n,) or (B, n)

        def kap(a, b):
            po = (a == b).mean(axis=-1)
            pa, pb = a.mean(axis=-1), b.mean(axis=-1)
            pe = pa * pb + (1 - pa) * (1 - pb)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(pe >= 1.0, 0.0, (po - pe) / (1 - pe))

        selfs = [np.mean([kap(col(fw, r1), col(fw, r2))
                          for r1, r2 in itertools.combinations(range(R), 2)],
                         axis=0) for fw in frameworks]
        crosses = [np.mean([kap(col(fa, r1), col(fb, r2))
                            for r1 in range(R) for r2 in range(R)], axis=0)
                   for fa, fb in itertools.combinations(frameworks, 2)]
        return selfs, crosses

    selfs, crosses = contrast()
    point = float(np.min(selfs) - np.max(crosses))
    n = cell.n
    idx = rng.integers(0, n, size=(n_boot, n))
    bs, bc = contrast(idx)
    boots = np.min(np.stack(bs), axis=0) - np.max(np.stack(bc), axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fw_pairs = list(itertools.combinations(frameworks, 2))
    return {
        "judge": cell.judge, "task": cell.task, "R": R,
        "kappa_self": {fw: round(float(s), 4)
                       for fw, s in zip(frameworks, selfs)},
        "kappa_cross": {f"{a} <-> {b}": round(float(c), 4)
                        for (a, b), c in zip(fw_pairs, crosses)},
        "contrast_min_self_minus_max_cross": round(point, 4),
        "contrast_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_boot": n_boot,
    }
