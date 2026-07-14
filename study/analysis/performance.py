"""P4 — default-τ precision/recall/F1 vs gold + F1 spread; plus the
label-dependent secondaries (balanced accuracy, prevalence-standardized
metrics) and per-framework ops (api_error_rate/cost/latency).

Primary: errors-as-failures (errored record = flagged), Condition A run 0.
Secondary: complete-case (errored records dropped per framework).
Condition B (locked τ) reported alongside for H4/figures.
"""
from __future__ import annotations

import numpy as np

from . import stats
from .data import Cell

PREVALENCES = (0.10, 0.25, 0.50)


def _vs_gold(pred: np.ndarray, gold: np.ndarray) -> dict:
    p, r, f1 = stats.prf1(pred, gold)
    sens, spec = stats.sens_spec(pred, gold)
    out = {"n": int(len(gold)), "precision": round(p, 4),
           "recall": round(r, 4), "f1": round(f1, 4),
           "balanced_accuracy": round((sens + spec) / 2, 4),
           "specificity": round(spec, 4)}
    out["prevalence_standardized"] = {
        f"prev={pv:.2f}": {k: round(v, 4) for k, v in
                           stats.prevalence_standardized(sens, spec, pv).items()}
        for pv in PREVALENCES}
    return out


def p4_cell(cell: Cell, frameworks: list[str], gold: np.ndarray,
            n_boot: int, rng: np.random.Generator) -> dict:
    out = {"judge": cell.judge, "task": cell.task, "n_items": cell.n,
           "n_boot": n_boot, "conditions": {}}
    for cond in ("A", "B"):
        by_fw = {}
        for fw in frameworks:
            pred = cell.v[cond][fw][0].astype(bool)
            err0 = cell.err[fw][0]
            entry = {"errors_as_failures": _vs_gold(pred, gold),
                     "n_errors": int(err0.sum())}
            keep = ~err0
            entry["complete_case"] = _vs_gold(pred[keep], gold[keep])
            by_fw[fw] = entry
        f1s = {fw: by_fw[fw]["errors_as_failures"]["f1"] for fw in frameworks}
        spread, ci = _spread_ci(cell, frameworks, cond, gold, n_boot, rng)
        out["conditions"][cond] = {
            "per_framework": by_fw,
            "f1_spread_max_minus_min": round(spread, 4),
            "f1_spread_ci95": [round(x, 4) for x in ci],
            "f1_max_framework": max(f1s, key=f1s.get),
            "f1_min_framework": min(f1s, key=f1s.get),
        }
    return out


def _spread_ci(cell: Cell, frameworks: list[str], cond: str,
               gold: np.ndarray, n_boot: int,
               rng: np.random.Generator) -> tuple[float, list[float]]:
    """max−min F1 across frameworks (errors-as-failures), percentile
    item-cluster bootstrap. Descriptive-with-CI per addendum §7 (the F1-gap
    test was demoted)."""
    P = np.stack([cell.v[cond][fw][0] for fw in frameworks]).astype(bool)
    G = gold.astype(bool)
    point = float(stats.f1_vec(P, G).max() - stats.f1_vec(P, G).min())
    n = cell.n
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot)
    for lo in range(0, n_boot, 1000):
        sl = slice(lo, min(lo + 1000, n_boot))
        f1 = stats.f1_vec(P[:, idx[sl]], G[idx[sl]])   # (F, chunk)
        boots[sl] = f1.max(axis=0) - f1.min(axis=0)
    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
    return point, [float(lo_), float(hi_)]


def ops_cell(cell: Cell, frameworks: list[str]) -> dict:
    """api_error_rate / cost / latency per framework, pooled over all runs
    in the cell. Cells at >= 10% error rate get the loud-warning flag
    (plan §4 error semantics)."""
    out = {"judge": cell.judge, "task": cell.task, "R": cell.R,
           "frameworks": {}}
    for fw in frameworks:
        o = dict(cell.ops[fw])
        o = {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in o.items()}
        o["error_rate_warning_ge_10pct"] = bool(o["api_error_rate"] >= 0.10)
        out["frameworks"][fw] = o
    return out
