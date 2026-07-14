"""P3 — false-positive-batch gate-flip rate (plan §7, H3).

Relabeled per addendum §12.5 (construction byte-unchanged): batches are
drawn only from the faithful-labeled half, so P3 estimates the flip
probability of a false-positive CONTROL batch under framework
substitution — not generic deployed release-gate behavior.

Gate: "fail the build if flagged rate > 20%" on batches of 50 items
bootstrap-resampled from the faithful-labeled half (Condition A verdicts,
errors-as-failures). 1,000 batches (preregistered) give the Monte-Carlo
flip-probability estimates; inference on T = p_cross − p_within uses the
committed H3 test from study/power_sim_endpoints.py: bivariate-normal
closed form for the batch-count gate probabilities with an exact
leave-one-item-out jackknife SE, |T|/SE > 1.96.

Cross pairs: same run, different frameworks. Within pairs: same framework,
different runs — hence R = 1 cells report cross-framework flips only and
SKIP the difference.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import stats
from .data import Cell

BATCH = 50
GATE_CUT = 10.5      # fail iff flagged count > 10 of 50 (rate > 20%)
N_BATCHES = 1000


def _columns(frameworks: list[str], R: int):
    cols = [(fw, r) for fw in frameworks for r in range(R)]
    fw_of = np.repeat(np.arange(len(frameworks)), R)
    cross = [(a, b) for a, b in itertools.combinations(range(len(cols)), 2)
             if fw_of[a] != fw_of[b] and a % R == b % R]
    within = [(a, b) for a, b in itertools.combinations(range(len(cols)), 2)
              if fw_of[a] == fw_of[b]]
    return cols, cross, within


def _flip_closed_form(mu, mab, pa, pb, n_cross):
    """T = p_cross − p_within from first moments (power_sim_endpoints
    flip_stat, verbatim math)."""
    var = np.maximum(mu - mu * mu, 1e-12)
    za = (BATCH * mu - GATE_CUT) / np.sqrt(BATCH * var)
    za = np.clip(za, -37.0, 37.0)
    mu_a, mu_b = mu[..., pa], mu[..., pb]
    cov = mab - mu_a * mu_b
    rho = cov / np.sqrt(np.maximum(var[..., pa] * var[..., pb], 1e-24))
    rho = np.clip(rho, -0.999, 0.999)
    p_a = stats.norm_cdf(za[..., pa])
    p_b = stats.norm_cdf(za[..., pb])
    both = stats.phi2_lower(za[..., pa], za[..., pb], rho)
    flip = p_a + p_b - 2.0 * both
    cross = flip[..., :n_cross].mean(axis=-1)
    if flip.shape[-1] == n_cross:
        return cross, None
    return cross, flip[..., n_cross:].mean(axis=-1)


def p3_cell(cell: Cell, frameworks: list[str], gold: np.ndarray) -> dict:
    R = cell.R
    faithful = ~gold
    m = int(faithful.sum())
    cols, cross, within = _columns(frameworks, R)
    V = np.stack([cell.v["A"][fw][r][faithful]
                  for fw, r in cols]).astype(float)          # (C, m)

    # Monte Carlo — the preregistered 1,000-batch simulation.
    rng = np.random.default_rng([stats.SEED, 3, _code(cell)])
    idx = rng.integers(0, m, size=(N_BATCHES, BATCH))
    counts = V[:, idx].sum(axis=2)                           # (C, N_BATCHES)
    fail = counts > GATE_CUT
    def flip_mc(pairs):
        if not pairs:
            return None
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        return float((fail[a] != fail[b]).mean())
    cross_mc, within_mc = flip_mc(cross), flip_mc(within)

    # Closed form + exact leave-one-item-out jackknife (committed H3 test).
    pairs = cross + within
    pa = np.array([p[0] for p in pairs])
    pb = np.array([p[1] for p in pairs])
    mu = V.mean(axis=1)
    VP = V[pa] * V[pb]                                       # (P, m)
    mab = VP.mean(axis=1)
    cross_cf, within_cf = _flip_closed_form(mu, mab, pa, pb, len(cross))

    out = {"judge": cell.judge, "task": cell.task, "R": R,
           "n_faithful": m, "batch": BATCH, "n_batches": N_BATCHES,
           "endpoint": "false-positive-batch gate-flip rate (addendum "
                       "§12.5 relabel; construction unchanged)",
           "gate": "fail if flagged rate > 20% (count > 10 of 50)",
           "gate_fail_prob_mc": {
               fw: round(float(fail[[i for i, (f, _) in enumerate(cols)
                                     if f == fw]].mean()), 4)
               for fw in frameworks},
           "cross_flip_mc": round(cross_mc, 4),
           "cross_flip_closed_form": round(float(cross_cf), 4)}
    if R < 2:
        out["within_flip"] = None
        out["flip_difference"] = {
            "skipped": f"[{cell.judge}/{cell.task}] within-framework flips "
                       f"need R >= 2 runs; R={R} ({cell.split} split) — "
                       f"confirmatory H3 difference not computable"}
        return out

    out["within_flip_mc"] = round(within_mc, 4)
    out["within_flip_closed_form"] = round(float(within_cf), 4)
    t_full = float(cross_cf - within_cf)

    # Jackknife over faithful items on the closed-form statistic.
    s1, sp = V.sum(axis=1), VP.sum(axis=1)
    mu_loo = (s1[None, :] - V.T) / (m - 1)                   # (m, C)
    mab_loo = (sp[None, :] - VP.T) / (m - 1)                 # (m, P)
    c_loo, w_loo = _flip_closed_form(mu_loo, mab_loo, pa, pb, len(cross))
    t_loo = c_loo - w_loo
    var_j = (m - 1) / m * ((t_loo - t_loo.mean()) ** 2).sum()
    se = float(np.sqrt(max(var_j, 1e-24)))
    out["flip_difference"] = {
        "T_cross_minus_within": round(t_full, 4),
        "T_mc": round(cross_mc - within_mc, 4),
        "se_jackknife": round(se, 4),
        "ci95": [round(t_full - stats.Z975 * se, 4),
                 round(t_full + stats.Z975 * se, 4)],
        "h3_ci_excludes_0": bool(abs(t_full) / se > stats.Z975),
    }
    return out


def _code(cell: Cell) -> int:
    from .data import JUDGES, TASKS
    if cell.judge not in JUDGES:        # synthetic selftest cells
        return 99
    return JUDGES.index(cell.judge) * 10 + TASKS.index(cell.task)
