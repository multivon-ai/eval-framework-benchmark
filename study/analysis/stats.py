"""Statistics for the preregistered endpoints (plan §6/§9).

All CIs: item-cluster bootstrap — resample items, keep every framework/
judge/run observation of a resampled item (Miller 2024 framing). BCa for
Δ_fg (plan §9), percentile elsewhere. Deterministic: every consumer passes
a ``numpy.random.default_rng([SEED, ...])`` child stream.
"""
from __future__ import annotations

import itertools
from statistics import NormalDist

import numpy as np

SEED = 42          # plan §10: bootstrap seed
Z975 = 1.959963984540054
_ND = NormalDist()


def norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    return norm.cdf(x)


def norm_ppf(p: float) -> float:
    return _ND.inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


# ── agreement ────────────────────────────────────────────────────────────────

def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    po = float((a == b).mean())
    pi = (float(a.mean()) + float(b.mean())) / 2.0
    pe = 2.0 * pi * (1.0 - pi)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def _pair_kappas(V: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Pairwise kappa for rows of V (F, n) over given index pairs."""
    return np.array([cohen_kappa(V[f], V[g]) for f, g in pairs])


def median_kappa_ci(V: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> dict:
    """Confirmatory P1 statistic: median pairwise Cohen's κ over the rows of
    V (F, n) with a percentile item-cluster bootstrap CI."""
    F, n = V.shape
    pairs = list(itertools.combinations(range(F), 2))
    point = float(np.median(_pair_kappas(V, pairs)))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot)
    Vb = V.astype(float)
    for chunk in range(0, n_boot, 500):
        sl = slice(chunk, min(chunk + 500, n_boot))
        S = Vb[:, idx[sl]]                       # (F, chunk, n)
        ks = []
        for f, g in pairs:
            A, B = S[f], S[g]
            po = (A == B).mean(axis=1)
            pa, pb = A.mean(axis=1), B.mean(axis=1)
            pe = pa * pb + (1 - pa) * (1 - pb)
            with np.errstate(divide="ignore", invalid="ignore"):
                k = np.where(pe >= 1.0, 0.0, (po - pe) / (1 - pe))
            ks.append(k)
        boots[sl] = np.median(np.stack(ks), axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"median_kappa": point, "ci95": [float(lo), float(hi)],
            "n_pairs": len(pairs), "n_items": n, "n_boot": n_boot}


# ── bootstrap machinery ──────────────────────────────────────────────────────

def percentile_boot_mean(w: np.ndarray, n_boot: int,
                         rng: np.random.Generator) -> tuple[float, list[float]]:
    """Mean of per-item values w with percentile item-bootstrap CI."""
    n = len(w)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = w[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(w.mean()), [float(lo), float(hi)]


def bca_ci_mean(w: np.ndarray, n_boot: int,
                rng: np.random.Generator) -> tuple[float, list[float]]:
    """BCa CI for the mean of per-item values w (items are the clusters;
    the Δ_fg statistic is linear in per-item terms so mean() is exact)."""
    n = len(w)
    theta = float(w.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.sort(w[idx].mean(axis=1))
    # z0: median-bias correction; ties handled with the half-tie convention.
    prop = ((boots < theta).sum() + 0.5 * (boots == theta).sum()) / n_boot
    z0 = norm_ppf(prop)
    # acceleration from exact leave-one-out jackknife of the mean
    jack = (w.sum() - w) / (n - 1)
    d = jack.mean() - jack
    denom = (d ** 2).sum() ** 1.5
    a = float((d ** 3).sum() / (6.0 * denom)) if denom > 0 else 0.0
    out = []
    for zq in (-Z975, Z975):
        adj = z0 + (z0 + zq) / (1 - a * (z0 + zq))
        q = _ND.cdf(adj)
        out.append(float(np.quantile(boots, min(max(q, 0.0), 1.0))))
    return theta, out


def boot_stat_ci(per_item: dict[str, np.ndarray], stat, n_boot: int,
                 rng: np.random.Generator) -> tuple[float, list[float]]:
    """Percentile bootstrap for a statistic over aligned per-item arrays.
    ``stat`` maps {name: resampled (n,) or (B, n) arrays} -> float / (B,)."""
    n = len(next(iter(per_item.values())))
    point = float(stat({k: v for k, v in per_item.items()}))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.asarray(stat({k: v[idx] for k, v in per_item.items()}))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, [float(lo), float(hi)]


# ── classification metrics ───────────────────────────────────────────────────

def prf1(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def f1_vec(P: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Vectorized F1 over the last axis: P, G broadcastable (..., n) bool."""
    tp = (P & G).sum(axis=-1)
    fp = (P & ~G).sum(axis=-1)
    fn = (~P & G).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return f1


def sens_spec(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    sens = float(pred[gold].mean()) if gold.any() else 0.0
    spec = float((~pred[~gold]).mean()) if (~gold).any() else 0.0
    return sens, spec


def prevalence_standardized(sens: float, spec: float, prev: float) -> dict:
    """Precision/recall/F1 at a standardized prevalence (plan §3/§7 secondary
    sensitivity analysis at 0.10/0.25/0.50)."""
    flag_rate = prev * sens + (1 - prev) * (1 - spec)
    ppv = prev * sens / flag_rate if flag_rate > 0 else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens > 0 else 0.0
    return {"precision": ppv, "recall": sens, "f1": f1}


# ── McNemar (plan §9: Holm within each judge×task family) ───────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant counts."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[key] = running
    return adj


# ── bivariate normal (mirrors committed power_sim_endpoints.py) ─────────────

_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def phi2_lower(x: np.ndarray, y: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(X < x, Y < y), standard bivariate normal, correlation rho.
    Identical construction to study/power_sim_endpoints.py (the committed
    H3 test definition)."""
    x = np.clip(np.asarray(x, dtype=float), -8.0, 8.0)
    y = np.clip(np.asarray(y, dtype=float), -8.0, 8.0)
    rho = np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)
    x, y, rho = np.broadcast_arrays(x, y, rho)
    r = rho[..., None] * (_GL_X + 1.0) / 2.0
    one_m_r2 = 1.0 - r * r
    num = (x[..., None] ** 2 - 2.0 * r * x[..., None] * y[..., None]
           + y[..., None] ** 2)
    integrand = np.exp(-num / (2.0 * one_m_r2)) / np.sqrt(one_m_r2)
    integral = (rho / 2.0) * (integrand * _GL_W).sum(axis=-1)
    return norm_cdf(x) * norm_cdf(y) + integral / (2.0 * np.pi)
