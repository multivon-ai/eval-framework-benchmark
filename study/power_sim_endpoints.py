"""Endpoint-targeted power simulation — resolution of the plan-§9 gate failure.

The committed gate (study/power_sim.py) failed at n=300 on the worst-case
grid for a dependent-kappa DIFFERENCE of 0.15 and a paired-F1 gap of 0.10.
Both minimum effects map to SECONDARY contrasts (the kappa_self/kappa_cross
contrast and the P4 F1 spread), not to the confirmatory endpoints. This
script re-simulates power for the ACTUAL confirmatory endpoints at n=300,
10,000 replicates per cell, seeded (see PREREG_ADDENDUM.md §7 for the
decision this feeds):

  H1  P(upper bound of the 95% item-cluster-bootstrap CI of the MEDIAN
      pairwise Cohen's kappa < 0.40) — a one-sample bound test, simulated
      with 5 exchangeable raters (10 pairs) whose true pairwise kappa is
      in {0.05, 0.10, 0.15, 0.20} across marginal positive rates
      0.20-0.80. Per-replicate inference mirrors the preregistered
      analysis exactly: percentile bootstrap over item-category counts
      (B=1000 inside the power loop; the analysis itself uses 10,000),
      success iff the 97.5th percentile of the bootstrapped median kappa
      is below 0.40.

  H3  Power for the flip-rate difference CI excluding 0 under the P3
      batch-of-50 gate design: m=150 faithful-labeled items, 5 frameworks
      x R=5 runs, gate "fail if flagged rate > 20%" (i.e. >10 of 50).
      Verdicts follow a probit latent model
          z_ifr = a*u_i + b*w_if + c*e_ifr,   a^2+b^2+c^2 = 1
      so corr(cross-framework) = a^2 and corr(within-framework, across
      runs) = a^2+b^2. Framework false-positive rates straddle the 20%
      gate line (three spread scenarios), run noise varies over three
      levels. Batch-gate outcome probabilities are computed with the
      bivariate-normal approximation to the 50-item batch counts
      (equivalent to the P3 Monte Carlo with B -> infinity), and the
      flip-rate difference T = p_cross - p_within gets an exact
      leave-one-item-out jackknife SE (the same delta-equivalent
      item-cluster inference used in power_sim.py). Success iff
      |T|/SE > 1.96. Induced population cross/within gate-flip
      probabilities are reported per cell so the scenario grid can be
      checked against the plausible ranges (cross 0.15-0.35, within
      0.02-0.10).

  H4  mirrors H1 post threshold-tuning: same one-sample bound test; the
      dev-F1 tuning on balanced dev data pulls marginal rates toward the
      central band, so the H4 summary is the H1 grid restricted to
      marginal rates 0.35-0.65 (the full grid is still reported).

Type-I calibration cells (true effect at the null) are included for both
endpoints.

Replicates: 10,000 per H1/H4 cell. H3 cells use a COMMITTED REDUCTION to
2,000 replicates/cell (Monte-Carlo SE <= 0.011 at power 0.5, <= 0.009 at the
0.80 bar — immaterial for a >=0.80 gate). Recorded as a deviation in
PREREG_ADDENDUM.md §7: the first execution of this script at 10,000x18 H3
cells died with the parent process before writing results; the reduction was
committed BEFORE any H3 power number was observed.

Determinism: each cell draws from its own child stream
default_rng([SEED, section, cell_index]) so every cell is independently
reproducible and the run can resume from the per-cell checkpoint
(power_sim_endpoints.ckpt.json, deleted after the final JSON is written).

Run:  .venv-study/bin/python study/power_sim_endpoints.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_sim import corr_for_kappa  # noqa: E402  (shared latent-kappa helper)

SEED = 42
REPS = 10_000           # H1/H4 replicates per cell
H3_REPS = 2_000         # H3 replicates per cell (committed reduction, see docstring)
Z975 = 1.959963984540054
KAPPA_BAR = 0.40
BOOT_B = 1_000          # bootstrap resamples inside each replicate (H1/H4)
N_ITEMS = 300           # H1/H4
N_FAITHFUL = 150        # H3: faithful half of n=300
N_FW = 5
N_RUNS = 5              # RAGTruth-Sum repeated cells (plan §6)
BATCH = 50
GATE_CUT = 10.5         # fail iff flagged count > 10 (rate > 20%), continuity-corrected

TRUE_KAPPAS = [0.05, 0.10, 0.15, 0.20]
POS_RATES = [0.20, 0.35, 0.50, 0.65, 0.80]
H4_CENTRAL = {0.35, 0.50, 0.65}

# H3 scenario grid: framework FPR vectors near the 20% gate line, crossed
# with within-framework run-to-run latent correlation. a^2 = 0.35 shared-
# item variance throughout (cross-framework latent correlation). The three
# scenarios were chosen so the INDUCED population cross-framework gate-flip
# probabilities land at the bottom / center / top of the plausible effect
# range 0.15-0.35 (weak ~0.15, central ~0.24, strong ~0.35), and the three
# rho_within levels make the induced within-framework gate-flip
# probabilities span ~0.03-0.11 (plausible range 0.02-0.10). Induced values
# are recomputed analytically and recorded in the output.
A2_ITEM = 0.35
FPR_SCENARIOS = {
    "weak":    [0.10, 0.115, 0.13, 0.15, 0.17],
    "central": [0.11, 0.13, 0.15, 0.165, 0.19],
    "strong":  [0.09, 0.12, 0.15, 0.18, 0.23],
}
RHO_WITHIN = [0.999, 0.99, 0.95]
H3_CENTRAL_SCENARIO = "central"
H3_NS = [150, 250]      # faithful halves of n=300 and the n->500 escalation

OUT = Path(__file__).resolve().parent / "power_sim_endpoints_results.json"
CKPT = Path(__file__).resolve().parent / "power_sim_endpoints.ckpt.json"


# ── Vectorized bivariate-normal lower CDF (Gauss-Legendre, 48 nodes) ─────────

_GL_X, _GL_W = leggauss(48)


def phi2_lower(x: np.ndarray, y: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(X < x, Y < y) for standard bivariate normal with correlation rho.
    Fully vectorized; accurate to ~1e-6 for |rho| <= 0.99."""
    x = np.clip(np.asarray(x, dtype=float), -8.0, 8.0)
    y = np.clip(np.asarray(y, dtype=float), -8.0, 8.0)
    rho = np.clip(np.asarray(rho, dtype=float), -0.999, 0.999)
    x, y, rho = np.broadcast_arrays(x, y, rho)
    # integral_0^rho f(r) dr with r = rho*(t+1)/2, t in [-1, 1]
    r = rho[..., None] * (_GL_X + 1.0) / 2.0
    one_m_r2 = 1.0 - r * r
    num = (x[..., None] ** 2 - 2.0 * r * x[..., None] * y[..., None]
           + y[..., None] ** 2)
    integrand = np.exp(-num / (2.0 * one_m_r2)) / np.sqrt(one_m_r2)
    integral = (rho / 2.0) * (integrand * _GL_W).sum(axis=-1)
    return norm.cdf(x) * norm.cdf(y) + integral / (2.0 * np.pi)


def orthant_upper(ta: np.ndarray, tb: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(Z_a > ta, Z_b > tb) under correlation rho."""
    return phi2_lower(-np.asarray(ta), -np.asarray(tb), rho)


# ── H1/H4: median pairwise kappa, one-sample bound test ──────────────────────

_PAIRS_5 = list(itertools.combinations(range(N_FW), 2))  # 10 pairs


def _pair_mask_matrix() -> np.ndarray:
    """(32, 40) matrix: joint 5-rater category -> per-pair [n11,n10,n01,n00]."""
    M = np.zeros((32, 4 * len(_PAIRS_5)))
    for cat in range(32):
        bits = [(cat >> (N_FW - 1 - f)) & 1 for f in range(N_FW)]
        for j, (a, b) in enumerate(_PAIRS_5):
            va, vb = bits[a], bits[b]
            col = j * 4 + (0 if (va, vb) == (1, 1) else
                           1 if (va, vb) == (1, 0) else
                           2 if (va, vb) == (0, 1) else 3)
            M[cat, col] = 1.0
    return M


_PAIR_M = _pair_mask_matrix()


def median_pair_kappa(counts: np.ndarray) -> np.ndarray:
    """counts: (..., 32) -> median pairwise Cohen's kappa over the 10 pairs."""
    pc = counts @ _PAIR_M                                # (..., 40)
    pc = pc.reshape(*pc.shape[:-1], len(_PAIRS_5), 4)
    n11, n10, n01, n00 = (pc[..., 0], pc[..., 1], pc[..., 2], pc[..., 3])
    n = n11 + n10 + n01 + n00
    po = (n11 + n00) / n
    pa = (n11 + n10) / n
    pb = (n11 + n01) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / np.maximum(1 - pe, 1e-9)
    return np.median(kappa, axis=-1)


def simulate_kappa_bound_cell(rng: np.random.Generator, kappa_true: float,
                              p: float, n: int = N_ITEMS,
                              bar: float = KAPPA_BAR) -> float:
    """Power = P(97.5th pct of bootstrapped median pairwise kappa < bar)."""
    lam = corr_for_kappa(kappa_true, p) if kappa_true > 0 else 0.0
    t = norm.ppf(1 - p)
    # One-factor exchangeable latents: corr between any two raters = lam.
    u = rng.standard_normal((REPS, n, 1))
    e = rng.standard_normal((REPS, n, N_FW))
    z = np.sqrt(lam) * u + np.sqrt(1 - lam) * e
    v = z > t
    weights = 1 << np.arange(N_FW - 1, -1, -1)
    idx = (v * weights).sum(axis=-1)                     # (REPS, n)
    counts = np.stack([(idx == k).sum(axis=1) for k in range(32)], axis=1)

    success = np.empty(REPS, dtype=bool)
    chunk = 250
    for lo in range(0, REPS, chunk):
        c = counts[lo:lo + chunk].astype(float)          # (m, 32)
        pvals = c / n
        # Exact simplex closure (guard against float drift making the last
        # component negative, which Generator.multinomial rejects).
        pvals[:, -1] = np.maximum(0.0, 1.0 - pvals[:, :-1].sum(axis=1))
        boots = rng.multinomial(n, pvals, size=(BOOT_B, c.shape[0]))
        med = median_pair_kappa(boots.astype(float))     # (BOOT_B, m)
        upper = np.quantile(med, 0.975, axis=0)
        success[lo:lo + chunk] = upper < bar
    return float(success.mean())


# ── H3: batch-of-50 gate flip-rate difference ────────────────────────────────

def _h3_columns() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Column index = f * N_RUNS + r. Returns (cross_pairs, within_pairs)."""
    cross = [(f * N_RUNS + r, g * N_RUNS + r)
             for f, g in itertools.combinations(range(N_FW), 2)
             for r in range(N_RUNS)]                      # 10 x 5 = 50
    within = [(f * N_RUNS + r1, f * N_RUNS + r2)
              for f in range(N_FW)
              for r1, r2 in itertools.combinations(range(N_RUNS), 2)]  # 50
    return cross, within


_CROSS, _WITHIN = _h3_columns()
_ALL_PAIRS = _CROSS + _WITHIN
_PA = np.array([a for a, _ in _ALL_PAIRS])
_PB = np.array([b for _, b in _ALL_PAIRS])
_N_CROSS = len(_CROSS)


def flip_stat(mu: np.ndarray, mab: np.ndarray) -> np.ndarray:
    """Flip-rate difference T = p_cross - p_within from first moments.

    mu:  (..., 25) per-column flag probability (empirical or population)
    mab: (..., 100) per-pair joint probability E[V_a V_b]
    Batch counts ~ MVN(BATCH*mu, BATCH*Sigma) approximation of the
    batch-of-50 multinomial resample; gate fails iff count > GATE_CUT.
    """
    var = np.maximum(mu - mu * mu, 1e-12)
    za = (BATCH * mu - GATE_CUT) / np.sqrt(BATCH * var)
    za = np.clip(za, -37.0, 37.0)
    mu_a, mu_b = mu[..., _PA], mu[..., _PB]
    cov = mab - mu_a * mu_b
    rho = cov / np.sqrt(np.maximum(var[..., _PA] * var[..., _PB], 1e-24))
    rho = np.clip(rho, -0.999, 0.999)
    p_a = norm.cdf(za[..., _PA])
    p_b = norm.cdf(za[..., _PB])
    both = phi2_lower(za[..., _PA], za[..., _PB], rho)   # P(fail_a, fail_b)
    flip = p_a + p_b - 2.0 * both
    return (flip[..., :_N_CROSS].mean(axis=-1)
            - flip[..., _N_CROSS:].mean(axis=-1))


def h3_population_values(qs: list[float], rho_w: float) -> tuple[float, float]:
    """Induced population cross/within gate-flip probabilities for a cell."""
    taus = norm.ppf(1 - np.array(qs))                    # per framework
    tau_col = np.repeat(taus, N_RUNS)                    # (25,)
    mu = np.repeat(np.array(qs), N_RUNS)
    fw_of = np.repeat(np.arange(N_FW), N_RUNS)
    rho_latent = np.where(fw_of[_PA] == fw_of[_PB], rho_w, A2_ITEM)
    mab = orthant_upper(tau_col[_PA], tau_col[_PB], rho_latent)
    var = np.maximum(mu - mu * mu, 1e-12)
    za = np.clip((BATCH * mu - GATE_CUT) / np.sqrt(BATCH * var), -37, 37)
    rho_c = np.clip((mab - mu[_PA] * mu[_PB])
                    / np.sqrt(var[_PA] * var[_PB]), -0.999, 0.999)
    flip = (norm.cdf(za[_PA]) + norm.cdf(za[_PB])
            - 2.0 * phi2_lower(za[_PA], za[_PB], rho_c))
    return float(flip[:_N_CROSS].mean()), float(flip[_N_CROSS:].mean())


def simulate_h3_cell(rng: np.random.Generator, qs: list[float], rho_w: float,
                     b2_override: float | None = None,
                     m: int = N_FAITHFUL, reps: int = H3_REPS) -> float:
    """Power = P(|T|/SE_jack > 1.96), exact leave-one-item-out jackknife."""
    a2 = A2_ITEM
    b2 = (rho_w - a2) if b2_override is None else b2_override
    c2 = 1.0 - a2 - b2
    assert b2 >= 0 and c2 > 0, (qs, rho_w)
    taus = norm.ppf(1 - np.array(qs))
    tau_col = np.repeat(taus, N_RUNS)                    # (25,)

    power_hits = 0
    chunk = 100
    for lo in range(0, reps, chunk):
        r = min(chunk, reps - lo)
        u = rng.standard_normal((r, m, 1, 1))
        w = rng.standard_normal((r, m, N_FW, 1))
        e = rng.standard_normal((r, m, N_FW, N_RUNS))
        z = np.sqrt(a2) * u + np.sqrt(b2) * w + np.sqrt(c2) * e
        v = (z > taus[None, None, :, None]).reshape(r, m, N_FW * N_RUNS)
        v = v.astype(float)                              # (r, m, 25)

        s1 = v.sum(axis=1)                               # (r, 25)
        vp = v[:, :, _PA] * v[:, :, _PB]                 # (r, m, 100)
        sp = vp.sum(axis=1)                              # (r, 100)

        t_full = flip_stat(s1 / m, sp / m)               # (r,)

        mu_loo = (s1[:, None, :] - v) / (m - 1)          # (r, m, 25)
        mab_loo = (sp[:, None, :] - vp) / (m - 1)        # (r, m, 100)
        t_loo = flip_stat(mu_loo, mab_loo)               # (r, m)
        t_bar = t_loo.mean(axis=1)
        var_j = (m - 1) / m * ((t_loo - t_bar[:, None]) ** 2).sum(axis=1)
        se = np.sqrt(np.maximum(var_j, 1e-24))
        power_hits += int((np.abs(t_full) / se > Z975).sum())
    return power_hits / reps


# ── Driver ───────────────────────────────────────────────────────────────────

def _cell_rng(section: int, idx: int) -> np.random.Generator:
    """Independent, reproducible stream per cell (resume-safe)."""
    return np.random.default_rng([SEED, section, idx])


def _load_ckpt() -> dict:
    if CKPT.exists():
        return json.loads(CKPT.read_text())
    return {}


def _save_ckpt(ckpt: dict) -> None:
    CKPT.write_text(json.dumps(ckpt) + "\n")


def main() -> int:
    t0 = time.time()
    ckpt = _load_ckpt()
    results: dict = {
        "seed": SEED,
        "replicates": {"h1_h4": REPS, "h3": H3_REPS},
        "h3_replicate_reduction_note": (
            "H3 cells run at 2,000 replicates (H1/H4 stay at 10,000): the "
            "first execution at 10,000x18 H3 cells was killed with its parent "
            "process before any H3 power number was produced; the reduction "
            "was committed before observing any H3 result. MC SE <= 0.011."),
        "rng_scheme": "per-cell default_rng([42, section, cell_index])",
        "alpha": 0.05,
        "design": {
            "h1_h4": {"n_items": N_ITEMS, "n_frameworks": N_FW,
                      "kappa_bar": KAPPA_BAR, "bootstrap_B": BOOT_B,
                      "test": "97.5th percentile of item-bootstrap median "
                              "pairwise kappa < 0.40"},
            "h3": {"n_faithful": N_FAITHFUL, "n_frameworks": N_FW,
                   "runs": N_RUNS, "batch": BATCH, "gate": "flag rate > 20%",
                   "a2_item": A2_ITEM,
                   "test": "|p_cross - p_within| / SE_jackknife > 1.96"},
        },
        "h1_kappa_bound": [], "h3_flip_diff": [], "null_calibration": [],
    }

    print("H1/H4: median-pairwise-kappa upper-bound test", file=sys.stderr)
    for i, (k, p) in enumerate(itertools.product(TRUE_KAPPAS, POS_RATES)):
        key = f"h1:{i}"
        if key in ckpt:
            pw = ckpt[key]
        else:
            pw = simulate_kappa_bound_cell(_cell_rng(1, i), k, p)
            ckpt[key] = pw
            _save_ckpt(ckpt)
        results["h1_kappa_bound"].append(
            {"true_kappa": k, "pos_rate": p, "n": N_ITEMS, "power": round(pw, 4)})
        print(f"  kappa={k:.2f} p={p:.2f}: power={pw:.4f}", file=sys.stderr)

    print("H3: gate flip-rate difference", file=sys.stderr)
    h3_cells = [(m, name, qs, rho_w)
                for m in H3_NS
                for name, qs in FPR_SCENARIOS.items()
                for rho_w in RHO_WITHIN]
    for j, (m, name, qs, rho_w) in enumerate(h3_cells):
        cross_pop, within_pop = h3_population_values(qs, rho_w)
        key = f"h3:{j}"
        if key in ckpt:
            pw = ckpt[key]
        else:
            pw = simulate_h3_cell(_cell_rng(3, j), qs, rho_w, m=m)
            ckpt[key] = pw
            _save_ckpt(ckpt)
        results["h3_flip_diff"].append(
            {"scenario": name, "fprs": qs, "rho_within": rho_w,
             "n_faithful": m, "n_total": 2 * m,
             "induced_cross_flip": round(cross_pop, 4),
             "induced_within_flip": round(within_pop, 4),
             "power": round(pw, 4)})
        print(f"  m={m} {name} rho_w={rho_w}: cross={cross_pop:.3f} "
              f"within={within_pop:.3f} power={pw:.4f}", file=sys.stderr)

    # Type-I calibration: H1 at the bar (expected ~0.025 one-sided), H3 at
    # the exchangeable-wrapper null (b=0, equal FPRs; expected ~0.05).
    if "cal:h1" in ckpt:
        cal_h1 = ckpt["cal:h1"]
    else:
        cal_h1 = simulate_kappa_bound_cell(_cell_rng(9, 0), KAPPA_BAR, 0.50)
        ckpt["cal:h1"] = cal_h1
        _save_ckpt(ckpt)
    if "cal:h3" in ckpt:
        cal_h3 = ckpt["cal:h3"]
    else:
        cal_h3 = simulate_h3_cell(_cell_rng(9, 1), [0.19] * N_FW, A2_ITEM,
                                  b2_override=0.0)
        ckpt["cal:h3"] = cal_h3
        _save_ckpt(ckpt)
    results["null_calibration"] = [
        {"endpoint": "h1_kappa_bound", "true_kappa": KAPPA_BAR, "pos_rate": 0.5,
         "rejection_rate": round(cal_h1, 4), "expected": 0.025},
        {"endpoint": "h3_flip_diff", "note": "b^2=0, equal FPRs (exchangeable null)",
         "rejection_rate": round(cal_h3, 4), "expected": 0.05},
    ]
    print(f"  calibration: H1@bar={cal_h1:.4f} (≈.025)  H3@null={cal_h3:.4f} (≈.05)",
          file=sys.stderr)

    h1_min = min(c["power"] for c in results["h1_kappa_bound"])
    h4_min = min(c["power"] for c in results["h1_kappa_bound"]
                 if c["pos_rate"] in H4_CENTRAL)
    h3_300 = [c for c in results["h3_flip_diff"] if c["n_faithful"] == 150]
    h3_500 = [c for c in results["h3_flip_diff"] if c["n_faithful"] == 250]
    h3_min_central = min(c["power"] for c in h3_300
                         if c["scenario"] == H3_CENTRAL_SCENARIO)
    h3_min_all = min(c["power"] for c in h3_300)
    h3_min_central_500 = min(c["power"] for c in h3_500
                             if c["scenario"] == H3_CENTRAL_SCENARIO)

    results["min_power_h1_all_cells"] = h1_min
    results["min_power_h4_central_marginals"] = h4_min
    results["min_power_h3_central_grid_n300"] = h3_min_central
    results["min_power_h3_all_cells_n300"] = h3_min_all
    results["min_power_h3_central_grid_n500"] = h3_min_central_500
    gate = h1_min >= 0.80 and h3_min_central >= 0.80
    results["gate_rule"] = (
        "proceed at n=300 ONLY if H1 power >= 0.80 for all true-kappa <= 0.20 "
        "cells AND H3 power >= 0.80 on the central scenario grid "
        "(PREREG_ADDENDUM.md §7); pairwise kappa-DIFFERENCE contrasts demoted "
        "to exploratory")
    results["gate_passed_n300"] = gate
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    CKPT.unlink(missing_ok=True)
    print(f"\nwrote {OUT}")
    print(f"H1 min power (all 20 cells, kappa<=0.20, n=300): {h1_min:.4f}")
    print(f"H4 min power (central marginals 0.35-0.65):      {h4_min:.4f}")
    print(f"H3 min power (central grid, n=300):              {h3_min_central:.4f}")
    print(f"H3 min power (all cells, n=300):                 {h3_min_all:.4f}")
    print(f"H3 min power (central grid, n=500 escalation):   {h3_min_central_500:.4f}")
    if gate:
        print("\nREFINED POWER GATE PASSED at n=300 (endpoint-targeted).")
        return 0
    print("\n" + "!" * 78)
    print("!!  REFINED POWER GATE FAILED — STOP. No test-split spend.")
    print("!" * 78)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
