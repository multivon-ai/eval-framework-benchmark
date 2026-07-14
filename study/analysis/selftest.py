"""Self-test for the repeated-cell code paths that dev data (R=1) cannot
exercise: P2 Δ/B−W, P3 flip difference + jackknife, κ_self/κ_cross,
pass@k/pass^k. Synthetic verdicts with a known generative structure;
checks are sanity bounds, not preregistered numbers.

Run:  .venv-analysis/bin/python -m study.analysis.selftest
"""
from __future__ import annotations

import numpy as np

from . import agreement, decomposition, gateflip
from .data import Cell, FRAMEWORKS


def synthetic_cell(R: int = 5, n: int = 300,
                   seed: int = 7) -> tuple[Cell, np.ndarray]:
    """Latent-threshold verdicts: shared item effect (cross-fw corr) plus a
    framework-specific effect (higher within-run corr), FPR/TPR spread so
    frameworks genuinely disagree — Δ should come out positive."""
    rng = np.random.default_rng(seed)
    gold = np.zeros(n, dtype=bool)
    gold[: n // 2] = True
    u = rng.standard_normal((1, n))
    cell = Cell(judge="synthetic", task="synthetic", split="selftest",
                ids=[f"it_{i:04d}" for i in range(n)], R=R,
                v={"A": {}, "B": {}})
    cuts = np.linspace(0.6, 1.6, len(FRAMEWORKS))
    for fw, cut in zip(FRAMEWORKS, cuts):
        w = rng.standard_normal((1, n))
        e = rng.standard_normal((R, n))
        z = 0.6 * u + 0.75 * w + 0.28 * e + 1.2 * gold[None, :]
        v = (z > cut).astype(np.int8)
        cell.v["A"][fw] = v
        cell.v["B"][fw] = v
        cell.err[fw] = np.zeros((R, n), dtype=bool)
    return cell, gold


def main() -> int:
    cell, gold = synthetic_cell()
    p2 = decomposition.p2_cell(cell, FRAMEWORKS, n_boot=1000)
    assert "skipped" not in p2, p2
    deltas = [d["Delta_fg"] for d in p2["pairs"].values()]
    assert all(np.isfinite(deltas)), deltas
    assert p2["h2_majority_positive"], deltas
    bw = p2["pooled_BW"]
    assert bw["B_minus_W"] > 0 and bw["W"] > 0, bw
    print(f"P2 ok: Δ range [{min(deltas):.3f}, {max(deltas):.3f}], "
          f"B−W={bw['B_minus_W']:.4f} {bw['B_minus_W_ci95']}")

    p3 = gateflip.p3_cell(cell, FRAMEWORKS, gold)
    d = p3["flip_difference"]
    assert "skipped" not in d, d
    assert abs(d["T_cross_minus_within"] - d["T_mc"]) < 0.05, d
    assert d["ci95"][0] < d["T_cross_minus_within"] < d["ci95"][1]
    print(f"P3 ok: T={d['T_cross_minus_within']:.4f} (MC {d['T_mc']:.4f}), "
          f"CI {d['ci95']}, cross={p3['cross_flip_mc']:.3f}, "
          f"within={p3['within_flip_mc']:.3f}")

    ks = agreement.kappa_self_cross(cell, FRAMEWORKS, n_boot=500,
                                    rng=np.random.default_rng(0))
    assert "skipped" not in ks
    assert min(ks["kappa_self"].values()) > max(ks["kappa_cross"].values())
    print(f"S2 ok: contrast={ks['contrast_min_self_minus_max_cross']:.4f} "
          f"{ks['contrast_ci95']}")

    pk = decomposition.passk_cell(cell, FRAMEWORKS, gold)
    assert "skipped" not in pk
    one = pk["frameworks"][FRAMEWORKS[0]]
    assert one["pass^k k=5"]["value"] <= one["pass@k k=1"]["value"] \
        <= one["pass@k k=5"]["value"]
    print(f"S7 ok: {FRAMEWORKS[0]} pass@1={one['pass@k k=1']['value']:.4f}, "
          f"pass^5={one['pass^k k=5']['value']:.4f}, "
          f"pass@5={one['pass@k k=5']['value']:.4f}")
    print("SELFTEST PASSED (repeated-cell paths exercised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
