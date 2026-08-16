"""Robustness sweep: is the P3 result an artifact of the 20% gate bar?

The preregistered gate is "fail if flagged rate > 20% on a batch of 50"
(gateflip.GATE_CUT = 10.5). That bar was chosen before any data was seen
and is not derived from anything; a reader is entitled to ask whether the
per-framework degeneracy P3 exhibits survives moving it.

This module does not touch the frozen P3 path. It re-draws the SAME 1,000
batches (identical seed derivation to gateflip.p3_cell) and re-thresholds
them at several bars, so the only quantity that moves between rows is the
bar itself. Monte-Carlo flip rates and per-framework gate-fail
probabilities only -- the closed-form jackknife H3 test is confirmatory
and stays at its preregistered bar.

    python -m study.analysis.sensitivity_gate --split test
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import stats
from .data import Cell
from .gateflip import BATCH, N_BATCHES, _code, _columns

# 20% is the preregistered bar; the rest bracket it symmetrically.
BARS = [0.10, 0.15, 0.20, 0.25, 0.30]


def _cut(bar: float) -> float:
    """Count strictly above ``bar * BATCH``, matching the frozen 10.5 for
    bar=0.20 (fail iff count > 10 of 50)."""
    return np.floor(bar * BATCH) + 0.5


def sweep_cell(cell: Cell, frameworks: list[str], gold: np.ndarray) -> dict:
    R = cell.R
    faithful = ~gold
    m = int(faithful.sum())
    cols, cross, within = _columns(frameworks, R)
    V = np.stack([cell.v["A"][fw][r][faithful]
                  for fw, r in cols]).astype(float)

    rng = np.random.default_rng([stats.SEED, 3, _code(cell)])
    idx = rng.integers(0, m, size=(N_BATCHES, BATCH))
    counts = V[:, idx].sum(axis=2)

    rows = []
    for bar in BARS:
        fail = counts > _cut(bar)

        def flip(pairs):
            if not pairs:
                return None
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            return round(float((fail[a] != fail[b]).mean()), 4)

        per_fw = {fw: round(float(fail[[i for i, (f, _) in enumerate(cols)
                                        if f == fw]].mean()), 4)
                  for fw in frameworks}
        rows.append({
            "bar": bar,
            "preregistered": bar == 0.20,
            "cut": _cut(bar),
            "gate_fail_prob_mc": per_fw,
            "n_frameworks_ever_failing": sum(p > 0.0 for p in per_fw.values()),
            "cross_flip_mc": flip(cross),
            "within_flip_mc": flip(within),
        })
    return {"judge": cell.judge, "task": cell.task, "R": R,
            "n_faithful": m, "batch": BATCH, "n_batches": N_BATCHES,
            "note": "same 1,000 batches at every bar; only the bar moves",
            "bars": rows}


def main() -> None:
    from . import data
    from .data import FRAMEWORKS

    # Mirrors run_all.KILLSWITCH; imported by value rather than by module
    # because run_all pulls in matplotlib, which this sweep does not need.
    KILLSWITCH = [f for f in FRAMEWORKS if f != "multivon-eval"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--out", default="study/analysis/out/sensitivity_gate.json")
    args = ap.parse_args()

    if args.split == "test":
        data.assert_test_unblind_allowed()

    out = {"split": args.split, "bars": BARS, "primary": [], "killswitch": []}
    for judge in data.JUDGES:
        for task in data.TASKS:
            cell = data.load_cell(judge, task, args.split)
            gold = data.load_labels(task, args.split, cell.ids)
            out["primary"].append(sweep_cell(cell, FRAMEWORKS, gold))
            out["killswitch"].append(sweep_cell(cell, KILLSWITCH, gold))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}: {len(out['primary'])} cells x {len(BARS)} bars")


if __name__ == "__main__":
    main()
