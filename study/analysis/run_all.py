"""Entry point for the preregistered analysis (plan §7/§9; addendum §7/§10).

    .venv-analysis/bin/python -m study.analysis.run_all --split dev
    .venv-analysis/bin/python -m study.analysis.run_all --split test --full

--split dev  uses the tag-gated dev-label carve-out (analyze_study
             load_hidden_labels_dev); R=1 everywhere, so repeated-cell
             analyses (P2, κ_self/κ_cross, pass@k, H3 difference) SKIP
             with a printed reason.
--split test refuses unless the raw-output freeze is complete
             (study-freeze-* tag AND study/FREEZE manifest) — exactly the
             analyze_study --unblind gate.
Bootstrap resamples: 2,000 on dev for speed; --full (and always on test)
uses the preregistered 10,000.

Outputs: study/analysis/out/*.json, study/analysis/ANALYSIS.md,
study/analysis/figures/*.pdf. Byte-deterministic (fixed seeds; PDF
timestamps pinned via SOURCE_DATE_EPOCH).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDFs

if __package__ in (None, ""):                     # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "study.analysis"

import numpy as np  # noqa: E402

from . import agreement, data, decomposition, gateflip, performance  # noqa: E402
from . import figures, report, stats  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIGDIR = HERE / "figures"
KILLSWITCH = [f for f in data.FRAMEWORKS if f != "multivon-eval"]


def _rng(*codes: int) -> np.random.Generator:
    return np.random.default_rng([stats.SEED, *codes])


def _cellcode(judge: str, task: str) -> int:
    return data.JUDGES.index(judge) * 10 + data.TASKS.index(task)


def analyze(split: str, n_boot: int) -> dict:
    cells: dict[tuple[str, str], data.Cell] = {}
    gold: dict[str, np.ndarray] = {}
    for judge in data.JUDGES:
        for task in data.TASKS:
            cells[(judge, task)] = data.load_cell(judge, task, split)
    for task in data.TASKS:
        ids = cells[(data.JUDGES[0], task)].ids
        gold[task] = data.load_labels(task, split, ids)
        for judge in data.JUDGES[1:]:
            assert cells[(judge, task)].ids == ids, (judge, task)

    res: dict = {"meta": {
        "split": split, "n_boot": n_boot, "seed": stats.SEED,
        "frameworks": data.FRAMEWORKS, "judges": data.JUDGES,
        "tasks": data.TASKS,
        "conditions": {"A": "as-shipped default threshold",
                       "B": "locked dev-fitted threshold "
                            "(study/thresholds_locked.json) applied to scores"},
        "error_semantics": "errors-as-failures primary (errored record = "
                           "flagged); complete-case secondary (P4); "
                           "test-split policy: NO repair pass (addendum "
                           "§12.8)",
        "runs_per_cell": {f"{j}/{t}": cells[(j, t)].R
                          for j in data.JUDGES for t in data.TASKS},
    }}

    p1, p2, p3, p4 = [], [], [], []
    p5 = {"frameworks": KILLSWITCH, "p1": [], "p2": [], "p3": [], "p4": []}
    sec = {"mcnemar": [], "kappa_self_cross": [], "passk": [], "ops": []}

    for judge in data.JUDGES:
        for task in data.TASKS:
            cell = cells[(judge, task)]
            cc = _cellcode(judge, task)
            g = gold[task]
            for ci, cond in enumerate(data.CONDITIONS):
                p1.append(agreement.p1_cell(
                    cell, cond, data.FRAMEWORKS, n_boot, _rng(1, cc, ci, 0)))
                p5["p1"].append(agreement.p1_cell(
                    cell, cond, KILLSWITCH, n_boot, _rng(1, cc, ci, 1)))
            p2.append(_tag(decomposition.p2_cell(cell, data.FRAMEWORKS,
                                                 n_boot), cell))
            p5["p2"].append(_tag(decomposition.p2_cell(cell, KILLSWITCH,
                                                       n_boot), cell))
            p3.append(gateflip.p3_cell(cell, data.FRAMEWORKS, g))
            p5["p3"].append(gateflip.p3_cell(cell, KILLSWITCH, g))
            p4.append(performance.p4_cell(cell, data.FRAMEWORKS, g,
                                          n_boot, _rng(4, cc, 0)))
            p5["p4"].append(performance.p4_cell(cell, KILLSWITCH, g,
                                                n_boot, _rng(4, cc, 1)))
            sec["mcnemar"].append(agreement.mcnemar_family(
                cell, data.FRAMEWORKS))
            sec["kappa_self_cross"].append(_tag(agreement.kappa_self_cross(
                cell, data.FRAMEWORKS, n_boot, _rng(5, cc)), cell))
            sec["passk"].append(_tag(
                decomposition.passk_cell(cell, data.FRAMEWORKS, g), cell))
            sec["ops"].append(performance.ops_cell(cell, data.FRAMEWORKS))

    res.update({"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
                "secondary": sec})
    res["skipped"] = _collect_skips(res)
    return res, cells


def _tag(d: dict, cell: data.Cell) -> dict:
    if "skipped" in d:
        return {"judge": cell.judge, "task": cell.task, **d}
    return d


def _collect_skips(res: dict) -> list[str]:
    skips = []
    def walk(obj, path):
        if isinstance(obj, dict):
            if "skipped" in obj:
                label = path
                if "judge" in obj and "task" in obj:
                    label += f" [{obj['judge']}/{obj['task']}]"
                skips.append(f"{label}: {obj['skipped']}")
            for k, v in obj.items():
                if k != "skipped":
                    walk(v, f"{path}/{k}" if path else k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, path)
    for key in ("p2", "p3", "p5", "secondary"):
        walk(res[key], key.upper() if key.startswith("p") else key)
    return skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--full", action="store_true",
                    help="10,000 bootstrap resamples (preregistered; "
                         "always on for --split test)")
    args = ap.parse_args()

    if args.split == "test":
        try:
            data.assert_test_unblind_allowed()
        except data.BlindingError as exc:
            print(f"BLINDING GUARD: {exc}", file=sys.stderr)
            return 2
    n_boot = 10_000 if (args.full or args.split == "test") else 2_000

    res, cells = analyze(args.split, n_boot)

    OUT.mkdir(exist_ok=True)
    for name in ("p1", "p2", "p3", "p4", "p5", "secondary", "meta"):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps({"meta": res["meta"], name: res[name]}
                                   if name != "meta" else res["meta"],
                                   indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")

    FIGDIR.mkdir(exist_ok=True)
    written = figures.make_all(res, FIGDIR)
    for f in written:
        print(f"wrote {f}")

    md = HERE / "ANALYSIS.md"
    report.write_markdown(res, written, md)
    print(f"wrote {md}")

    for s in res["skipped"]:
        print(f"SKIPPED {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
