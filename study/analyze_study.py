"""Guarded study analysis — plan §3 label blinding, §7 endpoints.

Decision note (Day 1): implemented as a NEW entry point rather than a flag
bolted onto analyze.py, because (a) analyze.py serves the published pilot
whose loaders return labels inline — retrofitting blinding there would
leave label-bearing code paths importable by the runner; (b) the study
raw-output layout (results/study/raw/<judge>/<fw>_<task>_run<i>.jsonl)
and item files differ from the pilot's. analyze.py is untouched.

Blinding contract:
  * ``data/labels_hidden/`` is read by exactly one function in the whole
    repo: ``load_hidden_labels`` below.
  * It REFUSES to run unless BOTH hold:
      1. ``git tag -l 'study-freeze-*'`` is non-empty (output freeze tagged), and
      2. ``study/FREEZE`` exists and lists the raw-output files covered by
         the freeze (every listed path must exist).
  * Without ``--unblind`` this script only computes label-free diagnostics
    (flag rates, pairwise inter-framework kappa, error rates) — these need
    no gold labels and are safe during the blinded phase.

Usage:
    python study/analyze_study.py --task ragtruth-sum                # blinded
    python study/analyze_study.py --task ragtruth-sum --unblind     # post-freeze
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analyze import _cohen_kappa, _precision_recall_f1  # noqa: E402

STUDY_DIR = REPO / "study"
LABELS_DIR = REPO / "data" / "labels_hidden"
FREEZE_FILE = STUDY_DIR / "FREEZE"

FRAMEWORKS = ["multivon-eval", "deepeval", "ragas", "trulens", "opik"]
TASKS = ("ragtruth-sum", "halueval-sum", "halueval-qa")


class BlindingError(RuntimeError):
    pass


def _freeze_tags() -> list[str]:
    out = subprocess.run(["git", "tag", "-l", "study-freeze-*"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return [t for t in out.stdout.splitlines() if t.strip()]


def load_hidden_labels(task: str) -> dict[str, bool]:
    """The ONLY sanctioned reader of data/labels_hidden/. Returns
    {item_id: is_hallucinated}. Raises BlindingError unless the output
    freeze is complete (git tag + FREEZE manifest)."""
    tags = _freeze_tags()
    if not tags:
        raise BlindingError(
            "unblind refused: no 'study-freeze-*' git tag exists. Freeze the raw "
            "outputs first (plan §3/§10 freeze order).")
    if not FREEZE_FILE.exists():
        raise BlindingError(
            "unblind refused: study/FREEZE manifest missing. It must list every "
            "raw-output file covered by the freeze tag.")
    listed = [ln.strip() for ln in FREEZE_FILE.read_text().splitlines()
              if ln.strip() and not ln.startswith("#")]
    if not listed:
        raise BlindingError("unblind refused: study/FREEZE lists no raw-output files.")
    missing = [f for f in listed if not (REPO / f).exists()]
    if missing:
        raise BlindingError(f"unblind refused: FREEZE-listed files missing: {missing}")
    labels = json.loads((LABELS_DIR / f"{task}_labels.json").read_text())
    return {cid: lab == "hallucinated" for cid, lab in labels.items()}


# ── Raw-output loading (label-free) ──────────────────────────────────────────

def load_verdicts(results_dir: Path, task: str, judge: str,
                  framework: str, run: int = 0) -> dict[str, bool]:
    path = results_dir / "raw" / judge / f"{framework}_{task}_run{run}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            out[row["case_id"]] = True  # error = CI failure = flagged (plan §4)
        else:
            out[row["case_id"]] = bool(row["flagged_hallucinated"])
    return out


def blinded_report(results_dir: Path, task: str, judge: str) -> dict:
    verdicts = {fw: load_verdicts(results_dir, task, judge, fw)
                for fw in FRAMEWORKS}
    verdicts = {fw: v for fw, v in verdicts.items() if v}
    rep: dict = {"task": task, "judge": judge, "frameworks": sorted(verdicts),
                 "flag_rates": {}, "pairwise_kappa": {}}
    for fw, v in verdicts.items():
        rep["flag_rates"][fw] = round(sum(v.values()) / len(v), 4)
    for a, b in combinations(sorted(verdicts), 2):
        common = sorted(set(verdicts[a]) & set(verdicts[b]))
        if not common:
            continue
        rep["pairwise_kappa"][f"{a} <-> {b}"] = round(_cohen_kappa(
            [verdicts[a][i] for i in common], [verdicts[b][i] for i in common]), 4)
    return rep


def unblinded_report(results_dir: Path, task: str, judge: str) -> dict:
    labels = load_hidden_labels(task)  # guarded
    rep = blinded_report(results_dir, task, judge)
    rep["vs_gold"] = {}
    for fw in rep["frameworks"]:
        v = load_verdicts(results_dir, task, judge, fw)
        ids = sorted(set(v) & set(labels))
        pred = [v[i] for i in ids]
        act = [labels[i] for i in ids]
        p, r, f1 = _precision_recall_f1(pred, act)
        rep["vs_gold"][fw] = {"n": len(ids), "precision": round(p, 4),
                              "recall": round(r, 4), "f1": round(f1, 4)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(REPO / "results" / "study"))
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--judge", action="append", default=None)
    ap.add_argument("--unblind", action="store_true",
                    help="Compute label-dependent metrics. Refuses unless a "
                         "study-freeze-* git tag AND a complete study/FREEZE "
                         "manifest exist.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    raw = results_dir / "raw"
    judges = args.judge or (sorted(p.name for p in raw.iterdir() if p.is_dir())
                            if raw.exists() else [])
    if not judges:
        print(f"No raw study outputs under {raw} yet — nothing to analyze.",
              file=sys.stderr)
        return 0
    tasks = list(TASKS) if args.task == "all" else [args.task]

    reports = []
    for judge in judges:
        for task in tasks:
            try:
                rep = (unblinded_report if args.unblind else blinded_report)(
                    results_dir, task, judge)
            except BlindingError as exc:
                print(f"BLINDING GUARD: {exc}", file=sys.stderr)
                return 2
            reports.append(rep)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
