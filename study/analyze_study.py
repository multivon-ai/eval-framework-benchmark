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

Dev-split carve-out (``--unblind-dev`` / ``load_hidden_labels_dev``,
PREREG_ADDENDUM.md §12): plan §4 Condition B *requires* fitting one scalar
threshold per framework×judge×task on the dev split before any test run —
i.e. dev labels must be readable after the prereg freeze but before the
raw-output freeze. The carve-out is narrow by construction:
  * gated on the ``study-freeze-*`` tag ONLY (the prereg freeze, which
    exists before dev fitting), NOT on the ``study/FREEZE`` raw-output
    manifest — which continues to gate ``--unblind`` (test labels);
  * returns labels for exactly the 100 ``{task}_dev`` item ids (loaded via
    the blinded ``sample_items.load_study_items``) and nothing else;
  * hard-refuses if any returned id appears in the test split, and exposes
    ``assert_dev_only`` so downstream consumers (fit_thresholds.py) can
    refuse any test id before joining scores to labels.

Usage:
    python study/analyze_study.py --task ragtruth-sum                # blinded
    python study/analyze_study.py --task ragtruth-sum --unblind-dev # dev only
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
sys.path.insert(0, str(REPO / "study"))

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


def _split_ids(task: str, split: str) -> set[str]:
    from sample_items import load_study_items  # blinded loader (label="hidden")
    return {c.id for c in load_study_items(task, split)}


def load_hidden_labels_dev(task: str) -> dict[str, bool]:
    """Narrowly-scoped dev-split unblinding for the preregistered Condition B
    threshold fit (plan §4; PREREG_ADDENDUM.md §12).

    Gate: requires a ``study-freeze-*`` git tag (the prereg freeze) — and
    ONLY that; the ``study/FREEZE`` raw-output manifest continues to gate
    full unblinding (test labels) via ``load_hidden_labels``.

    Scope: returns {item_id: is_hallucinated} for exactly the ``{task}_dev``
    ids. Raises BlindingError if the label file would contribute any test-
    split id or if any dev id lacks a label."""
    tags = _freeze_tags()
    if not tags:
        raise BlindingError(
            "unblind-dev refused: no 'study-freeze-*' git tag exists. The "
            "prereg freeze must precede dev-threshold fitting (plan §4).")
    dev_ids = _split_ids(task, "dev")
    test_ids = _split_ids(task, "test")
    labels = json.loads((LABELS_DIR / f"{task}_labels.json").read_text())
    out = {cid: lab == "hallucinated" for cid, lab in labels.items()
           if cid in dev_ids}
    leaked = set(out) & test_ids
    if leaked:
        raise BlindingError(
            f"unblind-dev refused: {len(leaked)} id(s) belong to the test "
            f"split (dev/test overlap should be impossible): "
            f"{sorted(leaked)[:5]}")
    missing = dev_ids - set(out)
    if missing:
        raise BlindingError(
            f"unblind-dev refused: {len(missing)} dev id(s) have no label: "
            f"{sorted(missing)[:5]}")
    return out


def assert_dev_only(item_ids, task: str) -> None:
    """Refuse any id that is not a dev id for ``task`` — in particular every
    test-split id. Downstream consumers of dev labels MUST call this on the
    exact id set they are about to join to labels."""
    dev_ids = _split_ids(task, "dev")
    bad = sorted(set(item_ids) - dev_ids)
    if bad:
        test_ids = _split_ids(task, "test")
        kind = "TEST-SPLIT" if set(bad) & test_ids else "unknown"
        raise BlindingError(
            f"refused: {len(bad)} non-dev id(s) ({kind}) requested against "
            f"dev labels for {task}: {bad[:5]}")


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


STUDY_RUNS_RAW = STUDY_DIR / "runs" / "raw"


def load_dev_run_verdicts(task: str, judge: str, framework: str,
                          run: int = 0) -> dict[str, bool]:
    """Verdicts from the study runner's dev layout
    (study/runs/raw/{judge}/{task}_dev/{framework}_run{run}.jsonl).
    error != null => flagged (errors-as-failures, plan §4/§7 P4)."""
    path = STUDY_RUNS_RAW / judge / f"{task}_dev" / f"{framework}_run{run}.jsonl"
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["item_id"]] = (True if row.get("error")
                               else bool(row["flagged_hallucinated"]))
    return out


def dev_unblinded_report(task: str, judge: str) -> dict:
    """Dev-split-only vs-gold metrics through the narrow carve-out."""
    labels = load_hidden_labels_dev(task)  # tag-gated, dev ids only
    rep: dict = {"task": task, "split": "dev", "judge": judge, "vs_gold": {}}
    for fw in FRAMEWORKS:
        v = load_dev_run_verdicts(task, judge, fw)
        if not v:
            continue
        assert_dev_only(v.keys(), task)
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
    ap.add_argument("--unblind-dev", action="store_true",
                    help="Dev-split-only vs-gold metrics (plan §4 Condition B "
                         "fit support). Requires only the study-freeze-* tag; "
                         "loads ONLY {task}_dev labels and refuses any test "
                         "id. Reads the study runner layout study/runs/raw/.")
    args = ap.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]

    if args.unblind_dev:
        judges = args.judge or (
            sorted(p.name for p in STUDY_RUNS_RAW.iterdir() if p.is_dir())
            if STUDY_RUNS_RAW.exists() else [])
        if not judges:
            print(f"No dev outputs under {STUDY_RUNS_RAW} — nothing to "
                  f"analyze.", file=sys.stderr)
            return 0
        reports = []
        for judge in judges:
            for task in tasks:
                try:
                    reports.append(dev_unblinded_report(task, judge))
                except BlindingError as exc:
                    print(f"BLINDING GUARD: {exc}", file=sys.stderr)
                    return 2
        print(json.dumps(reports, indent=2))
        return 0

    results_dir = Path(args.results_dir)
    raw = results_dir / "raw"
    judges = args.judge or (sorted(p.name for p in raw.iterdir() if p.is_dir())
                            if raw.exists() else [])
    if not judges:
        print(f"No raw study outputs under {raw} yet — nothing to analyze.",
              file=sys.stderr)
        return 0

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
