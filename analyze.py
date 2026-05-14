"""
Read raw NDJSON results, compute the headline metrics, write
``results/RESULTS.md``.

Metrics, per (framework × task):
  * F1, precision, recall of "framework says hallucinated" vs human label,
    at the framework's default/calibrated threshold.
  * Mean score std across the configured ``runs`` repeated runs.
  * Flaky-case rate: cases where the framework's pass/fail verdict was
    not unanimous across runs.
  * Mean latency (ms) per case.

Per (task) we also report inter-framework agreement (Cohen's kappa) on
the binary verdict.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

from data.loader import Case, load_qa, load_sum

FRAMEWORKS = ["multivon-eval", "deepeval", "ragas"]


def _load_runs(results_dir: Path, framework: str, task: str,
               judge: str | None = None) -> list[list[dict]]:
    """Load all runs for (framework, task), optionally filtered by judge.

    v1 layout: ``results/raw/{framework}_{task}_run*.jsonl``
    v2 layout: ``results/raw/{judge}/{framework}_{task}_run*.jsonl``

    Both work — we glob the v2 layout when ``judge`` is set, else fall
    back to v1.
    """
    runs = []
    if judge is not None:
        paths = sorted(results_dir.glob(f"raw/{judge}/{framework}_{task}_run*.jsonl"))
    else:
        paths = sorted(results_dir.glob(f"raw/{framework}_{task}_run*.jsonl"))
        if not paths:
            # Try any judge subdirectory if no flat-layout files exist.
            paths = sorted(results_dir.glob(f"raw/*/{framework}_{task}_run*.jsonl"))
    for p in paths:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        runs.append(rows)
    return runs


def _detect_judges(results_dir: Path) -> list[str]:
    """Return the list of judge subdirectories present in results/raw."""
    raw = results_dir / "raw"
    if not raw.exists():
        return []
    return sorted(p.name for p in raw.iterdir() if p.is_dir())


def _cases_by_id(cases: list[Case]) -> dict[str, Case]:
    return {c.id: c for c in cases}


def _aggregate_run(runs: list[list[dict]], cases_by_id: dict[str, Case]) -> dict[str, dict]:
    """Per case_id: aggregated score (mean), flagged-hallucinated majority, per-run scores, errors."""
    per_case: dict[str, dict] = {}
    if not runs:
        return per_case
    for run in runs:
        for row in run:
            cid = row["case_id"]
            slot = per_case.setdefault(cid, {
                "case_id": cid,
                "scores": [],
                "flagged": [],
                "latencies": [],
                "errors": 0,
                "threshold": row.get("threshold"),
            })
            if row.get("error"):
                slot["errors"] += 1
                continue
            slot["scores"].append(row["score"])
            slot["flagged"].append(bool(row["flagged_hallucinated"]))
            slot["latencies"].append(row["latency_ms"])
    return per_case


def _precision_recall_f1(predicted_pos: list[bool], actual_pos: list[bool]) -> tuple[float, float, float]:
    tp = sum(1 for p, a in zip(predicted_pos, actual_pos) if p and a)
    fp = sum(1 for p, a in zip(predicted_pos, actual_pos) if p and not a)
    fn = sum(1 for p, a in zip(predicted_pos, actual_pos) if not p and a)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _cohen_kappa(a: list[bool], b: list[bool]) -> float:
    if not a:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    pa = sum(a) / len(a)
    pb = sum(b) / len(b)
    p_e = pa * pb + (1 - pa) * (1 - pb)
    if p_e >= 1.0:
        return 0.0
    return (agree - p_e) / (1 - p_e)


def analyze_task(results_dir: Path, task: str, cases: list[Case],
                 judge: str | None = None) -> dict:
    by_id = _cases_by_id(cases)
    label_by_id = {c.id: c.label == "hallucinated" for c in cases}
    out: dict = {"task": task, "n_cases": len(cases), "by_framework": {}}

    framework_verdicts: dict[str, dict[str, bool]] = {}

    for fw in FRAMEWORKS:
        runs = _load_runs(results_dir, fw, task, judge=judge)
        if not runs:
            out["by_framework"][fw] = {"status": "no-runs"}
            continue
        per_case = _aggregate_run(runs, by_id)
        if not per_case:
            out["by_framework"][fw] = {"status": "no-completed-cases", "runs": len(runs)}
            continue
        ids_in_order = [c.id for c in cases if c.id in per_case]

        actuals = [label_by_id[i] for i in ids_in_order]
        # Majority-flag = framework's verdict for the case.
        predicted = []
        score_stds = []
        flaky = 0
        latencies = []
        errors = 0
        for cid in ids_in_order:
            slot = per_case[cid]
            errors += slot["errors"]
            flagged = slot["flagged"]
            scores = slot["scores"]
            if not flagged:
                # Every run errored; treat as not-hallucinated and miss it.
                predicted.append(False)
                continue
            n_flag = sum(flagged)
            predicted.append(n_flag > len(flagged) / 2)
            if 0 < n_flag < len(flagged):
                flaky += 1
            if len(scores) >= 2:
                score_stds.append(statistics.pstdev(scores))
            if slot["latencies"]:
                latencies.append(statistics.median(slot["latencies"]))

        framework_verdicts[fw] = dict(zip(ids_in_order, predicted))

        precision, recall, f1 = _precision_recall_f1(predicted, actuals)
        out["by_framework"][fw] = {
            "status": "ok",
            "threshold": next(iter(per_case.values())).get("threshold"),
            "runs": len(runs),
            "n_complete": len(ids_in_order),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_score_std_across_runs": round(statistics.fmean(score_stds), 4) if score_stds else None,
            "flaky_case_rate": round(flaky / len(ids_in_order), 4) if ids_in_order else None,
            "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
            "errors": errors,
        }

    # Inter-framework kappa AND raw disagreement count on the verdict columns.
    kappa: dict[str, float] = {}
    disagree_pct: dict[str, str] = {}
    for a, b in combinations(framework_verdicts.keys(), 2):
        common_ids = sorted(set(framework_verdicts[a]) & set(framework_verdicts[b]))
        if not common_ids:
            continue
        av = [framework_verdicts[a][i] for i in common_ids]
        bv = [framework_verdicts[b][i] for i in common_ids]
        kappa[f"{a} ↔ {b}"] = round(_cohen_kappa(av, bv), 4)
        n_disagree = sum(1 for x, y in zip(av, bv) if x != y)
        disagree_pct[f"{a} ↔ {b}"] = f"{n_disagree}/{len(common_ids)} ({100*n_disagree/len(common_ids):.0f}%)"
    out["pairwise_kappa"] = kappa
    out["pairwise_disagreement"] = disagree_pct

    # Threshold sweep per framework — compute F1/P/R at a fixed set of thresholds
    # so the blog post / RESULTS.md can show all three at apples-to-apples cutoffs.
    out["threshold_sweep"] = _threshold_sweep(results_dir, task, label_by_id, judge=judge)
    return out


def _threshold_sweep(results_dir: Path, task: str, label_by_id: dict[str, bool],
                     judge: str | None = None) -> dict:
    """For each framework, compute F1/P/R at thresholds 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95.

    Uses the mean score across all available runs as the per-case score.
    A case is flagged hallucinated when ``mean_score < threshold``.
    """
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sweep: dict[str, dict] = {}
    for fw in FRAMEWORKS:
        # Collect per-case mean scores across all runs.
        scores: dict[str, list[float]] = {}
        if judge is not None:
            globpat = f"raw/{judge}/{fw}_{task}_run*.jsonl"
        else:
            globpat = f"raw/{fw}_{task}_run*.jsonl"
            if not list(results_dir.glob(globpat)):
                globpat = f"raw/*/{fw}_{task}_run*.jsonl"
        for p in sorted(results_dir.glob(globpat)):
            for line in p.read_text().splitlines():
                row = json.loads(line)
                if row.get("error"):
                    continue
                scores.setdefault(row["case_id"], []).append(row["score"])
        if not scores:
            continue
        mean_score = {cid: statistics.fmean(s) for cid, s in scores.items()}
        ids = sorted(mean_score.keys() & label_by_id.keys())
        actl = [label_by_id[i] for i in ids]
        fw_sweep = []
        for t in thresholds:
            pred = [mean_score[i] < t for i in ids]
            precision, recall, f1 = _precision_recall_f1(pred, actl)
            fw_sweep.append({
                "threshold": t, "f1": round(f1, 4),
                "precision": round(precision, 4), "recall": round(recall, 4),
            })
        sweep[fw] = fw_sweep
    return sweep


def _format_md(report: dict) -> str:
    rows = []
    for task in report["tasks"]:
        judge = task.get("judge") or ""
        suffix = f"  ·  judge: `{judge}`" if judge and judge != "(flat layout)" else ""
        rows.append(f"## {task['task']} (n={task['n_cases']}){suffix}")
        rows.append("")
        rows.append("| Framework | Threshold | F1 | Precision | Recall | Score std (cross-run) | Flaky case rate | Median latency (ms) | Errors |")
        rows.append("|---|---|---|---|---|---|---|---|---|")
        for fw in FRAMEWORKS:
            f = task["by_framework"].get(fw, {})
            if f.get("status") != "ok":
                rows.append(f"| {fw} | — | _no runs_ | — | — | — | — | — | — |")
                continue
            rows.append(
                f"| {fw} "
                f"| {f.get('threshold', '—')} "
                f"| {f.get('f1', '—')} "
                f"| {f.get('precision', '—')} "
                f"| {f.get('recall', '—')} "
                f"| {f.get('mean_score_std_across_runs', '—')} "
                f"| {f.get('flaky_case_rate', '—')} "
                f"| {f.get('median_latency_ms', '—')} "
                f"| {f.get('errors', '—')} |"
            )
        rows.append("")
        if task.get("pairwise_disagreement"):
            rows.append("### Inter-framework verdict disagreement")
            rows.append("")
            rows.append("| Pair | Cases flipped | Cohen's κ |")
            rows.append("|---|---|---|")
            for k in task["pairwise_disagreement"]:
                rows.append(f"| {k} | {task['pairwise_disagreement'][k]} | {task['pairwise_kappa'][k]} |")
            rows.append("")
        sweep = task.get("threshold_sweep", {})
        if sweep:
            rows.append("### Threshold sweep")
            rows.append("")
            rows.append("F1, precision, recall for each framework over a fixed set of thresholds. A case is flagged hallucinated when its mean score across runs falls below the threshold.")
            rows.append("")
            for fw, rowset in sweep.items():
                rows.append(f"**{fw}**")
                rows.append("")
                rows.append("| Threshold | F1 | Precision | Recall |")
                rows.append("|---|---|---|---|")
                for r in rowset:
                    rows.append(f"| {r['threshold']:.2f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")
                rows.append("")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/RESULTS.md")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--task", choices=["qa", "sum", "ragtruth-sum", "both"],
                   default="both")
    p.add_argument("--judges", nargs="*", default=None,
                   help="Restrict to specific judge subdirectories. "
                        "Default: auto-detect all judges in results/raw/.")
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    judges = args.judges if args.judges else _detect_judges(results_dir) or [None]
    report: dict = {"judges": [], "tasks": []}

    if args.task == "both":
        tasks = ["qa", "sum"]
    else:
        tasks = [args.task]

    for judge in judges:
        for t in tasks:
            if t == "ragtruth-sum":
                from data.ragtruth_loader import load_ragtruth_summary
                cases = load_ragtruth_summary(n=args.n)
            elif t == "qa":
                cases = load_qa(n=args.n)
            else:
                cases = load_sum(n=args.n)
            task_report = analyze_task(results_dir, t, cases, judge=judge)
            task_report["judge"] = judge or "(flat layout)"
            report["tasks"].append(task_report)
        if judge:
            report["judges"].append(judge)

    md = "# Results\n\n" + _format_md(report) + "\n"
    Path(args.out).write_text(md)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
