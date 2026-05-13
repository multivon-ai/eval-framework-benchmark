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


def _load_runs(results_dir: Path, framework: str, task: str) -> list[list[dict]]:
    runs = []
    paths = sorted(results_dir.glob(f"raw/{framework}_{task}_run*.jsonl"))
    for p in paths:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        runs.append(rows)
    return runs


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


def analyze_task(results_dir: Path, task: str, cases: list[Case]) -> dict:
    by_id = _cases_by_id(cases)
    label_by_id = {c.id: c.label == "hallucinated" for c in cases}
    out: dict = {"task": task, "n_cases": len(cases), "by_framework": {}}

    framework_verdicts: dict[str, dict[str, bool]] = {}

    for fw in FRAMEWORKS:
        runs = _load_runs(results_dir, fw, task)
        if not runs:
            out["by_framework"][fw] = {"status": "no-runs"}
            continue
        per_case = _aggregate_run(runs, by_id)
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

    # Inter-framework kappa on the verdict columns.
    kappa: dict[str, float] = {}
    for a, b in combinations(framework_verdicts.keys(), 2):
        common_ids = sorted(set(framework_verdicts[a]) & set(framework_verdicts[b]))
        if not common_ids:
            continue
        av = [framework_verdicts[a][i] for i in common_ids]
        bv = [framework_verdicts[b][i] for i in common_ids]
        kappa[f"{a} ↔ {b}"] = round(_cohen_kappa(av, bv), 4)
    out["pairwise_kappa"] = kappa
    return out


def _format_md(report: dict) -> str:
    rows = []
    for task in report["tasks"]:
        rows.append(f"## {task['task']} (n={task['n_cases']})")
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
        if task.get("pairwise_kappa"):
            rows.append("### Inter-framework verdict agreement (Cohen's κ)")
            rows.append("")
            for k, v in task["pairwise_kappa"].items():
                rows.append(f"- **{k}** — κ = {v}")
            rows.append("")
    return "\n".join(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/RESULTS.md")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--task", choices=["qa", "sum", "both"], default="both")
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    report = {"tasks": []}
    tasks = ["qa", "sum"] if args.task == "both" else [args.task]
    for t in tasks:
        cases = load_qa(n=args.n) if t == "qa" else load_sum(n=args.n)
        report["tasks"].append(analyze_task(results_dir, t, cases))

    md = "# Results\n\n" + _format_md(report) + "\n"
    Path(args.out).write_text(md)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
