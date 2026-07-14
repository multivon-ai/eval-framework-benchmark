"""Cell-addressed loading of study raw outputs + guarded label access.

A *cell* is (judge, task, split). Within a cell every framework has R run
files study/runs/raw/{judge}/{task}_{split}/{framework}_run{r}.jsonl with
one record per item (identical item-id sets asserted).

Verdict conditions (plan §4/§7):
  A  as-shipped: recorded ``flagged_hallucinated`` at the framework default τ.
  B  locked τ from study/thresholds_locked.json applied to ``score``
     (flag rule per framework: opik ``score <= t``, others ``score < t``).
Errors-as-failures (primary): an errored record is flagged=True in both
conditions. Complete-case (secondary) drops errored records via ``err``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
STUDY = REPO / "study"
RAW = STUDY / "runs" / "raw"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(STUDY))

import analyze_study  # noqa: E402  (guarded label loaders + BlindingError)

BlindingError = analyze_study.BlindingError

FRAMEWORKS = ["multivon-eval", "deepeval", "ragas", "trulens", "opik"]
JUDGES = ["gpt-4o-mini", "claude-haiku-4-5"]
TASKS = ["ragtruth-sum", "halueval-sum", "halueval-qa"]
CONDITIONS = ("A", "B")

# Preregistered repeat design (plan §6, addendum §7.1): runs per cell.
RUNS = {
    "dev": {(t, j): 1 for t in TASKS for j in JUDGES},
    "test": {("ragtruth-sum", "gpt-4o-mini"): 5,
             ("ragtruth-sum", "claude-haiku-4-5"): 5,
             ("halueval-sum", "gpt-4o-mini"): 3,
             ("halueval-sum", "claude-haiku-4-5"): 1,
             ("halueval-qa", "gpt-4o-mini"): 3,
             ("halueval-qa", "claude-haiku-4-5"): 1},
}
N_ITEMS = {"dev": {t: 100 for t in TASKS},
           "test": {"ragtruth-sum": 500, "halueval-sum": 300,
                    "halueval-qa": 300}}

_THRESHOLDS = json.loads((STUDY / "thresholds_locked.json").read_text())


def locked_tau(framework: str, judge: str, task: str) -> float:
    return float(_THRESHOLDS[framework][judge][task]["threshold"])


def default_tau(framework: str, judge: str, task: str) -> float:
    return float(_THRESHOLDS[framework][judge][task]["default_threshold"])


def flag_b(framework: str, score: float, tau: float) -> bool:
    # opik stores score = 1 - native hallucination score with ties flagged
    # (thresholds_locked.json _meta.flag_rules).
    return score <= tau if framework == "opik" else score < tau


@dataclass
class Cell:
    judge: str
    task: str
    split: str
    ids: list[str]                      # sorted item ids, axis order everywhere
    R: int
    v: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # v[cond][fw]: (R, n) int8 verdicts, errors-as-failures
    err: dict[str, np.ndarray] = field(default_factory=dict)    # (R, n) bool
    score: dict[str, np.ndarray] = field(default_factory=dict)  # (R, n) float, nan=err
    ops: dict[str, dict] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)


def _read_run(judge: str, task: str, split: str, framework: str,
              run: int) -> dict[str, dict]:
    path = RAW / judge / f"{task}_{split}" / f"{framework}_run{run}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing raw cell file: {path}")
    recs = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        recs[row["item_id"]] = row
    return recs


def load_cell(judge: str, task: str, split: str,
              frameworks: list[str] | None = None) -> Cell:
    frameworks = frameworks or FRAMEWORKS
    R = RUNS[split][(task, judge)]
    per_fw_runs = {fw: [_read_run(judge, task, split, fw, r) for r in range(R)]
                   for fw in frameworks}

    ids = sorted(per_fw_runs[frameworks[0]][0])
    n_expect = N_ITEMS[split][task]
    if len(ids) != n_expect:
        raise ValueError(f"{judge}/{task}_{split}: {len(ids)} items, "
                         f"expected {n_expect}")
    for fw, runs in per_fw_runs.items():
        for r, recs in enumerate(runs):
            if sorted(recs) != ids:
                raise ValueError(
                    f"item-id mismatch: {judge}/{task}_{split}/{fw}_run{r}")

    cell = Cell(judge=judge, task=task, split=split, ids=ids, R=R,
                v={"A": {}, "B": {}})
    for fw, runs in per_fw_runs.items():
        tau_b = locked_tau(fw, judge, task)
        va = np.zeros((R, len(ids)), dtype=np.int8)
        vb = np.zeros((R, len(ids)), dtype=np.int8)
        er = np.zeros((R, len(ids)), dtype=bool)
        sc = np.full((R, len(ids)), np.nan)
        agg = {"n": 0, "n_err": 0, "cost_usd": 0.0, "latency_ms": 0.0,
               "judge_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        for r, recs in enumerate(runs):
            for i, iid in enumerate(ids):
                row = recs[iid]
                errored = bool(row.get("error"))
                er[r, i] = errored
                if errored:
                    va[r, i] = 1        # error = CI failure = flagged (plan §4)
                    vb[r, i] = 1
                else:
                    s = float(row["score"])
                    sc[r, i] = s
                    va[r, i] = int(bool(row["flagged_hallucinated"]))
                    vb[r, i] = int(flag_b(fw, s, tau_b))
                agg["n"] += 1
                agg["n_err"] += int(errored)
                for k in ("cost_usd", "latency_ms", "judge_calls",
                          "prompt_tokens", "completion_tokens"):
                    agg[k] += row.get(k) or 0
        cell.v["A"][fw] = va
        cell.v["B"][fw] = vb
        cell.err[fw] = er
        cell.score[fw] = sc
        cell.ops[fw] = {
            "n_evals": agg["n"], "n_errors": agg["n_err"],
            "api_error_rate": agg["n_err"] / agg["n"],
            "total_cost_usd": agg["cost_usd"],
            "mean_cost_usd": agg["cost_usd"] / agg["n"],
            "mean_latency_ms": agg["latency_ms"] / agg["n"],
            "mean_judge_calls": agg["judge_calls"] / agg["n"],
            "mean_prompt_tokens": agg["prompt_tokens"] / agg["n"],
            "mean_completion_tokens": agg["completion_tokens"] / agg["n"],
        }
    return cell


def load_labels(task: str, split: str, ids: list[str]) -> np.ndarray:
    """Gold labels as bool array aligned to ``ids`` — exclusively through
    analyze_study's guards. Raises BlindingError when the split's gate is
    not satisfied (test: study-freeze tag + FREEZE manifest)."""
    if split == "dev":
        labels = analyze_study.load_hidden_labels_dev(task)
        analyze_study.assert_dev_only(ids, task)
    else:
        labels = analyze_study.load_hidden_labels(task)
    missing = [i for i in ids if i not in labels]
    if missing:
        raise ValueError(f"{task}_{split}: {len(missing)} ids lack labels")
    return np.array([labels[i] for i in ids], dtype=bool)


def assert_test_unblind_allowed() -> None:
    """Refuse the whole test analysis unless the raw-output freeze gate
    passes — the same gate as analyze_study --unblind."""
    for task in TASKS:
        analyze_study.load_hidden_labels(task)
