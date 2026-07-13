"""Condition B threshold fit — plan §4, PREREG_ADDENDUM.md §12.

Preregistered algorithm (plan §4, COI controls, Condition B "symmetrically
tuned"): the ONLY fitted parameter is one scalar threshold per
framework x judge x task, selected deterministically on the dev split:

  * candidate set = all midpoints between consecutive unique dev scores
    plus both extremes of the score domain (0.0 and 1.0 — every adapter
    emits ``FrameworkResult.score`` in [0, 1]);
  * pick the candidate maximizing dev F1 (hallucinated = positive class,
    errors excluded from fitting — an errored record has no usable score;
    the count is disclosed per cell as ``n_errored``);
  * ties broken toward the STRICTER gate. Score direction (verified per
    adapter, see FLAG_RULES below): all five frameworks store a
    faithfulness-like score — LOW score = flagged hallucinated — so the
    stricter gate is the HIGHER threshold.

Gate semantics per framework (from the adapters, empirically confirmed
against every dev record):
  * multivon-eval, deepeval, ragas, trulens: flagged iff score <  t
  * opik:                                    flagged iff score <= t
    (opik stores score = 1 - native hallucination score; the adapter flags
    native halluc >= 0.5, i.e. faithfulness <= t with ties flagged)

Note on the extremes: under the ``score < t`` rule the extreme t=1.0 does
not flag items scoring exactly 1.0 — thresholds outside the framework's
legal [0, 1] config space are not candidates. t=0.0 flags nothing
(``<``) or only exact-zero scores (``<=``).

Dev labels come exclusively through the narrow, tag-gated carve-out
``analyze_study.load_hidden_labels_dev`` (dev ids only; any test id is
refused — ``assert_dev_only`` is called on the exact id set used).

Output: study/thresholds_locked.json —
  {framework: {judge: {task: {threshold, dev_f1, n_used, n_errored,
                              default_threshold, default_dev_f1,
                              flag_rule}}}}
The shipped defaults are carried alongside so Condition A needs no second
lookup. The file is byte-deterministic (no timestamps; sorted keys): run
the script twice and diff.

Usage:  .venv-study/bin/python study/fit_thresholds.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "study"))

from analyze import _precision_recall_f1  # noqa: E402
from analyze_study import assert_dev_only, load_hidden_labels_dev  # noqa: E402

STUDY_DIR = REPO / "study"
RAW = STUDY_DIR / "runs" / "raw"
OUT_PATH = STUDY_DIR / "thresholds_locked.json"

FRAMEWORKS = ("multivon-eval", "deepeval", "ragas", "trulens", "opik")
JUDGES = ("gpt-4o-mini", "claude-haiku-4-5")
TASKS = ("ragtruth-sum", "halueval-sum", "halueval-qa")

# Gate comparison per framework (see module docstring; verified against the
# adapters and every recorded dev verdict).
FLAG_RULES = {"multivon-eval": "score < t", "deepeval": "score < t",
              "ragas": "score < t", "trulens": "score < t",
              "opik": "score <= t"}


def _flag(framework: str, score: float, t: float) -> bool:
    return score <= t if framework == "opik" else score < t


def _load_dev_records(task: str, judge: str, framework: str) -> list[dict]:
    path = RAW / judge / f"{task}_dev" / f"{framework}_run0.jsonl"
    if not path.exists():
        raise SystemExit(f"ABORT: missing dev cell {path}")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(recs) != 100:
        raise SystemExit(f"ABORT: {path} has {len(recs)} records, expected 100")
    return recs


def _candidates(scores: list[float]) -> list[float]:
    uniq = sorted(set(scores))
    mids = [(a + b) / 2.0 for a, b in zip(uniq, uniq[1:])]
    return sorted(set([0.0, 1.0] + mids))


def fit_cell(task: str, judge: str, framework: str,
             labels: dict[str, bool]) -> dict:
    recs = _load_dev_records(task, judge, framework)
    used = [r for r in recs if not r.get("error")]
    errored = [r for r in recs if r.get("error")]
    assert_dev_only([r["item_id"] for r in used], task)

    scores = [float(r["score"]) for r in used]
    gold = [labels[r["item_id"]] for r in used]

    best_t, best_f1 = None, -1.0
    for t in _candidates(scores):
        pred = [_flag(framework, s, t) for s in scores]
        _, _, f1 = _precision_recall_f1(pred, gold)
        # strict > for F1; ties broken toward the HIGHER threshold (stricter
        # gate, low score = flagged). Candidates are scanned in ascending
        # order, so >= keeps the last (highest) F1-maximizing threshold.
        if f1 > best_f1 or (f1 == best_f1 and t > best_t):
            best_t, best_f1 = t, f1

    default_t = {r["threshold"] for r in recs}
    assert len(default_t) == 1, (task, judge, framework, default_t)
    default_t = default_t.pop()
    # Default-threshold dev F1 on the same fitting set (errors excluded),
    # from the recorded as-shipped verdicts.
    pred_default = [bool(r["flagged_hallucinated"]) for r in used]
    _, _, default_f1 = _precision_recall_f1(pred_default, gold)

    return {"threshold": round(best_t, 10), "dev_f1": round(best_f1, 4),
            "n_used": len(used), "n_errored": len(errored),
            "default_threshold": default_t,
            "default_dev_f1": round(default_f1, 4),
            "flag_rule": FLAG_RULES[framework]}


def main() -> int:
    out: dict = {"_meta": {
        "spec": "plan §4 Condition B: one scalar threshold per "
                "framework x judge x task, fitted on dev only; candidates = "
                "midpoints between consecutive unique dev scores + extremes "
                "0.0/1.0; max dev F1 (hallucinated=positive); ties -> higher "
                "(stricter) threshold; errored dev records excluded from "
                "fitting (disclosed as n_errored).",
        "dev_files": "study/runs/raw/{judge}/{task}_dev/{framework}_run0.jsonl",
        "labels_via": "analyze_study.load_hidden_labels_dev (tag-gated, "
                      "dev ids only)",
        "flag_rules": FLAG_RULES,
        "score_direction": "all frameworks: FrameworkResult.score in [0,1], "
                           "higher = more faithful; LOW = flagged",
    }}
    for fw in FRAMEWORKS:
        out[fw] = {}
        for judge in JUDGES:
            out[fw][judge] = {}
            for task in TASKS:
                labels = load_hidden_labels_dev(task)
                out[fw][judge][task] = fit_cell(task, judge, fw, labels)

    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"\n{'framework':<15}{'judge':<19}{'task':<14}"
          f"{'default':<9}{'fitted':<9}{'devF1@def':<11}{'devF1@fit':<11}"
          f"{'n_used':<7}n_err")
    for fw in FRAMEWORKS:
        for judge in JUDGES:
            for task in TASKS:
                c = out[fw][judge][task]
                print(f"{fw:<15}{judge:<19}{task:<14}"
                      f"{c['default_threshold']:<9.4g}{c['threshold']:<9.4g}"
                      f"{c['default_dev_f1']:<11.4f}{c['dev_f1']:<11.4f}"
                      f"{c['n_used']:<7}{c['n_errored']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
