"""Preregistered study item sampler — plan §3 (research-plan-cross-framework-disagreement).

Draws the dev split FIRST (seed 20260714), then the test split (seed
20260713) from the remaining pool, per task:

    ragtruth-sum   500 test (250/250; §7 escalation — prefix-stable
                   deterministic extension of the original 300) + 100 dev (50/50)
    halueval-sum   300 test (150/150) + 100 dev (50/50)
    halueval-qa    300 test (150/150) + 100 dev (50/50)

Blinding: the committed item files (``study/items/``) carry ids + inputs
only. Gold labels are written exclusively to ``data/labels_hidden/`` and
must never be read by runner code paths (see study/analyze_study.py for
the guarded unblind procedure).

Anti-leak id rule: the upstream loaders embed the label in the case id
(``sum_00106_faithful``). Study ids strip the label suffix. This is safe
because each source unit contributes exactly one variant to the whole
study (units are disjoint within and across splits), so the stripped id
is still unique.

Sampling units (documented deviation-refinement, PREREG_ADDENDUM.md §4):
  * HaluEval: the unit is the source row. Each sampled row contributes
    exactly one variant (faithful or hallucinated), and rows are disjoint
    within and across splits — no two study items share a source document.
  * RAGTruth: the unit is the source document. One response per source,
    sources disjoint within and across splits. Pool restricted to the
    'train' split (the committed pilots exhausted the 'test' split's
    hallucinated responses; see excluded_ids.json rules).

Usage:
    python study/sample_items.py --build-exclusions   # step 1, commit first
    python study/sample_items.py --sample             # step 2
    python study/sample_items.py --verify             # determinism + audits
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.loader import (  # noqa: E402
    QA_FULL, SUM_FULL, QA_URL, SUM_URL, Case,
    _download_if_missing, _expand_qa, _expand_sum, _read_jsonl,
)
from data.ragtruth_loader import (  # noqa: E402
    RESPONSES_FULL, RESPONSES_URL, SOURCES_FULL, SOURCES_URL, _source_text,
)

STUDY_DIR = REPO / "study"
ITEMS_DIR = STUDY_DIR / "items"
LABELS_DIR = REPO / "data" / "labels_hidden"
EXCLUDED_PATH = STUDY_DIR / "excluded_ids.json"

SEED_DEV = 20260714   # drawn FIRST (plan §3)
SEED_TEST = 20260713
N_TEST, N_DEV = 300, 100

# §7 escalation (PREREG_ADDENDUM.md, 2026-07-13): RAGTruth-Sum test split
# escalated 300 → 500 (250/250) per the plan's sole preregistered escalation.
# The extension is PREFIX-STABLE: the original 300-item draw is reproduced
# byte-for-byte (same seed 20260713, same draw order), then the SAME rng
# object continues to draw 100 more hallucinated + 100 more faithful source
# units from the remaining pool. The committed 300-item file is therefore
# exactly the first 300 items of the 500-item file (verified in --verify).
# Dev split unchanged; HaluEval tasks stay at n=300.
N_TEST_RAGTRUTH = 500


def n_test_for(task: str) -> int:
    return N_TEST_RAGTRUTH if task == "ragtruth-sum" else N_TEST

# multivon-eval calibration used [:n] prefixes of the raw HaluEval files
# (benchmarks/run_threshold_calibration.py, run_calibration_v2.py,
# run_truly_held_out.py). Max prefix observed: 50 rows (QA), 30 rows (Sum).
# We exclude the first 100 rows of both files as a conservative superset.
CALIBRATION_PREFIX_ROWS = 100

TASKS = ("ragtruth-sum", "halueval-sum", "halueval-qa")


# ── Exclusions ────────────────────────────────────────────────────────────────

def build_exclusions() -> dict:
    pilot_ids: dict[str, list[str]] = {"halueval_qa": [], "halueval_sum": [], "ragtruth_sum": []}
    for path in sorted(glob.glob(str(REPO / "data" / "*pilot*.json"))):
        for row in json.loads(Path(path).read_text()):
            cid = row["id"]
            if cid.startswith("ragtruth_sum_"):
                pilot_ids["ragtruth_sum"].append(cid)
            elif cid.startswith("qa_"):
                pilot_ids["halueval_qa"].append(cid)
            elif cid.startswith("sum_"):
                pilot_ids["halueval_sum"].append(cid)
            else:
                raise ValueError(f"Unrecognized pilot id {cid!r} in {path}")

    def _rows(ids: list[str]) -> list[int]:
        return sorted({int(cid.split("_")[1]) for cid in ids})

    qa_rows = sorted(set(_rows(pilot_ids["halueval_qa"])) | set(range(CALIBRATION_PREFIX_ROWS)))
    sum_rows = sorted(set(_rows(pilot_ids["halueval_sum"])) | set(range(CALIBRATION_PREFIX_ROWS)))
    rt_resp_ids = sorted({cid.rsplit("_", 1)[0].replace("ragtruth_sum_", "")
                          for cid in pilot_ids["ragtruth_sum"]}, key=int)

    return {
        "built": "2026-07-13",
        "rules": {
            "pilot": "Every item id appearing in any committed data/*pilot*.json file "
                     "is excluded; for HaluEval the whole source row is excluded (both "
                     "variants) to avoid paired-counterpart leakage.",
            "calibration": f"multivon-eval threshold calibration loads [:n] prefixes of "
                           f"the raw HaluEval files (max observed n: 50 QA rows / 30 Sum "
                           f"rows). Conservative superset: source rows 0..{CALIBRATION_PREFIX_ROWS - 1} "
                           f"of both halueval_qa_full.json and halueval_sum_full.json are excluded.",
            "ragtruth": "No RAGTruth item was ever used in multivon-eval calibration "
                        "(verified by repo grep — RAGTruth appears only in a roadmap "
                        "note). The committed RAGTruth pilots were drawn from the "
                        "'test' split, which retains only 104 unseen hallucinated "
                        "responses — fewer than the 200 the study needs — so the study "
                        "pool is the 'train' split (disjoint from 'test' at the "
                        "source-document level; the train/test names refer to RAGTruth's "
                        "original model-training protocol, not ours). Pilot response ids "
                        "listed below are additionally excluded as a belt-and-braces rule.",
        },
        "halueval_qa": {
            "pilot_case_ids": sorted(set(pilot_ids["halueval_qa"])),
            "excluded_source_rows": qa_rows,
        },
        "halueval_sum": {
            "pilot_case_ids": sorted(set(pilot_ids["halueval_sum"])),
            "excluded_source_rows": sum_rows,
        },
        "ragtruth_sum": {
            "pilot_case_ids": sorted(set(pilot_ids["ragtruth_sum"])),
            "excluded_response_ids": rt_resp_ids,
            "excluded_splits": ["test"],
        },
    }


# ── HaluEval sampling ────────────────────────────────────────────────────────

def _sample_halueval(task: str, excl: dict) -> tuple[dict, dict, dict]:
    import random
    if task == "halueval-qa":
        _download_if_missing(QA_URL, QA_FULL)
        cases = _expand_qa(_read_jsonl(QA_FULL))
        key, prefix = "halueval_qa", "qa"
    else:
        _download_if_missing(SUM_URL, SUM_FULL)
        cases = _expand_sum(_read_jsonl(SUM_FULL))
        key, prefix = "halueval_sum", "sum"

    excluded_rows = set(excl[key]["excluded_source_rows"])
    by_row: dict[int, dict[str, Case]] = {}
    for c in cases:
        by_row.setdefault(c.source_index, {})[c.label] = c
    eligible = sorted(r for r in by_row if r not in excluded_rows
                      and len(by_row[r]) == 2)

    rng_dev = random.Random(SEED_DEV)
    dev_rows = rng_dev.sample(eligible, N_DEV)
    remaining = sorted(set(eligible) - set(dev_rows))
    rng_test = random.Random(SEED_TEST)
    test_rows = rng_test.sample(remaining, N_TEST)

    def _emit(rows: list[int]) -> tuple[list[dict], dict[str, str]]:
        half = len(rows) // 2
        items, labels = [], {}
        for j, r in enumerate(rows):
            label = "faithful" if j < half else "hallucinated"
            c = by_row[r][label]
            sid = f"{prefix}_{r:05d}"
            items.append({"id": sid, "study_task": task, "task": c.task,
                          "context": c.context, "question": c.question,
                          "answer": c.answer, "source_index": r})
            labels[sid] = label
        items.sort(key=lambda d: d["id"])
        return items, labels

    dev_items, dev_labels = _emit(dev_rows)
    test_items, test_labels = _emit(test_rows)
    return {"dev": dev_items, "test": test_items}, dev_labels, test_labels


# ── RAGTruth sampling ────────────────────────────────────────────────────────

def _sample_ragtruth(excl: dict) -> tuple[dict, dict, dict]:
    import random
    _download_if_missing(RESPONSES_URL, RESPONSES_FULL)
    _download_if_missing(SOURCES_URL, SOURCES_FULL)

    sources = {str(r["source_id"]): r for r in _read_jsonl(SOURCES_FULL)
               if r.get("task_type") == "Summary"}
    excluded_resp = set(excl["ragtruth_sum"]["excluded_response_ids"])
    excluded_splits = set(excl["ragtruth_sum"]["excluded_splits"])

    by_source: dict[str, dict[str, list[dict]]] = {}
    for resp in _read_jsonl(RESPONSES_FULL):
        sid = str(resp.get("source_id"))
        if sid not in sources or resp.get("split") in excluded_splits:
            continue
        if str(resp["id"]) in excluded_resp:
            continue
        lab = "hallucinated" if resp.get("labels") else "faithful"
        by_source.setdefault(sid, {"faithful": [], "hallucinated": []})[lab].append(resp)
    for slot in by_source.values():
        for lab in slot:
            slot[lab].sort(key=lambda r: str(r["id"]))

    def _draw(rng: random.Random, pool: set[str], n_half: int) -> tuple[list[dict], dict[str, str]]:
        hal_sources = sorted(s for s in pool if by_source[s]["hallucinated"])
        picked_hal = rng.sample(hal_sources, n_half)
        rest = pool - set(picked_hal)
        fai_sources = sorted(s for s in rest if by_source[s]["faithful"])
        picked_fai = rng.sample(fai_sources, n_half)
        items, labels = [], {}
        for label, picked in (("hallucinated", picked_hal), ("faithful", picked_fai)):
            for s in picked:
                resp = rng.choice(by_source[s][label])
                sid = f"ragtruth_sum_{resp['id']}"
                items.append({"id": sid, "study_task": "ragtruth-sum", "task": "sum",
                              "context": _source_text(sources[s]), "question": "",
                              "answer": resp["response"], "source_index": s})
                labels[sid] = label
                pool.discard(s)
        items.sort(key=lambda d: d["id"])
        return items, labels

    pool = set(by_source)
    rng_dev = random.Random(SEED_DEV)
    dev_items, dev_labels = _draw(rng_dev, pool, N_DEV // 2)      # dev first (plan §3)
    rng_test = random.Random(SEED_TEST)
    test_items, test_labels = _draw(rng_test, pool, N_TEST // 2)
    # §7 escalation extension: continue the SAME rng past the original
    # 300-item draw so the committed 300 items are an exact prefix of the
    # 500. Do NOT re-sort across the boundary — prefix order is the audit
    # property.
    ext_items, ext_labels = _draw(rng_test, pool, (N_TEST_RAGTRUTH - N_TEST) // 2)
    test_items = test_items + ext_items
    test_labels = {**test_labels, **ext_labels}
    return {"dev": dev_items, "test": test_items}, dev_labels, test_labels


# ── Orchestration ────────────────────────────────────────────────────────────

def sample_all(write: bool = True) -> dict[str, str]:
    if not EXCLUDED_PATH.exists():
        raise SystemExit("study/excluded_ids.json missing — run --build-exclusions "
                         "and commit it before sampling (plan §3 order).")
    excl = json.loads(EXCLUDED_PATH.read_text())
    hashes: dict[str, str] = {}
    for task in TASKS:
        if task == "ragtruth-sum":
            splits, dev_labels, test_labels = _sample_ragtruth(excl)
        else:
            splits, dev_labels, test_labels = _sample_halueval(task, excl)
        for split, n in (("test", n_test_for(task)), ("dev", N_DEV)):
            items = splits[split]
            assert len(items) == n, (task, split, len(items))
            blob = json.dumps(items, indent=2, ensure_ascii=False) + "\n"
            path = ITEMS_DIR / f"{task}_{split}_{n}.json"
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(blob, encoding="utf-8")
            hashes[path.name] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            if task == "ragtruth-sum" and split == "test":
                # The superseded 300-item file stays committed for the audit
                # trail; it must equal the first 300 items of the 500 file
                # byte-for-byte (prefix-stability, §7 escalation).
                pblob = json.dumps(items[:N_TEST], indent=2, ensure_ascii=False) + "\n"
                hashes[f"{task}_test_{N_TEST}.json"] = hashlib.sha256(
                    pblob.encode("utf-8")).hexdigest()
        labels = {**test_labels, **dev_labels}
        lblob = json.dumps(dict(sorted(labels.items())), indent=2) + "\n"
        lpath = LABELS_DIR / f"{task}_labels.json"
        if write:
            lpath.parent.mkdir(parents=True, exist_ok=True)
            lpath.write_text(lblob, encoding="utf-8")
        hashes[f"labels_hidden/{lpath.name}"] = hashlib.sha256(lblob.encode("utf-8")).hexdigest()
    return hashes


def load_study_items(task: str, split: str) -> list[Case]:
    """Blinded loader for runner code: label is always the sentinel 'hidden'.

    run.py never reads Case.label; analysis must go through the guarded
    unblind path in study/analyze_study.py.
    """
    n = n_test_for(task) if split == "test" else N_DEV
    rows = json.loads((ITEMS_DIR / f"{task}_{split}_{n}.json").read_text())
    return [Case(id=r["id"], task=r["task"], context=r["context"],
                 question=r["question"], answer=r["answer"],
                 label="hidden", source_index=0 if isinstance(r["source_index"], str)
                 else r["source_index"])  # type: ignore[arg-type]
            for r in rows]


def verify() -> None:
    excl = json.loads(EXCLUDED_PATH.read_text())
    h1 = sample_all(write=False)
    h2 = sample_all(write=False)
    assert h1 == h2, "sampling is not deterministic"
    print("determinism: two in-memory re-samples give identical sha256 hashes")

    for task in TASKS:
        n_test = n_test_for(task)
        test = json.loads((ITEMS_DIR / f"{task}_test_{n_test}.json").read_text())
        dev = json.loads((ITEMS_DIR / f"{task}_dev_{N_DEV}.json").read_text())
        labels = json.loads((LABELS_DIR / f"{task}_labels.json").read_text())
        tids = {r["id"] for r in test}
        dids = {r["id"] for r in dev}
        assert not tids & dids, f"{task}: test/dev id overlap"
        tsrc = {r["source_index"] for r in test}
        dsrc = {r["source_index"] for r in dev}
        assert not tsrc & dsrc, f"{task}: test/dev share source units"
        for r in test + dev:
            assert "label" not in r, f"{task}: label leaked into item file"
            assert not r["id"].endswith(("_faithful", "_hallucinated")), \
                f"{task}: label leaked via id {r['id']}"
        assert set(labels) == tids | dids, f"{task}: label file id mismatch"
        n_hal_t = sum(labels[i] == "hallucinated" for i in tids)
        n_hal_d = sum(labels[i] == "hallucinated" for i in dids)
        assert n_hal_t == n_test // 2 and n_hal_d == N_DEV // 2, f"{task}: not balanced"

        if task == "ragtruth-sum":
            bad = {r["id"].replace("ragtruth_sum_", "")
                   for r in test + dev} & set(excl["ragtruth_sum"]["excluded_response_ids"])
            assert not bad, f"{task}: excluded pilot response ids present: {bad}"
            # §7 prefix-stability: the superseded committed 300-item file must
            # be byte-identical to the first 300 items of the 500-item file.
            old_blob = (ITEMS_DIR / f"{task}_test_{N_TEST}.json").read_text()
            new_prefix = json.dumps(test[:N_TEST], indent=2, ensure_ascii=False) + "\n"
            assert old_blob == new_prefix, f"{task}: 300-item file is NOT a prefix of the 500"
            h_old = hashlib.sha256(json.dumps(
                [r["id"] for r in json.loads(old_blob)]).encode()).hexdigest()
            h_new = hashlib.sha256(json.dumps(
                [r["id"] for r in test[:N_TEST]]).encode()).hexdigest()
            assert h_old == h_new
            print(f"{task}: prefix-stable escalation verified — sha256 of first "
                  f"300 ids of the 500 file == committed 300 file ids ({h_old[:16]}…)")
        else:
            key = task.replace("-", "_")
            bad = tsrc | dsrc
            bad &= set(excl[key]["excluded_source_rows"])
            assert not bad, f"{task}: excluded source rows present: {bad}"
        print(f"{task}: disjoint, balanced, exclusion-clean, label-free "
              f"({len(test)} test / {len(dev)} dev)")

    committed = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(ITEMS_DIR.glob("*.json"))}
    for name, digest in committed.items():
        assert h1[name] == digest, f"{name}: committed file differs from deterministic re-sample"
    print("committed item files match deterministic re-sample byte-for-byte")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-exclusions", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.build_exclusions:
        EXCLUDED_PATH.write_text(json.dumps(build_exclusions(), indent=2) + "\n")
        print(f"wrote {EXCLUDED_PATH}")
    if args.sample:
        for name, digest in sample_all(write=True).items():
            print(f"{digest}  {name}")
    if args.verify:
        verify()
    if not any([args.build_exclusions, args.sample, args.verify]):
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
