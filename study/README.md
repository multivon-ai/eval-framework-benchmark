# study/ — preregistered cross-framework disagreement study

Day-1 artifacts for the preregistered plan
(`research-plan-cross-framework-disagreement.md` v1.0, multivon-strategy
`reports/2026-07-13-best-framework-mission/`). Every deviation is in
`PREREG_ADDENDUM.md`; later deviations go in `DEVIATIONS.md`.

## Artifact map (Day 1 → plan section)

| Artifact | Plan § | What it is |
|---|---|---|
| `requirements.in` / `requirements.lock` | §4, §10 | Version-rule freeze (latest non-prerelease as of 2026-07-13 — date deviation in addendum §1). Hash-pinned; installed in `.venv-study` (Python 3.11.15). |
| `excluded_ids.json` | §3 | Pilot + multivon-eval-calibration exclusions, committed **before** sampling. Rules embedded in the file and in addendum §4. |
| `sample_items.py` | §3 | Deterministic sampler. Dev seed 20260714 drawn first, test seed 20260713, disjoint source units, balanced. |
| `items/{task}_test_300.json`, `items/{task}_dev_100.json` | §3 | Committed item ids + inputs, **no labels** (tasks: `ragtruth-sum`, `halueval-sum`, `halueval-qa`). |
| `../data/labels_hidden/{task}_labels.json` | §3 | Gold labels — read by exactly one guarded function in the repo. |
| `analyze_study.py` | §3, §7 | Study analysis entry point. `--unblind` refuses until a `study-freeze-*` git tag and a complete `study/FREEZE` manifest exist. Without it: label-free diagnostics only. |
| `power_sim.py` / `power_sim_results.json` | §9 | 10,000-replicate power gate. **Outcome: FAILED at n=300 on the strict worst-case grid (min 0.38/0.39)** — see addendum §6; escalation decision must be recorded before test-split spend. |
| `power_sim_endpoints.py` / `power_sim_endpoints_results.json` | §9 | Endpoint-targeted re-simulation of the gate for the ACTUAL confirmatory endpoints (H1/H4 median-κ bound, H3 gate-flip difference). **Outcome: FAILED at n=300 (H1/H4 ≥0.9999 everywhere; H3 central-grid min 0.5515 < 0.80; n→500 escalation clears H3 central at 0.8495)** — STOP on test-split spend; see addendum §7. |
| `PREREG_ADDENDUM.md` | §4, §10, §12 | Freeze-date deviation, OSF→git-tag substitution, exclusion rules, power outcome + resolution, strong-judge id (`gpt-5.5-2026-04-23`). |

## Rebuild the environment

```bash
uv venv --python 3.11 .venv-study
uv pip install --python .venv-study/bin/python --require-hashes -r study/requirements.lock
.venv-study/bin/python -c "import multivon_eval; print(multivon_eval.__version__)"  # 0.16.0
```

## Re-verify the sampling (no network needed once full datasets are cached)

```bash
.venv-study/bin/python study/sample_items.py --verify
```

Checks: byte-identical deterministic re-sample, test/dev disjointness (ids
and source units), 150/150 + 50/50 balance, exclusion cleanliness, no label
leakage into `study/items/`.

## Coming runs (Day 4–8; commands preregistered here)

Runner support for `--items study/items/...` and the TruLens/Opik adapters
land Day 2–3 (separate owner). Judges per plan §5: `gpt-4o-mini`,
`claude-haiku-4-5`, strong-judge `gpt-5.5-2026-04-23` (RAGTruth-Sum only,
n=150 subset, 1 run).

```bash
# Smoke (4 items × 5 frameworks × 2 judges; measures call multipliers, §11)
python run.py --smoke --judge gpt-4o-mini --judge claude-haiku-4-5

# Dev runs (thresholds fitted on dev, then locked — Condition B, §4)
python run.py --task ragtruth-sum --items study/items/ragtruth-sum_dev_100.json \
    --runs 1 --judge gpt-4o-mini --judge claude-haiku-4-5 --out results/study

# Test run 1 (all tasks) + repeats (RAGTruth R=5 both judges; HaluEval R=3 gpt-4o-mini)
python run.py --task ragtruth-sum --items study/items/ragtruth-sum_test_300.json \
    --runs 5 --judge gpt-4o-mini --judge claude-haiku-4-5 --out results/study

# Blinded diagnostics anytime; unblind only after the freeze tag exists
python study/analyze_study.py --task all
git tag study-freeze-2026-07-XX && python study/analyze_study.py --task all --unblind
```

Blinding contract: runner code paths read `study/items/` only (via
`sample_items.load_study_items`); nothing outside
`analyze_study.load_hidden_labels` may open `data/labels_hidden/`.
