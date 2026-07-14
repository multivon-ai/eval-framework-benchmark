# Prereg Addendum — Day 1 (2026-07-13)

Records every Day-1 deviation from and instantiation of the preregistered plan
(`research-plan-cross-framework-disagreement.md` v1.0, in multivon-strategy
`reports/2026-07-13-best-framework-mission/`). Written **before** any dev- or
test-split API call. Later deviations go in `DEVIATIONS.md` per plan §12.

## 1. Freeze-date deviation

Plan §4 fixed the version-rule timestamp at **2026-07-15 12:00 UTC**. The study
started two days early; the freeze was executed at **2026-07-13 (US/Pacific)**.
The rule itself is unchanged: latest non-prerelease PyPI release as of the
freeze timestamp. Verified for every pinned package against the PyPI JSON API
(PEP 440 ordering, prereleases/dev releases/yanked files excluded). Risk: a
framework release landing 2026-07-13→15 would have produced different pins;
we accept and disclose this.

## 2. OSF → git-tag substitution

Plan §4/§10 called for an OSF freeze before test-split spend. Substitution:
the prereg freeze is implemented as **git commits + a `study-freeze-*` git
tag** in this public repository (same immutability guarantee we already use
for results tags, e.g. `results-0.15.1-2026-06-26`). The unblinding guard in
`study/analyze_study.py` enforces the freeze order mechanically: gold labels
cannot be loaded until a `study-freeze-*` tag exists **and** `study/FREEZE`
lists the raw-output files it covers. The tag is created only after smoke +
raw-output freeze — it does not exist yet as of this commit.

## 3. Lockfile (plan §4)

`study/requirements.lock` — compiled with `uv pip compile --python-version 3.11
--generate-hashes` from `study/requirements.in`; installed into `.venv-study`
(Python **3.11.15**, uv 0.11.6) with `--require-hashes`. All five frameworks
import successfully. Framework pins (latest non-prerelease as of freeze):

| Package | Version | Wheel sha256 |
|---|---|---|
| multivon-eval | 0.16.0 | `a1bf2f596e63735672c4c05f9fae52808f748cc80fb76957effbfa72722eb9c8` |
| deepeval | 4.1.0 | `d493d80ac298eaa4336fd92a61539a7da836c0a8ee5870fb4075c30bd4d13dd6` |
| ragas | 0.4.3 | `ef1d75f674c294e9a6e7d8e9ad261b6bf4697dad1c9cbd1a756ba7a6b4849a38` |
| trulens-core | 2.8.1 | `ac93212130d744168998d1f8a59980140a1d749bb30b0960e1289dbcc7a51234` |
| trulens-feedback | 2.8.1 | `36a6557e2f9265720f233c2509aab9db484c28807239db72f5befcb29a0f79eb` |
| trulens-providers-openai | 2.8.1 | `7b89865727b75b5aa60866e6eeb8c8828cd1afafe542096a9a34db003baa25e3` |
| trulens-providers-litellm | 2.8.1 | `d48208f12a0db11d1ca4ee75924755eeaa13cb8e0618eaf466a9a082364b59e5` |
| opik | 2.1.22 | `c11bc8db4bdc370dd2ab6d6962901b9d2ae8e1ac04502ed53c27e4fe37f0ab98` |
| anthropic | 0.116.0 | `6c0a7698e8d652455da3499978279bb2588c7264d0a35be3666009a4258c8256` |
| openai | **1.109.1** | `6bcaf57086cf59159b8e27447e4e7dd019db5d29a438072fbd49c290c7e65315` |
| litellm | **1.80.0** | `fd0009758f4772257048d74bf79bb64318859adb4ea49a8b66fdbc718cd80b6e` |

**TruLens package-layout note.** `trulens` (2.8.1) on PyPI is a meta-package
whose only role is to pull `trulens-core`, `trulens-feedback`,
`trulens-dashboard[full]`, `trulens-otel-semconv`, `trulens_eval`. The
Groundedness feedback the study needs lives in `trulens-core`/
`trulens-feedback` with provider classes in `trulens-providers-openai`
(OpenAI judges) and `trulens-providers-litellm` (Anthropic judges). We pin the
four functional packages and skip the meta-package (which would drag in the
Streamlit dashboard, irrelevant to a CI-gate deployment).

**Resolver-forced deviations from "latest" for non-framework deps** (the
version rule in plan §4 applies to the *frameworks*; these are recorded
because the Day-1 instruction named them):

* `openai==1.109.1` (not 2.45.0): `trulens-providers-openai==2.8.1` caps
  `openai<2.0.0`. 1.109.1 is the newest release satisfying the joint
  constraint set. All five frameworks accept it (floors are >=1.x).
* `litellm==1.80.0` (not 1.92.0): litellm >=1.84 requires `openai>=2.20`,
  incompatible with the trulens cap; 1.80.0 is the newest litellm compatible
  with openai 1.x that also satisfies opik's exclusion list.
* `langchain-community>=0.3.29,<0.4` cap: ragas 0.4.3 imports
  `langchain_community.chat_models.vertexai`, removed in langchain-community
  0.4.x; the 0.3 line satisfies trulens-providers-openai's `>=0.3.29` floor.
  Without the cap, `import ragas` fails — this is itself a small piece of
  evidence for the fragility thesis, and is noted for the paper's limitations.

## 4. Sampling and exclusions (plan §3)

Seeds as preregistered: dev **20260714 drawn first**, test **20260713**,
disjoint; 150/150 test and 50/50 dev per task. Committed order: exclusions →
item ids → (later) runs. Instantiation details the plan left open:

* **Pilot exclusions** — every id in every committed `data/*pilot*.json`
  (halueval qa/sum pilots n=4/50/100; ragtruth pilots n=100/200) is excluded.
  For HaluEval the *entire source row* is excluded (both the faithful and the
  hallucinated variant), so no study item is the paired counterpart of a
  published pilot item.
* **Calibration exclusions** — multivon-eval's calibration
  (`multivon_eval/_calibration_data/v1.json`, `v2.json`; scripts
  `benchmarks/run_threshold_calibration.py`, `run_calibration_v2.py`,
  `run_truly_held_out.py`) loads `[:n]` prefixes of the raw HaluEval files:
  max observed prefixes are 50 QA rows and 30 Sum rows. Individual item ids
  are therefore exactly identifiable as file-order prefixes; we exclude a
  conservative superset — **source rows 0–99** of both `qa_data.json` and
  `summarization_data.json` — covering any prefix use up to n=100.
  No RAGTruth item was ever used in multivon-eval calibration (verified by
  repo-wide grep: RAGTruth appears only in a roadmap sentence in
  `benchmarks/README.md`).
* **RAGTruth pool = 'train' split.** The committed pilots were drawn from the
  'test' split, which contains only 204 hallucinated Summary responses; after
  pilot exclusion, 104 remain — fewer than the 200 the study needs. The study
  pool is therefore the Summary **'train' split** (3,276 faithful / 1,482
  hallucinated responses, 793 sources), which shares **zero source documents**
  with the pilot pool. RAGTruth's train/test names refer to its original
  model-training protocol and carry no evaluative meaning for this study.
* **Sampling-unit refinement** (stricter than the plan's "disjoint"):
  HaluEval — the unit is the source row; each sampled row contributes exactly
  one variant, rows disjoint within and across splits, so no two study items
  share a source document. RAGTruth — the unit is the source document; one
  response per source, sources disjoint within and across splits. This makes
  the item-cluster bootstrap's independence assumption exact.
* **Anti-leak ids** — upstream loaders embed the gold label in the case id
  (`sum_00106_faithful`); study ids strip the suffix (`sum_00106`,
  `ragtruth_sum_13731`). Unique because each unit contributes one variant.
* Verified (`sample_items.py --verify`): byte-identical re-sampling across
  processes, test/dev disjointness at id and source-unit level, 150/150 and
  50/50 balance, zero excluded ids present, no label field or label-bearing
  id in `study/items/`.

## 5. Label blinding (plan §3)

Gold labels live only in `data/labels_hidden/{task}_labels.json`. Blinding is
enforced in a **new** `study/analyze_study.py` rather than a flag on
`analyze.py` (decision noted per Day-1 instruction: the pilot's `analyze.py`
loaders return labels inline, so retrofitting would leave label-bearing paths
importable by the runner; the study's raw-output layout also differs).
`load_hidden_labels()` is the only reader of `labels_hidden/` in the repo and
refuses unless a `study-freeze-*` git tag exists **and** `study/FREEZE` lists
existing raw-output files. Runner code paths read `study/items/` only, via
`sample_items.load_study_items()` (label field = sentinel `"hidden"`).

## 6. Power gate outcome (plan §9) — **FAILED at n=300 on the strict grid**

`study/power_sim.py`, seed 42, 10,000 replicates/cell, two-sided α=.05,
grouped item-jackknife inference (type-I calibration verified: rejection rate
0.048 / 0.049 on true-null cells). Results in `study/power_sim_results.json`:

| Endpoint | min power n=300 | min power n=500 | best-case region |
|---|---|---|---|
| dependent-κ difference 0.15 | **0.38** | 0.57 | 0.73 (p=0.5, κ 0.70→0.55) |
| paired-F1 gap 0.10 | **0.39** | 0.57 | ≥0.99 (p≥0.65) |

The worst cells are the extreme positive rates (0.20 / 0.80). At the central
p=0.5 cells: κ-difference power 0.50–0.73; F1-gap power 0.79–0.91. The
preregistered sole escalation (RAGTruth-Sum n→500) does **not** clear the
0.80 bar on the worst-case grid reading either.

**Consequence, recorded before any spend:** under plan §9 the gate governs
test-split API spend. The confirmatory endpoints H1/H4 are median-κ bounds
against a fixed 0.40 bar (one-sample), not the 0.15 κ-difference — the gate's
minimum effects map to the secondary κ_self/κ_cross contrast and the P4 F1
spread. Options before test spend (decision to be recorded here before Day 5):
(a) escalate RAGTruth-Sum to n=500 (contingency line item, ~$400 cap) and
re-state the two underpowered endpoints as descriptive-with-CIs rather than
tested; (b) hold n=300 and preregister the same re-statement; (c) stop.
No test-split call happens until one of these is committed. Dev runs and the
Day-1 smoke are unaffected (the gate conditions *test-split* spend).

## 7. Refined, endpoint-targeted power gate — **FAILED at n=300 (H3)**

Resolution of the §6 escalation, recorded before any dev/test API call.
Code: `study/power_sim_endpoints.py`; results:
`study/power_sim_endpoints_results.json`.

**Deviation, stated plainly:** the original §6 gate (plan §9 grid) is
replaced by an endpoint-targeted gate, and the **pairwise κ-difference
contrasts (κ_self/κ_cross and the 0.15 dependent-κ difference) are DEMOTED
to exploratory**. Reason: the original grid conflated a *secondary contrast*
(the κ-difference, whose min power was 0.38 at n=300) with the *primary
bounds* — the confirmatory endpoints H1/H4 are one-sample median-κ bound
tests against the fixed 0.40 bar, and H3 is the gate flip-rate difference,
none of which the original grid simulated. The P4 F1-spread endpoint
(original min power 0.39) is likewise reported descriptive-with-CIs, not
tested. The demotion and the decision rule below were committed in the
script *before* observing any H3 power number: the first execution died
with its parent process after printing the H1 grid (whose cells all sit at
≈1.0) and before any H3 cell completed, and the rule was not altered
afterwards.

**Simulation spec:** seed 42, per-cell streams `default_rng([42, section,
cell])`; H1/H4 10,000 replicates/cell; **H3 2,000 replicates/cell — a
committed reduction** (MC SE ≤ 0.011) adopted after the 10,000×18-cell first
execution was killed mid-run, decided before observing any H3 result.
Type-I calibration: H1 at the bar 0.0196 (≈ nominal .025, one-sided); H3 at
the exchangeable null 0.0005 — the jackknife CI is *conservative*, never
anti-conservative.

**Decision rule (committed):** proceed at n=300 **only if** H1 power ≥ 0.80
in **all** cells (true κ ≤ 0.20, marginal rates 0.20–0.80) **and** H3 power
≥ 0.80 on the **central scenario grid** (induced cross-framework gate-flip
probability ≈ 0.24 within the plausible range 0.15–0.35; within-framework
0.02–0.10). If the gate fails: **stop — no test-split spend** — and report.

**Results (n=300):**

| Endpoint | Grid | Min power |
|---|---|---|
| H1 median-κ bound < 0.40 | all 20 cells (κ ≤ 0.20 × p 0.20–0.80) | **0.9999** |
| H4 (= H1 restricted to central marginals 0.35–0.65, post-tuning) | 12 cells | **1.0000** |
| H3 flip-rate difference | central grid (3 ρ_within levels) | **0.5515** |
| H3 flip-rate difference | all 9 cells (weak/central/strong) | 0.1840 |

H3 cell detail (m=150 faithful): weak (cross ≈ 0.149) 0.18–0.35; central
(cross ≈ 0.244) 0.55–0.68; strong (cross ≈ 0.350) 0.88–0.93.

**GATE OUTCOME: FAILED.** H1/H4 clear the bar everywhere; H3 does not
(0.5515 < 0.80 on the central grid at n=300). Under the committed rule this
**stops all test-split spend**. Recorded observation, no decision implied:
the preregistered sole escalation (RAGTruth-Sum n→500, i.e. m=250 faithful)
yields H3 central-grid min power **0.8495** (weak grid remains underpowered,
0.42–0.59) — escalation is a viable path but requires an explicit committed
decision here before any test-split call, per §6 options (a)/(c).

### §7.1 Committed escalation decision — **ESCALATE** (2026-07-13, before any test-split call)

**Decision: option (a).** The RAGTruth-Sum **test split is escalated to
n=500 (250/250 balanced)**, the plan §9's *sole preregistered escalation*
(contingency line item). Basis: the refined gate failed H3 at n=300
(central-grid min power 0.5515 < 0.80) and clears at n=500 (central-grid
min power **0.8495** ≥ 0.80, `power_sim_endpoints_results.json`).

**Scope of the H3 confirmatory claim, stated plainly:** H3 is confirmatory
**only on the central effect grid** (induced cross-framework gate-flip
probability ≈ 0.24; within-framework 0.02–0.10). The **weak-effect regime**
(cross-framework flip ≈ 0.15 vs within ≈ 0.10 style cells) remains
underpowered at n=500 (0.42–0.59) and is acknowledged as **possibly
indeterminate at this n**: a null H3 result there is uninformative and will
be reported descriptively, never as evidence of absence. HaluEval-Sum/QA
test splits **stay at n=300** (they are replications, not gatekeepers).

**Sampling instantiation (prefix-stable extension).** Same seed discipline:
dev (seed 20260714) drawn first and unchanged; test seed 20260713; splits
disjoint at id and source-document level; exclusions from
`study/excluded_ids.json`; balanced 250/250. The extension reproduces the
original 300-item draw exactly and then *continues the same RNG stream* to
draw 100 more hallucinated + 100 more faithful source units from the
remaining pool — so the committed 300-item file is **byte-for-byte the
prefix** of `study/items/ragtruth-sum_test_500.json` (never mixed, nothing
superseded-and-resampled). Verified by `sample_items.py --verify`
(two-process determinism, prefix hash equality, disjointness, balance,
exclusion-cleanliness, label-freedom). Hashes (sha256):

* `ragtruth-sum_test_500.json` `9145e0aa66ff604656ac5aecccb0075befd547c143125f5c4376f5362a7f8e10`
* `ragtruth-sum_test_300.json` (= first 300 items of the 500, unchanged)
  `feec0d4982cd37cde9b725f90e8f71044e1c71fe83a0f40d9bd49354e2983f15`
* first-300-ids hash of both files: `855de1c159cdc336…` (identical)
* `labels_hidden/ragtruth-sum_labels.json` (600 labels, still blinded)
  `ceec5c6b7cd121bccef98a0c972a837c033491d3692c9d6bd0613d47c221f7bb`

All other item files are byte-unchanged.

**Updated eval count (plan §6 table recomputed with RAGTruth test = 500):**

| Cell | Frameworks | Judges | Items | Runs | Evals |
|---|---|---|---|---|---|
| Dev | 5 | 2 | 100×3 tasks | 1 | 3,000 |
| Test run 1 (all tasks) | 5 | 2 | 500 + 300×2 | 1 | 11,000 |
| Repeats, RAGTruth-Sum | 5 | 2 | 500 | +4 (R=5) | 20,000 |
| Repeats, HaluEval-Sum/QA | 5 | gpt-4o-mini only | 300×2 | +2 (R=3) | 6,000 |
| Strong-judge ablation | 5 | 1 | 150 | 1 | 750 |
| **Total** | | | | | **40,750** |

(Previously 30,750; +10,000 evals, all RAGTruth-Sum.) The §7 gate condition
on test-split spend is hereby satisfied by escalation; test-split spend
remains additionally conditioned on the §8 re-issued cost projection
staying within the ~$780 cap.

## 8. Smoke run + measured call multipliers — **RUN 2026-07-13, adapters 5/5, WITHIN CAP**

Executed after the §7.1 escalation decision was committed (`study/smoke.py`;
raw traces in `study/smoke/raw/`, aggregates in `study/smoke/multipliers.json`).
Design: 4 blinded dev items (first 2 by id of ragtruth-sum + halueval-sum
dev) × 5 frameworks × 2 primary judges, sequential; every judge HTTP call
intercepted at the shared httpx transport (calls, provider-reported usage
tokens, latency, cost at pinned prices: gpt-4o-mini $0.15/$0.60,
claude-haiku-4-5 $1.00/$5.00 per Mtok).

**Adapter drop-rule check: all 5 adapters returned real verdicts on all
4 items under both judges (40/40 evals, 0 errors).** One transport fix was
required mid-smoke (recorded in §9): RAGAS's Anthropic path initially 400'd
on all 4 claude items and was re-run after the fix.

**Measured multiplier table (mean per eval; calls / prompt-tok /
completion-tok / latency s / $):**

| Framework | gpt-4o-mini | claude-haiku-4-5 |
|---|---|---|
| multivon-eval | 6.25 / 4,136 / 111 / 8.1 s / $0.0007 | 6.75 / 4,672 / 173 / 11.5 s / $0.0055 |
| deepeval | 3.0 / 2,216 / 549 / 11.8 s / $0.0007 | 3.0 / 2,340 / 785 / 8.4 s / $0.0063 |
| ragas | 2.0 / 2,056 / 609 / 132.4 s / $0.0007 | 2.25 / 4,679 / 1,247 / 161.2 s / $0.0109 |
| trulens | 4.5 / 4,800 / 379 / 6.2 s / $0.0009 | 4.75 / 8,021 / 767 / 4.4 s / $0.0119 |
| opik | 1.0 / 1,233 / 101 / 1.9 s / $0.0002 | 1.0 / 1,500 / 219 / 3.5 s / $0.0026 |

Measured calls/eval (1–6.75) sit **below** the plan's 4×-mean assumption;
RAGAS's pilot-era ~130 s/eval latency is confirmed (asyncio event-loop
setup per call, not token volume).

**Re-issued cost projection, full escalated design (40,750 evals, §7.1
cell structure):**

| Line item | USD |
|---|---|
| gpt-4o-mini cells (23,000 evals = 4,600/fw) | $14.72 |
| claude-haiku-4-5 cells (17,000 evals = 3,400/fw) | $126.48 |
| Strong-judge ablation, 750 evals (plan's $35 × measured token factor 0.238) | $8.33 |
| A2 proxy/replay + smoke + reruns (plan line) | $40.00 |
| **Subtotal (escalation now priced into the cells)** | **$189.53** |
| Contingency (plan line, retained as pure headroom) | $400.00 |
| **Projected total** | **$589.53** |

**$589.53 ≤ $780 cap — WITHIN CAP** (even carrying the full $400
contingency, whose original purpose — the n→500 escalation — is already
priced into the cells above). Test-split spend is authorized under the §7
gate-by-escalation **and** this projection.

**Wall-clock projection at `--workers 8`** (per §7.1-row cell × framework ×
judge, measured mean latency): every cell < 12 h; **no cell flagged**. The
worst cell is RAGAS × claude-haiku-4-5 RAGTruth repeats (2,000 evals) at
**11.19 h** — under the 12 h bar but with <7% headroom, so RAGAS repeat
runs will be scheduled first within D5–7. Next-worst: RAGAS × gpt-4o-mini
repeats 9.20 h; all non-RAGAS cells ≤ 0.82 h.

**Judge snapshots confirmed from response metadata (closes §11):**
`gpt-4o-mini-2024-07-18` and `claude-haiku-4-5-20251001` on every recorded
call — exactly the §11 predictions.

## 9. Harness instantiation resolutions (recorded before any dev/test call)

* **Static summarization string.** Plan §4's Unavoidable Configuration
  Rule used "Summarize the text." as its *example* string. Resolution: the
  study uses the harness's existing pilot-era string — **"Provide a
  faithful summary of the document."** — now hoisted to a single constant
  (`frameworks/base.py: STATIC_SUMMARIZATION_INPUT`) and supplied
  identically to every framework whose native schema requires a question
  field on summarization (DeepEval, RAGAS, Opik, multivon-eval). Rationale:
  (a) pilot comparability — results-0.15.1 was produced with this string
  for DeepEval/RAGAS; (b) the prereg's *intent* is one identical string
  across frameworks, which the letter-of-the-plan string would have broken
  against the pilot. Two adapter deviations recorded: the new Opik adapter
  briefly carried the plan's example string (never used in any run);
  the pilot's multivon-eval adapter embedded the full document into
  `input` ("Summarize this document.\n\n<context>") — it now receives the
  same static string as everyone else (context is already passed via the
  `context` field; the embedded variant was also a COI hazard, since only
  the in-house framework got the document twice).
* **Opik judge transport pinned to the native Anthropic SDK.** Opik
  2.1.22's `models_factory` routes `claude-*` model names to its native
  `AnthropicChatModel` (Anthropic SDK) whenever `anthropic` is importable;
  `anthropic==0.116.0` is hash-pinned in `study/requirements.lock`, so
  this is the deterministic lockfile outcome. The adapter now *asserts*
  the resolution (fails fast on a silent LiteLLM fallback). gpt-* judges
  keep Opik's `LiteLLMChatModel` default.
* **Claude-judge plumbing gaps in the pilot adapters (fixed).**
  DeepEval: passing the bare string `claude-haiku-4-5` routed to the
  OpenAI endpoint and 404'd on 100/100 items in the pilot's
  claude-haiku-4-5 column (`results/raw/claude-haiku-4-5/deepeval_*`);
  the adapter now wraps claude ids in DeepEval's shipped
  `deepeval.models.AnthropicModel` (native SDK). RAGAS: the pilot wiring
  was OpenAI-only (`ChatOpenAI`); claude ids now use RAGAS's canonical
  `ragas.llms.llm_factory(provider="anthropic")` with a native Anthropic
  client — `PydanticPrompt.generate` supports these natively, so shipped
  prompts/parsers are untouched. Both are transport plumbing, not
  configuration, and apply identically across Conditions A/B. All five
  adapters construct successfully for both primary judges (offline check,
  no API calls).
* **RAGAS Anthropic top_p conflict (found and fixed during the §8 smoke).**
  RAGAS's `InstructorModelArgs` defaults `top_p=0.1` alongside
  `temperature`, and its Anthropic parameter mapping is pass-through;
  `claude-haiku-4-5` rejects requests specifying *both* `temperature` and
  `top_p` (400 `invalid_request_error`), so the first smoke pass 400'd on
  all 4 RAGAS×claude items. Fix: the adapter drops `top_p` after
  `llm_factory(...)`, sending `temperature=0.0` only — exactly what the
  RAGAS OpenAI path (`ChatOpenAI(temperature=0.0)`, no top_p) already
  sends. Transport plumbing, symmetric across judges in effect; the 4
  failed evals were re-run after the fix (the raw file contains only
  post-fix results; the pre-fix error is quoted here as the record).

## 10. Strong-judge snapshot (plan §5, frozen Day 1)

Rule applied: from `GET /v1/models` (queried 2026-07-13; free metadata
endpoint, no inference spend), take ids matching
`^gpt-<major[.minor]>-<YYYY-MM-DD>$` — i.e. **dated snapshots of the mainline
(strong-tier) series**, excluding `-mini`/`-nano`/`-pro`/`-codex`/`-chat`/
`-audio`/`-realtime`/`-image`/`-search` variants and undated codename models
(`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra` were visible but have no dated
snapshot and are excluded as codenames) — then pick the highest version,
tiebreak by latest date.

Selected: **`gpt-5.5-2026-04-23`** (candidates considered: gpt-5-2025-08-07,
gpt-5.1-2025-11-13, gpt-5.2-2025-12-11, gpt-5.4-2026-03-05,
gpt-5.5-2026-04-23).

## 11. Primary-judge snapshots (to pin at smoke)

`gpt-4o-mini` resolves to dated snapshot `gpt-4o-mini-2024-07-18` (only dated
mini snapshot in the models list); `claude-haiku-4-5` resolves to
`claude-haiku-4-5-20251001` (the snapshot recorded throughout multivon-eval's
calibration data). Both will be re-confirmed and recorded from response
metadata during the Day-1 smoke run. **Confirmed at the §8 smoke
(2026-07-13): every recorded judge call reported exactly these two
snapshot ids.**

## §10 — Dev closeout: error triage, repair, Condition-B thresholds (locked)

**Triage (68 errored dev records):** deepeval×gpt-4o-mini 27 `RetryError` timeouts (transient class); ragas×claude-haiku-4-5 41 `InstructorRetryException` structured-output parse failures (systematic class — the framework's parser degrading against a cross-provider judge is a framework property and is retained as data). One documented repair pass re-attempted exactly the errored ids: deepeval 27→17 recovered 10; ragas 41→39 recovered 2. Final dev error census: 56/3,000 (1.9%); errored records carry no score and are excluded from threshold fitting with counts disclosed below (errors-as-failures remains the preregistered primary for test analysis).

**Dev-label carve-out:** `analyze_study.py --unblind-dev` loads dev-split labels only (requires the `study-freeze-*` tag; refuses any test id). Test labels remain gated on the FREEZE manifest.

**Condition-B fit (preregistered algorithm):** candidates = midpoints between consecutive unique dev scores + extremes; max dev-F1 (hallucinated = positive); ties toward the stricter gate. Deterministic (double-run hash-identical: cef43eac…). Early observation recorded verbatim, no test implication: Opik's shipped 0.5 default vs fitted on ragtruth-sum×claude-haiku moves dev-F1 0.179→0.811 — the default-threshold collapse observed for DeepEval in the pilot appears in a second framework.

| framework | judge | task | default τ | fitted τ | default dev-F1 | fitted dev-F1 | n errored |
|---|---|---|---|---|---|---|---|
| deepeval | claude-haiku-4-5 | halueval-qa | — | 1 | nan | 0.622 | 0 |
| deepeval | claude-haiku-4-5 | halueval-sum | — | 1 | nan | 0.692 | 0 |
| deepeval | claude-haiku-4-5 | ragtruth-sum | — | 0.9045 | nan | 0.444 | 0 |
| deepeval | gpt-4o-mini | halueval-qa | — | 1 | nan | 0.750 | 17 |
| deepeval | gpt-4o-mini | halueval-sum | — | 1 | nan | 0.634 | 0 |
| deepeval | gpt-4o-mini | ragtruth-sum | — | 0.9258 | nan | 0.513 | 0 |
| multivon-eval | claude-haiku-4-5 | halueval-qa | — | 1 | nan | 0.837 | 0 |
| multivon-eval | claude-haiku-4-5 | halueval-sum | — | 1 | nan | 0.691 | 0 |
| multivon-eval | claude-haiku-4-5 | ragtruth-sum | — | 1 | nan | 0.752 | 0 |
| multivon-eval | gpt-4o-mini | halueval-qa | — | 1 | nan | 0.774 | 0 |
| multivon-eval | gpt-4o-mini | halueval-sum | — | 1 | nan | 0.673 | 0 |
| multivon-eval | gpt-4o-mini | ragtruth-sum | — | 1 | nan | 0.637 | 0 |
| opik | claude-haiku-4-5 | halueval-qa | — | 0.925 | nan | 0.940 | 0 |
| opik | claude-haiku-4-5 | halueval-sum | — | 0.875 | nan | 0.733 | 0 |
| opik | claude-haiku-4-5 | ragtruth-sum | — | 0.9 | nan | 0.811 | 0 |
| opik | gpt-4o-mini | halueval-qa | — | 0.9 | nan | 0.891 | 0 |
| opik | gpt-4o-mini | halueval-sum | — | 0.55 | nan | 0.729 | 0 |
| opik | gpt-4o-mini | ragtruth-sum | — | 1 | nan | 0.667 | 0 |
| ragas | claude-haiku-4-5 | halueval-qa | — | 1 | nan | 0.832 | 0 |
| ragas | claude-haiku-4-5 | halueval-sum | — | 1 | nan | 0.699 | 4 |
| ragas | claude-haiku-4-5 | ragtruth-sum | — | 0.8661 | nan | 0.793 | 35 |
| ragas | gpt-4o-mini | halueval-qa | — | 1 | nan | 0.772 | 0 |
| ragas | gpt-4o-mini | halueval-sum | — | 1 | nan | 0.673 | 0 |
| ragas | gpt-4o-mini | ragtruth-sum | — | 1 | nan | 0.714 | 0 |
| trulens | claude-haiku-4-5 | halueval-qa | — | 1 | nan | 0.838 | 0 |
| trulens | claude-haiku-4-5 | halueval-sum | — | 1 | nan | 0.730 | 0 |
| trulens | claude-haiku-4-5 | ragtruth-sum | — | 0.9554 | nan | 0.838 | 0 |
| trulens | gpt-4o-mini | halueval-qa | — | 0.5 | nan | 0.761 | 0 |
| trulens | gpt-4o-mini | halueval-sum | — | 0.7889 | nan | 0.737 | 0 |
| trulens | gpt-4o-mini | ragtruth-sum | — | 0.9028 | nan | 0.709 | 0 |

## §11 — Strong-judge ablation (plan §5 judge 3 / A1) + A2 fallback decision (2026-07-15)

### §11.1 Ablation subset rule

`study/items/ragtruth-sum_ablation_150.json` = the **FIRST 150 items of the
committed `ragtruth-sum_test_500.json`** — a deterministic prefix; **no new
sampling decisions** were taken. The file is byte-derived
(`json.dumps(items[:150], indent=2, ensure_ascii=False)`) from the committed
500-item file (sha256 `9145e0aa…`, unchanged, §7.1) and verified byte-exact
by `sample_items.py --verify`. sha256 of the subset file:
`20453c0fc70105ec07e838e8036107703ea11af2660942e5211a8c4180c23a3d`; sha256 of
its id list: `c25ea5da29ba349f…`. Labels stay hidden; because the prefix is
id-sorted within the original 300-item draw, the subset's label balance is
**unknown by construction** (labels were never read) and will be reported at
unblinding. Runner: `run_study.py --split ablation` accepts **exactly one
cell shape** — ragtruth-sum × gpt-5.5 × run 0, all 5 frameworks — and refuses
every other ablation address and any non-ablation use of the gpt-5.5 judge.
(Operational addition: `--stop-after N` batches one invocation's submissions
for foreground execution; unlike `--limit` it changes no item set and stamps
nothing.)

### §11.2 Snapshot probe + forced temperature deviation

One-probe verification (raw chat completion, before any cell spend): the API
**accepts `gpt-5.5-2026-04-23` and echoes exactly that id** in response
metadata. However the snapshot is reasoning-tier: it **rejects
`temperature=0`** (400 `unsupported_value` — "Only the default (1) value is
supported") and **rejects `max_tokens`** (requires `max_completion_tokens`).
Plan §5's "temperature = 0 everywhere" is therefore **unsatisfiable on this
judge**; the effective ablation judge temperature is the provider default
(1) for all five frameworks — symmetric, and consistent with plan §5's rule
that framework-hardcoded judge params are "part of the framework". Per-
framework instantiation (verified with one off-cell dev-item probe each,
item `ragtruth_sum_0`):

* **multivon-eval** — judge path already omits temperature and uses
  `max_completion_tokens`; no change.
* **deepeval** — ships a model registry that knows `gpt-5.5-2026-04-23`
  (`supports_temperature=False` → auto-adjusts to 1); no change.
* **ragas** — hardwires a call-time judge temperature of **0.01**
  (`BaseRagasLLM.get_temperature(n=1)`), which bypasses langchain's own
  init-time gpt-5 temperature drop → 400 on every call. Adapter fix
  (transport plumbing, same class as the §9 top_p drop): on the gpt-5.x
  OpenAI path only, the wrapper's `get_temperature` returns the provider's
  sole accepted value (1.0). The primary-judge path is byte-unchanged
  (still 0.01).
* **trulens** — native reasoning-model handling (`gpt-5` prefix →
  temperature dropped, `reasoning_effort="medium"` added, OpenAI Responses
  API); no change; recorded as framework property.
* **opik** — LiteLLM drops `temperature=0.0` with a logged warning ("only
  supports temperature=1"); no change.

### §11.3 Run + error census (750 evals, run 2026-07-14→15, workers 4)

All 5 cells complete, 150/150 records each, no duplicate ids, no
limit-stamps, **no 429s observed** (main test lanes ran concurrently).
**Judge snapshot `gpt-5.5-2026-04-23` confirmed on all 600 thread-attributed
records** (deepeval's calls are never thread-attributable in this runner —
same as every deepeval cell; its model id is the pinned snapshot and the
probe confirmed the API echo).

| framework | records | errors | error classes |
|---|---|---|---|
| multivon-eval | 150 | **71 (47.3%)** | 67 ValueError empty-claims-reply (internal claims call hardcodes `max_tokens=512`; reasoning tokens exhaust it, ct=512 exactly); 3 JudgeUnavailable 400 "could not finish message: output limit reached" (same cause, API-side); 1 JudgeUnavailable verdict coverage 4/10 |
| deepeval | 150 | 0 | — |
| ragas | 150 | 0 | — |
| trulens | 150 | 0 | — |
| opik | 150 | 0 | — |

The multivon-eval errors are **systematic** (framework-internal token budget
vs reasoning-token accounting — a framework property under a stronger judge,
directly analogous to the ragas×claude parse degradation in §10): per the
triage protocol they are **kept as data, no repair pass** (errors-as-failures
is the preregistered primary). Overall ablation error rate 71/750 = 9.5%,
concentrated 100% in one framework — the api_error_rate secondary endpoint
will carry this.

**Ablation spend.** Attributed tokens: 1,178,272 prompt + 564,763
completion → **$22.84** at the registry price $5/$30 per Mtok (deepeval and
litellm registries agree; no public price sheet existed at freeze, §8).
Adding unattributed deepeval (~$9 est. from its call profile) and the
retried error attempts (~$5, last-attempt-only tokens are in the records),
total ≈ **$35–38 vs the §8 scaled estimate of $8.33**. Cause: reasoning
tokens billed as completion tokens — the §8 scaling used the gpt-4o-mini
token profile, which a reasoning tier does not follow. The ~$29 overrun is
absorbed by the untouched $400 contingency line; projected study total
rises to ≈ $620, still **within the $780 cap**.

### §11.4 A2 fallback decision (preregistered)

**Decision: the A2 logging-proxy replay is NOT attempted — the plan §8
fallback is the outcome.** The proxy's preregistered 1-day time-box was
consumed; per the plan's explicit clause ("if the proxy is not working
within 1 day, use standalone prompt extraction + a shared minimal parser,
and report the parsing confound as a limitation") the fallback instrument
was built and executed (`study/a2_prompts.py`): (a) exact judge request
bodies captured at the shared httpx transport for one fixed dev item
(`ragtruth_sum_0`) for all 5 frameworks × both primary judges →
`study/a2/prompts/{framework}_{judge}.json`; (b) RAGAS's canonical NLI
judge prompt sent once per primary judge (2 calls, temperature 0) and the
identical raw completion offered to each framework's own parse path where
standalone-reachable (trulens: primary parse path NOT-REACHABLE standalone,
terminal regex fallback exercised and labelled; opik's parser rejects the
foreign completion format — format coupling, both judges). Report with the
prompt-scaffold fingerprint table (call counts 1–8, prompt sizes 5.2k–22k
chars, Responses-API/structured-output/system-message splits, rubric
wording markers) and the paper's limitation paragraph:
`study/a2/A2_REPORT.md`. Part A+B spend: 10 evals + 2 completions ≈ $0.06,
under the $1 JOB-2 budget.

## §12 — Pre-unblind endpoint clarifications (2026-07-15)

Recorded **before** the `study/FREEZE` manifest exists and therefore before
any test-label access is mechanically possible (the §2/§5 guard). Every item
below closes an audited piece of decision-rule wiggle-room found in the
results-blind manuscript review (reviewers R1/R4); none touches a test
label, a raw test record's content, or any already-computed number. Where a
rule is encoded in `study/analysis/`, the pipeline was updated in this same
commit and re-validated on the dev split (deterministic double-run).

1. **H2 falsification condition (c), made symmetric with confirmation.**
   Condition (c) fires iff **≥6/10 Δ_fg BCa CIs include zero AND pooled
   B ≤ W** (point estimates, per repeated cell). On the P5 kill-switch set
   (4 frameworks = 6 pairs) the analogue is **≥4/6**; both instantiate the
   same strict-majority rule (count > n_pairs/2) that
   `analysis/decomposition.py` now computes (`h2_falsification_c`).
   *Rationale:* confirmation said "≥6/10" while falsification said "most" —
   an asymmetric escape hatch (R4, Results Finding 1).

2. **Kill-switch (P5) pair rule for H2.** On the reduced 4-framework set,
   H2's confirmation bar is **≥4/6 of the remaining pairs** with Δ_fg CI
   excluding zero from above (strict majority of the 6 third-party pairs),
   pooled B−W criterion unchanged. This is what the pipeline's existing
   `h2_majority_positive` (n_pos > n_pairs/2) already evaluates on the
   6-pair set. *Rationale:* "≥6/10" cannot be carried over literally to 6
   pairs; as written P5 was not mechanically executable for H2 (R1 top
   issue 5, scribe-verified).

3. **H5 descriptive endpoint fixed.** H5 (mechanism, exploratory) =
   (a) the A2 fallback prompt-scaffold fingerprint table (§11.4,
   `study/a2/A2_REPORT.md`) and (b) the **proportion of framework pairs
   whose disagreement pattern is consistent with scaffold differences**
   (pair disagrees AND the pair's scaffold fingerprints differ on ≥1
   recorded axis), reported **descriptively — no hypothesis test, no
   confirmation/falsification bar**. *Rationale:* "disagreement localizes
   to the scaffolds" had no defined metric, leaving post-unblind
   discretion (R4, Design Finding 1).

4. **Mixed-judge F1 falsification fires per-judge arm.** Falsification
   condition (b) — default-τ F1 spread within 0.05 — **fires separately
   for each primary-judge arm**: spread ≤ 0.05 for one judge fires (b) for
   that arm and is reported as such; there is no both-judges conjunction
   and no undefined mixed state. Encoded as
   `falsification_b_fires_this_judge_arm` in `analysis/performance.py`
   (Condition A). *Rationale:* the both-judges requirement made
   falsification needlessly hard and left mixed results in a gray area
   (R4, Results Finding 3).

5. **P3 relabeled "false-positive-batch gate-flip rate".** Construction
   byte-unchanged (1,000 batches of 50 resampled from the faithful-labeled
   half; gate fail iff flagged rate > 20%; H3 statistic T = p_cross −
   p_within with jackknife 95% CI). The label in `ANALYSIS.md` and the
   JSON `endpoint` field now say what the estimand is: flip probability of
   a **false-positive control batch**, not generic deployed release-gate
   behavior. *Rationale:* the old label invited a broader deployed-gate
   reading the estimand does not support (R1 top issue 3,
   scribe-verified).

6. **Δ_fg's stated model = conditional Bernoulli (framing only; math
   unchanged).** For each item i, framework f's runs are modeled iid
   Bernoulli(p_if), independent across runs given the item; under this
   model Δ_fg = E_i[(p_if − p_ig)²] ≥ 0, with equality iff p_if = p_ig
   a.s. — the null "exchangeable wrappers" is formally p_if = p_ig for all
   i. Outside the model, Δ_fg is read descriptively as cross-framework
   mismatch in excess of the mean within-framework mismatch, not as an
   assumption-free variance decomposition. Estimator, CI, and decision
   rule are byte-unchanged. *Rationale:* the "assumption-light" framing
   overclaimed (R1 top issue 1).

7. **B−W normalization convention = what `analysis/decomposition.py`
   implements (stated, not changed).** Per item: W_i = mean over
   frameworks of the **ddof=1** (unbiased, denominator R−1) sample
   variance across the R runs; B_i = **ddof=1** (denominator F−1) sample
   variance across the F framework run-means **minus W_i/R** — the
   unbiased finite-run correction *under the ddof=1 convention with
   independent run noise* (with population normalization the correction
   would instead be (1−1/F)·W/R; we use ddof=1 throughout). Pooling =
   unweighted mean over items; negative B_i not truncated; B/W and B−W
   CIs from the same item bootstrap. B is dispersion among the five
   *selected* framework configurations, not a population variance
   component. *Rationale:* the correction's validity depends on the
   normalization convention, which was previously unstated (R1 top
   issue 2).

8. **Test-split error policy: errors-as-failures primary, NO test repair
   pass.** The §10 dev repair pass was dev-only. On the test split,
   terminal errors (after the runner's built-in retries) are kept as data
   — errored record = flagged — in every primary analysis; there is no
   post-hoc re-attempt of errored test ids. Complete-case is the
   preregistered secondary sensitivity (P4 table; repository JSON).
   *Rationale:* a dev-style repair pass on test would be a post-hoc,
   outcome-adjacent intervention (R1, Execution-Trail major issue 4);
   fixing the policy before data access removes the degree of freedom.
