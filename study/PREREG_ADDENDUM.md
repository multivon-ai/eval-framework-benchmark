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

## 8. Smoke run + measured call multipliers — **NOT RUN**

Reserved for the measured framework×judge multiplier table and the re-issued
cost projection (plan §11). The Day-1 closeout instruction conditioned the
smoke run on the §7 refined gate passing; it failed, so no smoke API call
was made and no multiplier is recorded. (A prepared, unexecuted driver
exists as an uncommitted local file; it will be committed with this section
if and when a go decision is recorded.)

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
metadata during the Day-1 smoke run.
