# Canonical figure manifest (frozen pre-unblind, 2026-07-15)

The paper (`multivon-strategy/paper/main.tex`) adopts the GENERATOR's
filenames for its six `\pendingfigure` includes — the six names below are
therefore **canonical and stable**: `study/analysis/figures.py` must keep
writing exactly these names into `study/analysis/figures/`, and the paper
build copies them verbatim. Any rename is a freeze deviation and goes in
`DEVIATIONS.md`. Specs implement the R5 visualization review (2026-07-14),
applied while results-blind; produced by
`python -m study.analysis.run_all --split {dev|test}` (deterministic:
fixed seeds, `SOURCE_DATE_EPOCH=0`).

| # | Canonical filename | Content spec |
|---|---|---|
| 1 | `fig_kappa_condition_A.pdf` | P1 pairwise Cohen's κ, **RAGTruth-Sum only, 1×2 panels (gpt-4o-mini \| claude-haiku-4-5), Condition A**; 5×5 matrices, vmin/vmax = (−0.2, 1.0) matching the pilot figure; diagonal grayed and repurposed as per-framework flag counts; median-κ + 95% CI under each panel. |
| 2 | `fig_delta_forest.pdf` | P2 Δ_fg forest, repeated RAGTruth-Sum cells, one panel per judge; dashed zero line; **Δ ≤ 0 half-plane shaded** (one-sided rule); **open marker = raw δ_fg**, filled = Δ_fg (gap = judge-noise floor); rows sorted by Δ; **third-party (kill-switch) pairs marked †**. |
| 3 | `fig_bw_summary.pdf` | P2 pooled B / W / B−W as **dot + 95% CI** (signed estimates — never bars); zero line; negatives not truncated; B/W annotated (guarded for W = 0). |
| 4 | `fig_gateflip.pdf` | P3 false-positive-batch gate-flip rate: panel (a) **paired dots** cross (filled, vermillion) vs within (open, blue) connected per cell; panel (b) **difference T = p_cross − p_within with jackknife 95% CI**, zero line, ≤ 0 shading (the H3 statistic). |
| 5 | `fig_f1_default_vs_tuned.pdf` | P4, **RAGTruth-Sum only, 1×2 judge panels, dumbbells** per framework from Condition A (default τ, open dot) to Condition B (locked τ, filled dot), **bootstrap 95% CI whisker on every dot**; default-τ spread + CI annotated; errors-as-failures stated on the axis. |
| 6 | `fig_error_rate.pdf` | S8 terminal-error rate, all three tasks (1×3); 10% warning line; **judge encoded by hatch/edge, never alpha** (grayscale/CVD-safe); raw counts (`errors/evals`) annotated on bars above 10%. |

Repo-only twin (not a paper include): `fig_kappa_condition_B.pdf` — same
spec as #1 under Condition B (H4 has no figure in the frozen manuscript).

Repeated-cell figures (#2, #3, and panel (b) of #4) render fully only on
splits with R ≥ 2; on dev (R=1) #2/#3 are skipped with a printed reason
and #4 renders cross-framework dots with the difference panel annotated
"not computable (R=1)".
