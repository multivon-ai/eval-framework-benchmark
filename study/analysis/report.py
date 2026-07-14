"""ANALYSIS.md — every preregistered number, labeled by endpoint id.

Endpoint ids follow plan §7: P1..P5 primary, S* secondary. Cell address
format: [Pn/<condition>/<task>/<judge>]. Generated file; no timestamps
(byte-deterministic double runs).
"""
from __future__ import annotations

from pathlib import Path

from .data import FRAMEWORKS, JUDGES, TASKS


def _ci(ci) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _p1_section(lines, entries, pid: str) -> None:
    for e in entries:
        tag = f"{pid}/{e['condition']}/{e['task']}/{e['judge']}"
        lines.append(f"### [{tag}]")
        lines.append("")
        lines.append(f"- **[{tag}] median pairwise κ = "
                     f"{e['median_pairwise_kappa']:.4f}**, 95% "
                     f"item-cluster bootstrap CI {_ci(e['median_kappa_ci95'])} "
                     f"(n={e['n_items']}, {len(e['pairwise_kappa'])} pairs, "
                     f"B={e['n_boot']})")
        bar = "MET" if e["h_bar_0.40_upper_lt"] else "NOT MET"
        lines.append(f"- H1/H4 bar (CI upper bound < 0.40): **{bar}**")
        lines.append(f"- median Gwet AC1 = {e['median_gwet_ac1']:.4f} (S3)")
        lines.append(f"- flag rates: " + ", ".join(
            f"{fw} {v:.3f}" for fw, v in e["flag_rates"].items()))
        lines.append("")
        lines.append("| pair | κ | raw disagreement | AC1 |")
        lines.append("|---|---|---|---|")
        for pair in sorted(e["pairwise_kappa"]):
            lines.append(f"| {pair} | {e['pairwise_kappa'][pair]:.4f} | "
                         f"{e['pairwise_raw_disagreement'][pair]:.4f} | "
                         f"{e['pairwise_gwet_ac1'][pair]:.4f} |")
        lines.append("")


def _p2_section(lines, entries, pid: str) -> None:
    for e in entries:
        tag = f"{pid}/A/{e['task']}/{e['judge']}"
        lines.append(f"### [{tag}]")
        lines.append("")
        if "skipped" in e:
            lines.append(f"- **SKIPPED** — {e['skipped']}")
            lines.append("")
            continue
        lines.append(f"- R={e['R']}, n={e['n_items']}, B={e['n_boot']}")
        lines.append(f"- H2 pairs with Δ>0 and BCa CI excluding 0: "
                     f"**{e['h2_pairs_positive_ci_excl0']}** "
                     f"(strict majority — ≥6/10 full set / ≥4/6 "
                     f"kill-switch set, addendum §12: "
                     f"{e['h2_majority_positive']})")
        fc = e["h2_falsification_c"]
        lines.append(f"- H2 falsification (c) (addendum §12.1): CIs "
                     f"including 0 = {fc['pairs_ci_including_0']} "
                     f"(majority: {fc['majority_include_0']}), pooled "
                     f"B ≤ W: {fc['pooled_B_le_W']} → fires: "
                     f"**{fc['fires']}**")
        bw = e["pooled_BW"]
        lines.append(f"- **[{tag}] pooled B−W = {bw['B_minus_W']:.4f}** "
                     f"{_ci(bw['B_minus_W_ci95'])}; B = {bw['B']:.4f} "
                     f"{_ci(bw['B_ci95'])}; W = {bw['W']:.4f} "
                     f"{_ci(bw['W_ci95'])}; B/W = {bw['B_over_W']} "
                     f"{_ci(bw['B_over_W_ci95'])} "
                     f"(negatives not truncated)")
        fl = e["per_item_flip"]
        lines.append(f"- per-item flip probability: across frameworks "
                     f"{fl['mean_across_frameworks']:.4f} vs within-framework "
                     f"across runs "
                     f"{fl['mean_within_framework_across_runs']:.4f}")
        lines.append("")
        lines.append("| pair | δ_fg | (π_f+π_g)/2 | Δ_fg | BCa 95% CI | excl 0 |")
        lines.append("|---|---|---|---|---|---|")
        for pair in sorted(e["pairs"]):
            d = e["pairs"][pair]
            lines.append(f"| {pair} | {d['delta_fg']:.4f} | "
                         f"{d['pi_mean']:.4f} | {d['Delta_fg']:.4f} | "
                         f"{_ci(d['Delta_ci95_bca'])} | "
                         f"{d['ci_excludes_0']} |")
        lines.append("")
        lines.append("| framework | π_f | 95% CI |")
        lines.append("|---|---|---|")
        for fw, v in e["pi_f"].items():
            lines.append(f"| {fw} | {v['estimate']:.4f} | {_ci(v['ci95'])} |")
        lines.append("")


def _p3_section(lines, entries, pid: str) -> None:
    for e in entries:
        tag = f"{pid}/A/{e['task']}/{e['judge']}"
        lines.append(f"### [{tag}]")
        lines.append("")
        lines.append(f"- endpoint: false-positive-batch gate-flip rate "
                     f"(addendum §12.5 relabel; construction unchanged)")
        lines.append(f"- gate: {e['gate']}; {e['n_batches']} batches of "
                     f"{e['batch']} from the faithful half "
                     f"(n={e['n_faithful']}); R={e['R']}")
        lines.append(f"- **[{tag}] cross-framework flip = "
                     f"{e['cross_flip_mc']:.4f}** (MC; closed-form "
                     f"{e['cross_flip_closed_form']:.4f})")
        d = e["flip_difference"]
        if "skipped" in d:
            lines.append(f"- within-framework flip / H3 difference: "
                         f"**SKIPPED** — {d['skipped']}")
        else:
            lines.append(f"- within-framework flip = "
                         f"{e['within_flip_mc']:.4f} (MC; closed-form "
                         f"{e['within_flip_closed_form']:.4f})")
            lines.append(f"- **[{tag}] flip difference T = "
                         f"{d['T_cross_minus_within']:.4f}** "
                         f"(MC {d['T_mc']:.4f}), SE_jack = "
                         f"{d['se_jackknife']:.4f}, 95% CI {_ci(d['ci95'])}, "
                         f"H3 CI excludes 0: **{d['h3_ci_excludes_0']}**")
        lines.append(f"- per-framework gate-fail probability: " + ", ".join(
            f"{fw} {v:.3f}" for fw, v in e["gate_fail_prob_mc"].items()))
        lines.append("")


def _p4_section(lines, entries, pid: str) -> None:
    for e in entries:
        for cond in ("A", "B"):
            tag = f"{pid}/{cond}/{e['task']}/{e['judge']}"
            c = e["conditions"][cond]
            lines.append(f"### [{tag}]")
            lines.append("")
            lines.append(f"- **[{tag}] F1 spread (max−min) = "
                         f"{c['f1_spread_max_minus_min']:.4f}**, 95% CI "
                         f"{_ci(c['f1_spread_ci95'])} (descriptive-with-CI "
                         f"per addendum §7; max {c['f1_max_framework']}, "
                         f"min {c['f1_min_framework']})")
            if cond == "A":
                lines.append(f"- falsification (b), per-judge arm "
                             f"(addendum §12.4; spread ≤ 0.05 at "
                             f"defaults): fires for this arm: "
                             f"**{c['falsification_b_fires_this_judge_arm']}**")
            lines.append("")
            lines.append("| framework | P (EaF) | R (EaF) | F1 (EaF) | "
                         "bal.acc | F1 (complete-case) | n_err |")
            lines.append("|---|---|---|---|---|---|---|")
            for fw, v in c["per_framework"].items():
                a, cc = v["errors_as_failures"], v["complete_case"]
                lines.append(f"| {fw} | {a['precision']:.4f} | "
                             f"{a['recall']:.4f} | {a['f1']:.4f} | "
                             f"{a['balanced_accuracy']:.4f} | "
                             f"{cc['f1']:.4f} (n={cc['n']}) | "
                             f"{v['n_errors']} |")
            lines.append("")


def _secondary(lines, res) -> None:
    sec = res["secondary"]
    lines.append("### [S2] κ_self vs κ_cross contrast (exploratory per "
                 "addendum §7)")
    lines.append("")
    for e in sec["kappa_self_cross"]:
        tag = f"S2/{e['task']}/{e['judge']}"
        if "skipped" in e:
            lines.append(f"- [{tag}] **SKIPPED** — {e['skipped']}")
            continue
        lines.append(f"- [{tag}] min κ_self − max κ_cross = "
                     f"{e['contrast_min_self_minus_max_cross']:.4f} "
                     f"{_ci(e['contrast_ci95'])}; κ_self: " + ", ".join(
                         f"{k} {v:.3f}" for k, v in e["kappa_self"].items()))
    lines.append("")
    lines.append("### [S5] prevalence-standardized F1 (0.10/0.25/0.50) — "
                 "see out/p4.json per_framework.prevalence_standardized")
    lines.append("")
    for e in res["p4"]:
        for fw in FRAMEWORKS:
            v = e["conditions"]["A"]["per_framework"][fw]
            ps = v["errors_as_failures"]["prevalence_standardized"]
            lines.append(f"- [S5/A/{e['task']}/{e['judge']}/{fw}] F1@prev: "
                         + ", ".join(f"{k.split('=')[1]} → {d['f1']:.4f}"
                                     for k, d in ps.items()))
    lines.append("")
    lines.append("### [S6] McNemar (exact, Holm within judge×task family)")
    lines.append("")
    for e in sec["mcnemar"]:
        sig = [p for p, v in e["pairs"].items() if v["significant_.05"]]
        lines.append(f"- [S6/{e['task']}/{e['judge']}] significant at "
                     f"Holm-.05: {len(sig)}/{len(e['pairs'])}"
                     + (f" — {'; '.join(sig)}" if sig else ""))
    lines.append("")
    lines.append("### [S7] pass@k / pass^k (multivon-eval 0.16.0 passk; "
                 "success = verdict matches gold)")
    lines.append("")
    for e in sec["passk"]:
        tag = f"S7/{e['task']}/{e['judge']}"
        if "skipped" in e:
            lines.append(f"- [{tag}] **SKIPPED** — {e['skipped']}")
            continue
        for fw, kv in e["frameworks"].items():
            top = f"pass^k k={e['R']}"
            lines.append(f"- [{tag}/{fw}] pass@1 = "
                         f"{kv['pass@k k=1']['value']:.4f} "
                         f"{_ci(kv['pass@k k=1']['ci95'])}; {top} = "
                         f"{kv[top]['value']:.4f} {_ci(kv[top]['ci95'])}")
    lines.append("")
    lines.append("### [S8] per-framework ops (api_error_rate / cost / "
                 "latency; pooled over runs)")
    lines.append("")
    lines.append("| task | judge | framework | error rate | ≥10% warn | "
                 "mean cost $ | mean latency ms | mean judge calls |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in sec["ops"]:
        for fw, o in e["frameworks"].items():
            lines.append(f"| {e['task']} | {e['judge']} | {fw} | "
                         f"{o['api_error_rate']:.4f} | "
                         f"{'**YES**' if o['error_rate_warning_ge_10pct'] else 'no'} | "
                         f"{o['mean_cost_usd']:.6f} | "
                         f"{o['mean_latency_ms']:.0f} | "
                         f"{o['mean_judge_calls']:.2f} |")
    lines.append("")


def write_markdown(res, figures: list[Path], path: Path) -> None:
    m = res["meta"]
    lines = [
        "# Preregistered analysis — cross-framework disagreement study",
        "",
        f"Split: **{m['split']}** | bootstrap resamples: {m['n_boot']} | "
        f"seed: {m['seed']} | generated by study/analysis/run_all.py "
        "(deterministic; no timestamps)",
        "",
        "Conditions: A = " + m["conditions"]["A"] + "; B = "
        + m["conditions"]["B"] + ". " + m["error_semantics"] + ".",
        "",
        "Confirmatory scope (plan §9, addendum §7): H1/H4 median-κ bound and "
        "H3 flip difference are confirmatory on RAGTruth-Sum × both primary "
        "judges on the TEST split only; every dev-split number below is "
        "pipeline validation, not evidence." if m["split"] == "dev" else
        "Confirmatory endpoints evaluated on the test split per plan §9.",
        "",
        "Runs per cell: " + ", ".join(f"{k}: R={v}"
                                      for k, v in m["runs_per_cell"].items()),
        "",
        "## P1 — pairwise Cohen's κ + median-κ CI (H1; Condition B = H4)",
        "",
    ]
    _p1_section(lines, res["p1"], "P1")
    lines += ["## P2 — Δ_fg decomposition + pooled B−W (H2; repeated cells "
              "only)", ""]
    _p2_section(lines, res["p2"], "P2")
    lines += ["## P3 — false-positive-batch gate-flip rate (H3; "
              "addendum §12.5 relabel, construction unchanged)", ""]
    _p3_section(lines, res["p3"], "P3")
    lines += ["## P4 — default-τ precision/recall/F1 vs gold + F1 spread", ""]
    _p4_section(lines, res["p4"], "P4")

    lines += ["## P5 — kill-switch: P1–P4 excluding multivon-eval",
              "",
              f"Frameworks: {', '.join(res['p5']['frameworks'])} — H2 "
              "pair bar on this reduced set: ≥4/6 pairs (strict majority; "
              "addendum §12.2)", ""]
    _p1_section(lines, res["p5"]["p1"], "P5.1")
    _p2_section(lines, res["p5"]["p2"], "P5.2")
    _p3_section(lines, res["p5"]["p3"], "P5.3")
    _p4_section(lines, res["p5"]["p4"], "P5.4")

    lines += ["## Secondary endpoints", ""]
    _secondary(lines, res)

    lines += ["## Figures", ""]
    lines += [f"- figures/{f.name}" for f in figures]
    if not any("delta_forest" in f.name for f in figures):
        lines.append("- fig_delta_forest.pdf / fig_bw_summary.pdf: not "
                     "rendered (no repeated cells on this split)")
    lines += ["", "## SKIPPED (with reasons)", ""]
    lines += [f"- {s}" for s in res["skipped"]] or ["- none"]
    lines.append("")
    path.write_text("\n".join(lines))
