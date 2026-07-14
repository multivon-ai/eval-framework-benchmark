"""Publication figures (vector PDF), matching the style of
multivon-strategy/paper/figures/make_figures.py (serif, Okabe-Ito palette,
pcolormesh heatmaps kept vector). Deterministic: SOURCE_DATE_EPOCH pins the
PDF timestamp; no run-dependent state.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import FRAMEWORKS, JUDGES, TASKS  # noqa: E402

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

FW_LABEL = {"multivon-eval": "multivon-eval", "deepeval": "DeepEval",
            "ragas": "RAGAS", "trulens": "TruLens", "opik": "Opik"}
# Okabe-Ito colorblind-safe palette
FW_COLOR = {"multivon-eval": "#0072B2", "deepeval": "#D55E00",
            "ragas": "#009E73", "trulens": "#CC79A7", "opik": "#E69F00"}
TASK_LABEL = {"ragtruth-sum": "RAGTruth-Sum", "halueval-sum": "HaluEval-Sum",
              "halueval-qa": "HaluEval-QA"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.linewidth": 0.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL_W = 3.4


def _by(entries, **want):
    out = [e for e in entries
           if all(e.get(k) == v for k, v in want.items())]
    return out


def _kappa_matrix(entry) -> np.ndarray:
    fws = entry["frameworks"]
    M = np.ones((len(fws), len(fws)))
    for key, k in entry["pairwise_kappa"].items():
        a, b = key.split(" <-> ")
        i, j = fws.index(a), fws.index(b)
        M[i, j] = M[j, i] = k
    return M


def fig_kappa(res, figdir: Path, cond: str) -> Path:
    entries = _by(res["p1"], condition=cond)
    fig, axes = plt.subplots(len(JUDGES), len(TASKS),
                             figsize=(COL_W * 2.05, 4.4))
    im = None
    for r, judge in enumerate(JUDGES):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            e = _by(entries, judge=judge, task=task)[0]
            M = _kappa_matrix(e)
            F = len(e["frameworks"])
            im = ax.pcolormesh(np.arange(-0.5, F), np.arange(-0.5, F), M,
                               vmin=-0.2, vmax=1.0, cmap="cividis",
                               edgecolors="white", lw=0.4)
            ax.set_ylim(F - 0.5, -0.5)
            ax.set_aspect("equal")
            for i in range(F):
                for j in range(F):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="black" if M[i, j] > 0.55 else "white")
            names = [FW_LABEL[f] for f in e["frameworks"]]
            ax.set_xticks(range(F), names if r == len(JUDGES) - 1
                          else [""] * F, rotation=40, ha="right", fontsize=6.5)
            ax.set_yticks(range(F), names if c == 0 else [""] * F,
                          fontsize=6.5)
            if r == 0:
                ax.set_title(TASK_LABEL[task], fontsize=8)
            if c == 0:
                ax.text(-0.62, 0.5, judge, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=8)
            med, ci = e["median_pairwise_kappa"], e["median_kappa_ci95"]
            ax.set_xlabel(f"median $\\kappa$={med:.2f} "
                          f"[{ci[0]:.2f}, {ci[1]:.2f}]", fontsize=6.5)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.solids.set_rasterized(False)
    cbar.set_label("Cohen's $\\kappa$", fontsize=8)
    split = res["meta"]["split"]
    fig.suptitle(f"P1 pairwise Cohen's $\\kappa$ — Condition {cond} "
                 f"({split} split, run 0)", fontsize=9, y=0.99)
    out = figdir / f"fig_kappa_condition_{cond}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_delta_forest(res, figdir: Path) -> Path | None:
    entries = [e for e in res["p2"] if "skipped" not in e]
    if not entries:
        return None
    fig, axes = plt.subplots(1, len(entries),
                             figsize=(COL_W * len(entries), 3.0),
                             sharex=True, squeeze=False)
    for ax, e in zip(axes[0], entries):
        pairs = sorted(e["pairs"])
        y = np.arange(len(pairs))[::-1]
        for yi, pk in zip(y, pairs):
            d = e["pairs"][pk]
            lo, hi = d["Delta_ci95_bca"]
            ax.plot([lo, hi], [yi, yi], color="0.2", lw=1.0)
            ax.plot(d["Delta_fg"], yi, "o", ms=3.5, color="#0072B2")
        ax.axvline(0, color="0.6", lw=0.6, ls="--")
        short = {f: FW_LABEL[f][:2] for f in FRAMEWORKS}
        labels = [" vs ".join(short.get(s, s[:4]) for s in p.split(" <-> "))
                  for p in pairs]
        ax.set_yticks(y, labels, fontsize=6.5)
        ax.set_xlabel("$\\Delta_{fg}$ (BCa 95% CI)")
        ax.set_title(f"{TASK_LABEL[e['task']]} / {e['judge']} (R={e['R']})",
                     fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    out = figdir / "fig_delta_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_bw_summary(res, figdir: Path) -> Path | None:
    entries = [e for e in res["p2"] if "skipped" not in e]
    if not entries:
        return None
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    x = np.arange(len(entries))
    w = 0.27
    for off, key, color in ((-w, "B", "#D55E00"), (0, "W", "#0072B2"),
                            (w, "B_minus_W", "#009E73")):
        vals = [e["pooled_BW"][key] for e in entries]
        cis = np.array([e["pooled_BW"][f"{key}_ci95"] for e in entries])
        err = np.abs(cis.T - np.array(vals))
        ax.bar(x + off, vals, w * 0.92, color=color,
               yerr=err, error_kw=dict(lw=0.8, capsize=2),
               label={"B": "B (between-fw)", "W": "W (within-fw)",
                      "B_minus_W": "B $-$ W"}[key])
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(x, [f"{TASK_LABEL[e['task']]}\n{e['judge']}"
                      for e in entries], fontsize=7)
    ax.set_ylabel("Variance component")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    out = figdir / "fig_bw_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gateflip(res, figdir: Path) -> Path:
    entries = res["p3"]
    fig, ax = plt.subplots(figsize=(COL_W * 1.6, 2.6))
    x = np.arange(len(entries))
    w = 0.36
    cross = [e["cross_flip_mc"] for e in entries]
    within = [e.get("within_flip_mc") for e in entries]
    ax.bar(x - w / 2, cross, w, color="#D55E00",
           label="Cross-framework (same run)")
    have_w = [v if v is not None else 0.0 for v in within]
    ax.bar(x + w / 2, have_w, w, color="#0072B2",
           label="Within-framework (across runs)")
    for xi, v in zip(x, within):
        if v is None:
            ax.text(xi + w / 2, 0.004, "R=1\nn/a", ha="center", va="bottom",
                    fontsize=6, color="0.35")
    for xi, v in zip(x, cross):
        ax.text(xi - w / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x, [f"{TASK_LABEL[e['task']]}\n{e['judge']}"
                      for e in entries], fontsize=6.5)
    ax.set_ylabel("P(gate outcome flips)")
    ax.set_title("P3 CI-gate flip probability (batches of 50, "
                 "faithful half, gate: flag rate > 20%)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    out = figdir / "fig_gateflip.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_f1_default_vs_tuned(res, figdir: Path) -> Path:
    fig, axes = plt.subplots(len(JUDGES), len(TASKS),
                             figsize=(COL_W * 2.05, 4.0),
                             sharey=True, squeeze=False)
    for r, judge in enumerate(JUDGES):
        for c, task in enumerate(TASKS):
            ax = axes[r][c]
            e = _by(res["p4"], judge=judge, task=task)[0]
            x = np.arange(len(FRAMEWORKS))
            w = 0.38
            for off, cond, alpha, hatch in ((-w / 2, "A", 1.0, None),
                                            (w / 2, "B", 0.45, "///")):
                vals = [e["conditions"][cond]["per_framework"][fw]
                        ["errors_as_failures"]["f1"] for fw in FRAMEWORKS]
                ax.bar(x + off, vals, w,
                       color=[FW_COLOR[f] for f in FRAMEWORKS],
                       alpha=alpha, hatch=hatch,
                       edgecolor="white" if hatch else None, lw=0)
            ax.set_xticks(x, [FW_LABEL[f] for f in FRAMEWORKS]
                          if r == len(JUDGES) - 1 else [""] * len(FRAMEWORKS),
                          rotation=40, ha="right", fontsize=6.5)
            ax.set_ylim(0, 1.0)
            if r == 0:
                ax.set_title(TASK_LABEL[task], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{judge}\nF1 (hallucinated)", fontsize=7.5)
            sp = e["conditions"]["A"]["f1_spread_max_minus_min"]
            ci = e["conditions"]["A"]["f1_spread_ci95"]
            ax.text(0.02, 0.97, f"A spread {sp:.2f} [{ci[0]:.2f},{ci[1]:.2f}]",
                    transform=ax.transAxes, va="top", fontsize=6, color="0.3")
            ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="0.35"),
               plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=0.45,
                             hatch="///", ec="white", lw=0)]
    fig.legend(handles, ["Condition A (default $\\tau$)",
                         "Condition B (locked dev-fitted $\\tau$)"],
               ncols=2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    out = figdir / "fig_f1_default_vs_tuned.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_error_rate(res, figdir: Path) -> Path:
    entries = res["secondary"]["ops"]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(COL_W * 2.05, 2.2),
                             sharey=True, squeeze=False)
    for c, task in enumerate(TASKS):
        ax = axes[0][c]
        x = np.arange(len(FRAMEWORKS))
        w = 0.38
        for off, judge, alpha in ((-w / 2, JUDGES[0], 1.0),
                                  (w / 2, JUDGES[1], 0.5)):
            e = _by(entries, judge=judge, task=task)[0]
            vals = [e["frameworks"][fw]["api_error_rate"]
                    for fw in FRAMEWORKS]
            ax.bar(x + off, vals, w, color=[FW_COLOR[f] for f in FRAMEWORKS],
                   alpha=alpha)
        ax.axhline(0.10, color="0.3", lw=0.7, ls="--")
        ax.text(len(FRAMEWORKS) - 0.5, 0.102, "10% warning bar",
                ha="right", fontsize=6, color="0.3")
        ax.set_xticks(x, [FW_LABEL[f] for f in FRAMEWORKS],
                      rotation=40, ha="right", fontsize=6.5)
        ax.set_title(TASK_LABEL[task], fontsize=8)
        if c == 0:
            ax.set_ylabel("api_error_rate")
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="0.35", alpha=a)
               for a in (1.0, 0.5)]
    fig.legend(handles, JUDGES, ncols=2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5)
    out = figdir / "fig_error_rate.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all(res, figdir: Path) -> list[Path]:
    written = [fig_kappa(res, figdir, "A"), fig_kappa(res, figdir, "B")]
    for fn in (fig_delta_forest, fig_bw_summary):
        p = fn(res, figdir)
        if p is not None:
            written.append(p)
    written += [fig_gateflip(res, figdir),
                fig_f1_default_vs_tuned(res, figdir),
                fig_error_rate(res, figdir)]
    return written
