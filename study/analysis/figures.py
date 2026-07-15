"""Publication figures (vector PDF), matching the style of
multivon-strategy/paper/figures/make_figures.py (serif, Okabe-Ito palette,
pcolormesh heatmaps kept vector). Deterministic: SOURCE_DATE_EPOCH pins the
PDF timestamp; no run-dependent state.

CANONICAL FILENAMES (see FIGURES.md): the six names written by make_all()
are the names the paper's \\pendingfigure includes adopt — they are frozen.
Specs follow the R5 visualization review (2026-07-14), applied while
results-blind:
  fig_kappa_condition_A.pdf   RAGTruth-Sum only, 1x2 judge panels,
                              Condition A; diagonal = per-framework flag
                              counts (Feinstein-Cicchetti antidote);
                              vmin/vmax (-0.2, 1.0) matches the pilot
                              figure. (_B twin = repo artifact.)
  fig_delta_forest.pdf        zero line, Delta<=0 shading (one-sided
                              rule), open markers = raw delta_fg, filled =
                              Delta_fg, rows sorted by Delta, third-party
                              (kill-switch) pairs marked with a dagger.
  fig_bw_summary.pdf          dot+CI (signed point estimates, not bars).
  fig_gateflip.pdf            panel (a) paired dots cross vs within,
                              panel (b) difference T with jackknife 95% CI
                              and <=0 shading (the H3 statistic).
  fig_f1_default_vs_tuned.pdf RAGTruth-Sum only, 1x2 judge panels,
                              dumbbells default->tuned with bootstrap CI
                              whiskers per dot.
  fig_error_rate.pdf          judge encoded by hatch/edge (never alpha:
                              grayscale/CVD-safe); raw counts annotated on
                              bars above the 10% warning line.
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
FW_SHORT = {"multivon-eval": "mve", "deepeval": "DE", "ragas": "RG",
            "trulens": "TL", "opik": "OP"}
# Okabe-Ito colorblind-safe palette
FW_COLOR = {"multivon-eval": "#0072B2", "deepeval": "#D55E00",
            "ragas": "#009E73", "trulens": "#CC79A7", "opik": "#E69F00"}
TASK_LABEL = {"ragtruth-sum": "RAGTruth-Sum", "halueval-sum": "HaluEval-Sum",
              "halueval-qa": "HaluEval-QA"}
PAPER_TASK = "ragtruth-sum"          # frozen captions: RAGTruth-Sum panels

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


def _paper_cells(entries):
    """Prefer the paper task's cells; fall back to whatever exists
    (synthetic selftest cells carry non-study task names)."""
    sub = [e for e in entries if e.get("task") == PAPER_TASK]
    return sub or entries


def _kappa_matrix(entry) -> np.ndarray:
    fws = entry["frameworks"]
    M = np.ones((len(fws), len(fws)))
    for key, k in entry["pairwise_kappa"].items():
        a, b = key.split(" <-> ")
        i, j = fws.index(a), fws.index(b)
        M[i, j] = M[j, i] = k
    return M


def fig_kappa(res, figdir: Path, cond: str) -> Path:
    """RAGTruth-Sum only, one panel per primary judge, Condition `cond`.
    Diagonal grayed out and repurposed as per-framework flag counts."""
    entries = [e for e in res["p1"]
               if e["condition"] == cond and e["task"] == PAPER_TASK]
    cmap = matplotlib.colormaps["cividis"].copy()
    cmap.set_bad("0.9")
    fig, axes = plt.subplots(1, len(entries),
                             figsize=(COL_W * 2.0, 2.9), squeeze=False)
    im = None
    for ax, e in zip(axes[0], entries):
        fws = e["frameworks"]
        F = len(fws)
        M = _kappa_matrix(e)
        Mm = M.copy()
        np.fill_diagonal(Mm, np.nan)
        im = ax.pcolormesh(np.arange(-0.5, F), np.arange(-0.5, F),
                           np.ma.masked_invalid(Mm),
                           vmin=-0.2, vmax=1.0, cmap=cmap,
                           edgecolors="white", lw=0.4)
        ax.set_ylim(F - 0.5, -0.5)
        ax.set_aspect("equal")
        n_items = e["n_items"]
        for i in range(F):
            for j in range(F):
                if i == j:
                    cnt = int(round(e["flag_rates"][fws[i]] * n_items))
                    ax.text(j, i, f"{cnt}", ha="center", va="center",
                            fontsize=6, color="0.25", style="italic")
                else:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center",
                            va="center", fontsize=6,
                            color="black" if M[i, j] > 0.55 else "white")
        names = [FW_LABEL[f] for f in fws]
        ax.set_xticks(range(F), names, rotation=40, ha="right",
                      fontsize=6.5)
        ax.set_yticks(range(F), names if ax is axes[0][0] else [""] * F,
                      fontsize=6.5)
        ax.set_title(e["judge"], fontsize=8)
        med, ci = e["median_pairwise_kappa"], e["median_kappa_ci95"]
        ax.set_xlabel(f"median $\\kappa$={med:.2f} "
                      f"[{ci[0]:.2f}, {ci[1]:.2f}]  (n={n_items})",
                      fontsize=6.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.solids.set_rasterized(False)
    cbar.set_label("Cohen's $\\kappa$", fontsize=8)
    axes[0][0].text(0.0, 1.13, "diagonal: flag count (italic)",
                    transform=axes[0][0].transAxes, fontsize=6, color="0.4")
    out = figdir / f"fig_kappa_condition_{cond}.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def _is_killswitch_pair(pair_key: str) -> bool:
    a, b = pair_key.split(" <-> ")
    return "multivon-eval" not in (a, b)


def fig_delta_forest(res, figdir: Path) -> Path | None:
    entries = [e for e in res["p2"] if "skipped" not in e]
    if not entries:
        return None
    entries = _paper_cells(entries)
    fig, axes = plt.subplots(1, len(entries),
                             figsize=(COL_W * max(len(entries), 1.6), 3.0),
                             sharex=True, squeeze=False)
    for ax, e in zip(axes[0], entries):
        # Rows sorted by Delta point estimate (forest-plot convention).
        pairs = sorted(e["pairs"], key=lambda p: e["pairs"][p]["Delta_fg"])
        y = np.arange(len(pairs))
        for yi, pk in zip(y, pairs):
            d = e["pairs"][pk]
            lo, hi = d["Delta_ci95_bca"]
            ax.plot([lo, hi], [yi, yi], color="0.2", lw=1.0, zorder=3)
            ax.plot(d["Delta_fg"], yi, "o", ms=3.5, color="#0072B2",
                    zorder=4)
            # open marker = raw cross-run disagreement delta_fg; the gap
            # to the filled Delta marker is the judge-noise floor.
            ax.plot(d["delta_fg"], yi, "o", ms=3.5, mfc="none",
                    mec="0.45", mew=0.8, zorder=4)
        labels = []
        for pk in pairs:
            lab = " vs ".join(FW_SHORT.get(s, s[:4])
                              for s in pk.split(" <-> "))
            if _is_killswitch_pair(pk):
                lab += " $\\dagger$"
            labels.append(lab)
        ax.set_yticks(y, labels, fontsize=6.5)
        ax.axvline(0, color="0.6", lw=0.6, ls="--", zorder=2)
        ax.set_xlabel("$\\Delta_{fg}$ (BCa 95% CI)")
        ax.set_title(f"{TASK_LABEL.get(e['task'], e['task'])} / "
                     f"{e['judge']} (R={e['R']}, n={e['n_items']})",
                     fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    # Shade the Delta <= 0 half-plane (one-sided decision rule) using the
    # shared x-limits, then restore them.
    xlo, xhi = axes[0][0].get_xlim()
    for ax in axes[0]:
        ax.axvspan(xlo, 0, color="0.93", zorder=0)
        ax.set_xlim(xlo, xhi)
    axes[0][0].text(0.01, -0.24,
                    "open marker: raw $\\delta_{fg}$;  filled: "
                    "$\\Delta_{fg}$;  $\\dagger$ third-party "
                    "(kill-switch) pair",
                    transform=axes[0][0].transAxes, fontsize=6,
                    color="0.35")
    out = figdir / "fig_delta_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_bw_summary(res, figdir: Path) -> Path | None:
    """Signed point estimates with CIs: dot+CI grammar, not bars."""
    entries = [e for e in res["p2"] if "skipped" not in e]
    if not entries:
        return None
    entries = _paper_cells(entries)
    rows = [("B", "B (between-fw)", "#D55E00"),
            ("W", "W (within-fw)", "#0072B2"),
            ("B_minus_W", "B $-$ W", "#009E73")]
    fig, ax = plt.subplots(figsize=(COL_W, 0.6 + 0.42 * 4 * len(entries)))
    ylabels, ypos, headers = [], [], []
    yi = 0
    for e in entries:
        headers.append((yi, f"{TASK_LABEL.get(e['task'], e['task'])} / "
                            f"{e['judge']} (R={e['R']})"))
        yi += 1
        bw = e["pooled_BW"]
        for key, lab, color in rows:
            lo, hi = bw[f"{key}_ci95"]
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.1, zorder=3)
            ax.plot(bw[key], yi, "o", ms=4, color=color, zorder=4)
            ylabels.append(lab)
            ypos.append(yi)
            yi += 1
        ratio = bw.get("B_over_W")
        note = f"B/W = {ratio}" if ratio is not None else "B/W: n/a (W=0)"
        ax.text(0.99, yi - 2, note, transform=ax.get_yaxis_transform(),
                fontsize=6, color="0.35", va="center", ha="right",
                zorder=5)
        yi += 0.4
    ax.axvline(0, color="0.4", lw=0.6, zorder=2)
    ax.set_yticks(ypos, ylabels, fontsize=6.5)
    ax.set_ylim(yi - 0.9, -0.6)
    for hy, lab in headers:
        ax.text(0.0, hy, lab, transform=ax.get_yaxis_transform(),
                fontsize=7, color="0.15", va="center", ha="left")
    ax.set_xlabel("variance component (95% CI; negatives not truncated)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    out = figdir / "fig_bw_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gateflip(res, figdir: Path) -> Path:
    """(a) paired dots: cross- vs within-framework flip probability;
    (b) the H3 statistic T = p_cross - p_within with jackknife 95% CI."""
    entries = _paper_cells(res["p3"])
    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(COL_W * 1.7, 2.6),
        gridspec_kw={"width_ratios": [1.4, 1.0]})
    x = np.arange(len(entries))
    for xi, e in zip(x, entries):
        cross = e["cross_flip_mc"]
        within = e.get("within_flip_mc")
        if within is not None:
            ax.plot([xi, xi], [within, cross], color="0.55", lw=0.9,
                    zorder=2)
            ax.plot(xi, within, "o", ms=5, mfc="none", mec="#0072B2",
                    mew=1.1, zorder=3)
        else:
            ax.text(xi, cross + 0.03, "R=1:\nwithin n/a", ha="center",
                    va="bottom", fontsize=5.5, color="0.35")
        ax.plot(xi, cross, "o", ms=5, color="#D55E00", zorder=3)
    ax.set_xticks(x, [f"{TASK_LABEL.get(e['task'], e['task'])}\n"
                      f"{e['judge']}" for e in entries], fontsize=6)
    ax.set_xlim(-0.6, len(entries) - 0.4)
    ax.set_ylim(0, None)
    ax.set_ylabel("P(gate outcome flips)")
    ax.plot([], [], "o", ms=5, color="#D55E00",
            label="cross-framework (same run)")
    ax.plot([], [], "o", ms=5, mfc="none", mec="#0072B2", mew=1.1,
            label="within-framework (across runs)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    have = [(xi, e["flip_difference"]) for xi, e in zip(x, entries)
            if "skipped" not in e["flip_difference"]]
    if have:
        for xi, d in have:
            lo, hi = d["ci95"]
            axd.plot([xi, xi], [lo, hi], color="0.2", lw=1.1, zorder=3)
            axd.plot(xi, d["T_cross_minus_within"], "o", ms=5,
                     color="#009E73", zorder=4)
        ylo, yhi = axd.get_ylim()
        ylo = min(ylo, -0.02)
        axd.axhspan(ylo, 0, color="0.93", zorder=0)
        axd.set_ylim(ylo, yhi)
    else:
        axd.text(0.5, 0.5, "H3 difference\nnot computable\n(R=1 cells)",
                 ha="center", va="center", transform=axd.transAxes,
                 fontsize=7, color="0.35")
    axd.axhline(0, color="0.6", lw=0.6, ls="--", zorder=2)
    axd.set_xticks(x, [f"{TASK_LABEL.get(e['task'], e['task'])}\n"
                       f"{e['judge']}" for e in entries], fontsize=6)
    axd.set_xlim(-0.6, len(entries) - 0.4)
    axd.set_ylabel("$T = p_{cross} - p_{within}$ (jackknife 95% CI)",
                   fontsize=7.5)
    axd.spines[["top", "right"]].set_visible(False)
    out = figdir / "fig_gateflip.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_f1_default_vs_tuned(res, figdir: Path) -> Path:
    """RAGTruth-Sum only, one panel per judge: dumbbell per framework from
    Condition A (default tau, open dot) to Condition B (locked tau, filled
    dot), each dot with its item-bootstrap 95% CI whisker."""
    entries = _paper_cells(res["p4"])
    fig, axes = plt.subplots(1, len(entries),
                             figsize=(COL_W * 2.0, 2.7),
                             sharey=True, squeeze=False)
    off = 0.16
    for ax, e in zip(axes[0], entries):
        fws = list(e["conditions"]["A"]["per_framework"])
        x = np.arange(len(fws))
        for i, fw in enumerate(fws):
            color = FW_COLOR.get(fw, "0.3")
            pts = {}
            for cond, dx in (("A", -off), ("B", off)):
                v = e["conditions"][cond]["per_framework"][fw][
                    "errors_as_failures"]
                f1 = v["f1"]
                lo, hi = v["f1_ci95"]
                xx = i + dx
                ax.plot([xx, xx], [lo, hi], color=color, lw=0.9,
                        alpha=0.9, zorder=2)
                pts[cond] = (xx, f1)
            ax.plot([pts["A"][0], pts["B"][0]], [pts["A"][1], pts["B"][1]],
                    color=color, lw=1.0, zorder=3)
            ax.plot(*pts["A"], "o", ms=4.5, mfc="white", mec=color,
                    mew=1.2, zorder=4)
            ax.plot(*pts["B"], "o", ms=4.5, color=color, zorder=4)
        ax.set_xticks(x, [FW_LABEL.get(f, f) for f in fws], rotation=40,
                      ha="right", fontsize=6.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{TASK_LABEL.get(e['task'], e['task'])} / "
                     f"{e['judge']}", fontsize=8)
        spA = e["conditions"]["A"]["f1_spread_max_minus_min"]
        ciA = e["conditions"]["A"]["f1_spread_ci95"]
        ax.text(0.02, 0.97, f"default-$\\tau$ spread {spA:.2f} "
                            f"[{ciA[0]:.2f},{ciA[1]:.2f}]",
                transform=ax.transAxes, va="top", fontsize=6, color="0.3")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("F1, hallucinated positive\n(errors-as-failures)",
                          fontsize=7.5)
    hA = plt.Line2D([], [], marker="o", ls="none", ms=4.5, mfc="white",
                    mec="0.3", mew=1.2)
    hB = plt.Line2D([], [], marker="o", ls="none", ms=4.5, color="0.3")
    fig.legend([hA, hB], ["Condition A (default $\\tau$)",
                          "Condition B (locked dev-fitted $\\tau$)"],
               ncols=2, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
    out = figdir / "fig_f1_default_vs_tuned.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_error_rate(res, figdir: Path) -> Path:
    """Judge encoded by hatch + edge (grayscale/CVD-safe), never alpha.
    Raw counts annotated on any bar above the 10% warning line."""
    entries = res["secondary"]["ops"]
    fig, axes = plt.subplots(1, len(TASKS), figsize=(COL_W * 2.05, 2.2),
                             sharey=True, squeeze=False)
    w = 0.38
    for c, task in enumerate(TASKS):
        ax = axes[0][c]
        x = np.arange(len(FRAMEWORKS))
        for off, judge, hatched in ((-w / 2, JUDGES[0], False),
                                    (w / 2, JUDGES[1], True)):
            e = _by(entries, judge=judge, task=task)[0]
            for i, fw in enumerate(FRAMEWORKS):
                o = e["frameworks"][fw]
                rate = o["api_error_rate"]
                color = FW_COLOR[fw]
                if hatched:
                    ax.bar(i + off, rate, w, facecolor="white",
                           edgecolor=color, hatch="////", lw=0.6)
                else:
                    ax.bar(i + off, rate, w, color=color)
                if rate >= 0.10:
                    ax.text(i + off, rate - 0.008,
                            f"{o['n_errors']}/{o['n_evals']}",
                            ha="center", va="top", fontsize=5.5,
                            color="0.25", rotation=90)
        ax.axhline(0.10, color="0.3", lw=0.7, ls="--")
        if c == 0:
            ax.text(-0.45, 0.108, "10% warning bar", ha="left",
                    va="bottom", fontsize=6, color="0.3")
        ax.set_xticks(x, [FW_LABEL[f] for f in FRAMEWORKS],
                      rotation=40, ha="right", fontsize=6.5)
        ax.set_title(TASK_LABEL[task], fontsize=8)
        if c == 0:
            ax.set_ylabel("terminal-error rate\n(errors-as-failures)",
                          fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="0.45"),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.45",
                             hatch="////", lw=0.6)]
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
