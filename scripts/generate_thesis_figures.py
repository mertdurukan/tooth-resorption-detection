"""Generate the four core thesis figures (A, C, D, E).

A: Baseline CNN vs ViT-Base/16 — Macro-F1 + Accuracy bar chart.
C: Speed vs Performance scatter (Pareto view).
D: Class-wise Macro-F1 heatmap across all models.
E: Selective prediction curve (ensemble high-confidence thresholds).

Reads CSV/JSON files in ``results/metrics`` and writes PNGs into
``results/plots``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
TITLE_FS = 13
LABEL_FS = 11


# --------------------------------------------------------------------------- #
# A — Baseline CNN vs ViT-Base/16
# --------------------------------------------------------------------------- #
def figure_a(perf: pd.DataFrame) -> Path:
    rows = perf.set_index("Model").loc[["CNN_Classification", "vit_base_16"]]
    metrics = ["F1-Score", "Accuracy"]
    labels = ["Macro-F1", "Accuracy"]

    baseline = rows.loc["CNN_Classification", metrics].to_numpy(dtype=float)
    vit = rows.loc["vit_base_16", metrics].to_numpy(dtype=float)

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    b1 = ax.bar(x - width / 2, baseline, width,
                label="Baseline CNN", color="#bd6760", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width / 2, vit, width,
                label="ViT-Base/16", color="#3b7dbf", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=LABEL_FS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=LABEL_FS)
    ax.set_title("Baseline CNN vs ViT-Base/16", fontsize=TITLE_FS, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    delta_f1 = vit[0] - baseline[0]
    delta_acc = vit[1] - baseline[1]
    ax.text(0.99, 0.97,
            f"Δ Macro-F1 = +{delta_f1:.3f}\nΔ Accuracy = +{delta_acc:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="#f5f5f5", edgecolor="#999"))

    fig.tight_layout()
    out = PLOTS_DIR / "A_baseline_vs_vit_b16.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# C — Speed vs Performance scatter (size = model size MB)
# --------------------------------------------------------------------------- #
def figure_c(perf: pd.DataFrame) -> Path:
    df = perf.copy()
    df = df[df["Composite"] > 0]

    color_map = {
        "CNN": "#bd6760",
        "Attention/Transformer": "#3b7dbf",
        "YOLO": "#6aaa64",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for cat, grp in df.groupby("Category"):
        ax.scatter(
            grp["Inference(ms)"], grp["Composite"],
            s=grp["Size(MB)"] * 1.2 + 30,
            c=color_map.get(cat, "#888"),
            alpha=0.75, edgecolors="black", linewidths=0.7,
            label=cat,
        )

    for _, row in df.iterrows():
        dy = 0.012
        dx = 0.6
        ax.annotate(row["Model"],
                    (row["Inference(ms)"] + dx, row["Composite"] + dy),
                    fontsize=8, color="#222")

    ax.set_xlabel("Inference time (ms, lower is better)", fontsize=LABEL_FS)
    ax.set_ylabel("Composite Score (higher is better)", fontsize=LABEL_FS)
    ax.set_title("Speed vs Performance — bubble size ∝ model size (MB)",
                 fontsize=TITLE_FS, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    vit = df[df["Model"] == "vit_base_16"].iloc[0]
    ax.annotate("Best speed/perf\ntrade-off",
                xy=(vit["Inference(ms)"], vit["Composite"]),
                xytext=(vit["Inference(ms)"] + 25, vit["Composite"] - 0.06),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="#222", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8d1",
                          edgecolor="#999"))

    fig.tight_layout()
    out = PLOTS_DIR / "C_speed_vs_performance.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# D — Class-wise F1 heatmap
# --------------------------------------------------------------------------- #
def figure_d(classwise: pd.DataFrame, perf: pd.DataFrame) -> Path:
    pivot = classwise.pivot(index="Model", columns="Class", values="F1-Score")
    pivot = pivot[["Temaslı", "Bağımsız", "Rezorpsiyon"]]

    order = (perf.set_index("Model")
                  .loc[pivot.index, "Composite"]
                  .sort_values(ascending=False).index)
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(pivot) + 1.6))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   aspect="auto")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=LABEL_FS)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.35 < v < 0.85 else "white",
                    fontsize=9)

    ax.set_title("Class-wise F1 — models ranked by composite score",
                 fontsize=TITLE_FS, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("F1-Score", fontsize=10)

    fig.tight_layout()
    out = PLOTS_DIR / "D_classwise_f1_heatmap.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# E — Selective prediction curve
# --------------------------------------------------------------------------- #
def figure_e(hc: dict) -> Path:
    rows = hc["threshold_results"]
    th = np.array([r["threshold"] for r in rows])
    acc = np.array([r["accuracy"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    f1 = np.array([r["f1"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    color_acc = "#1f77b4"
    color_cov = "#d62728"

    l1 = ax1.plot(th, acc, "-o", color=color_acc, lw=2.0, label="Accuracy")
    l2 = ax1.plot(th, f1, "--s", color="#2ca02c", lw=1.6, label="F1")
    ax1.set_xlabel("Confidence threshold", fontsize=LABEL_FS)
    ax1.set_ylabel("Accuracy / F1", color=color_acc, fontsize=LABEL_FS)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0.95, 1.005)
    ax1.grid(linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    l3 = ax2.plot(th, cov, "-^", color=color_cov, lw=2.0, label="Coverage (%)")
    ax2.set_ylabel("Coverage (%)", color=color_cov, fontsize=LABEL_FS)
    ax2.tick_params(axis="y", labelcolor=color_cov)
    ax2.set_ylim(0, 105)

    rec = hc["best_recommendation"]
    ax1.axvline(rec["threshold"], color="#555", ls=":", lw=1.2)
    ax1.annotate(
        f"Recommended\nτ = {rec['threshold']}\nAcc = {rec['accuracy']:.2f}\n"
        f"Cov = {rec['coverage']:.1f}%",
        xy=(rec["threshold"], rec["accuracy"]),
        xytext=(rec["threshold"] - 0.32, 0.965),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8d1", edgecolor="#999"),
        arrowprops=dict(arrowstyle="->", color="#222"),
    )

    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines],
               loc="lower left", frameon=True)
    ax1.set_title(
        f"Selective prediction — ensemble of {', '.join(hc['ensemble_models'])}",
        fontsize=TITLE_FS, fontweight="bold")

    fig.tight_layout()
    out = PLOTS_DIR / "E_selective_prediction.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# LinkedIn — focused Baseline CNN vs ViT-Base/16 scatter
# --------------------------------------------------------------------------- #
def figure_linkedin(perf: pd.DataFrame) -> Path:
    """Editorial-style LinkedIn card (1200x1200) — Baseline CNN vs ViT-Base/16."""
    df = perf.copy()
    df = df[df["Composite"] > 0].reset_index(drop=True)

    base = df[df["Model"] == "CNN_Classification"].iloc[0]
    vit = df[df["Model"] == "vit_base_16"].iloc[0]
    speedup = base["Inference(ms)"] / vit["Inference(ms)"]
    delta_f1 = vit["F1-Score"] - base["F1-Score"]

    INK = "#0f1b2d"
    SUBINK = "#5a6477"
    MUTED = "#c4cad6"
    BG = "#f6f3ec"
    PANEL = "#ffffff"
    ACCENT_BLUE = "#1f4ea1"
    ACCENT_BLUE_SOFT = "#dbe5f5"
    ACCENT_RED = "#a8324a"
    ACCENT_RED_SOFT = "#f1d7dd"
    GOLD = "#b58900"

    fig = plt.figure(figsize=(10, 10), dpi=120)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[0.95, 0.55, 4.4, 0.35],
        hspace=0.45,
        left=0.085, right=0.93, top=0.955, bottom=0.045,
    )

    ax_title = fig.add_subplot(gs[0])
    ax_kpi = fig.add_subplot(gs[1])
    ax = fig.add_subplot(gs[2])
    ax_foot = fig.add_subplot(gs[3])

    for a in (ax_title, ax_kpi, ax_foot):
        a.set_facecolor(BG)
        a.axis("off")

    # ---- Title band -----------------------------------------------------
    ax_title.text(0.0, 0.92, "BENCHMARK  //  TOOTH RESORPTION DETECTION",
                  fontsize=10.5, color=SUBINK, fontweight="bold",
                  family="DejaVu Sans Mono", transform=ax_title.transAxes,
                  va="top")
    ax_title.text(0.0, 0.62, "Baseline CNN  →  ViT-Base/16",
                  fontsize=26, color=INK, fontweight="bold",
                  transform=ax_title.transAxes, va="top")
    ax_title.text(0.0, 0.10,
                  "Same data, same protocol — different inductive bias.",
                  fontsize=12, color=SUBINK, style="italic",
                  transform=ax_title.transAxes, va="top")
    ax_title.plot([0.0, 1.0], [0.0, 0.0], color=INK, lw=1.0,
                  transform=ax_title.transAxes, clip_on=False)

    # ---- KPI strip ------------------------------------------------------
    kpis = [
        ("MACRO-F1", f"+{delta_f1:.2f}", "0.199 → 0.898", ACCENT_BLUE),
        ("INFERENCE", f"{speedup:.1f}×", "85.0 → 7.44 ms", ACCENT_RED),
        ("PARAMETERS", "85.8 M", "ImageNet pretrained", INK),
    ]
    for i, (label, big, sub, col) in enumerate(kpis):
        x0 = i / 3 + 0.005
        x1 = (i + 1) / 3 - 0.005
        ax_kpi.add_patch(plt.Rectangle(
            (x0, 0.05), x1 - x0, 0.9, transform=ax_kpi.transAxes,
            facecolor=PANEL, edgecolor="#dcd6c8", linewidth=0.8))
        ax_kpi.text(x0 + 0.02, 0.78, label,
                    fontsize=9, color=SUBINK, fontweight="bold",
                    family="DejaVu Sans Mono",
                    transform=ax_kpi.transAxes, va="top")
        ax_kpi.text(x0 + 0.02, 0.55, big,
                    fontsize=22, color=col, fontweight="bold",
                    transform=ax_kpi.transAxes, va="top")
        ax_kpi.text(x0 + 0.02, 0.18, sub,
                    fontsize=9.5, color=SUBINK,
                    transform=ax_kpi.transAxes, va="top")

    # ---- Main scatter ---------------------------------------------------
    ax.set_facecolor(PANEL)
    others = df[~df["Model"].isin(["CNN_Classification", "vit_base_16"])]

    ax.scatter(others["Inference(ms)"], others["F1-Score"],
               s=others["Size(MB)"] * 0.9 + 35,
               c=MUTED, alpha=0.7,
               edgecolors="#9aa3b2", linewidths=0.5, zorder=3)

    ax.scatter(base["Inference(ms)"], base["F1-Score"],
               s=base["Size(MB)"] * 1.6 + 320,
               c=ACCENT_RED_SOFT, edgecolors=ACCENT_RED, linewidths=2.0,
               zorder=5)
    ax.scatter(base["Inference(ms)"], base["F1-Score"],
               s=60, c=ACCENT_RED, zorder=6)

    ax.scatter(vit["Inference(ms)"], vit["F1-Score"],
               s=vit["Size(MB)"] * 1.6 + 320,
               c=ACCENT_BLUE_SOFT, edgecolors=ACCENT_BLUE, linewidths=2.0,
               zorder=5)
    ax.scatter(vit["Inference(ms)"], vit["F1-Score"],
               s=60, c=ACCENT_BLUE, zorder=6)

    ax.annotate(
        "", xy=(vit["Inference(ms)"] + 0.5, vit["F1-Score"] - 0.01),
        xytext=(base["Inference(ms)"] - 0.5, base["F1-Score"] + 0.01),
        arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.45",
                        color=ACCENT_BLUE, lw=2.4,
                        connectionstyle="arc3,rad=-0.32"),
        zorder=4)

    ax.annotate(
        "Baseline CNN",
        xy=(base["Inference(ms)"], base["F1-Score"]),
        xytext=(base["Inference(ms)"] - 6, base["F1-Score"] + 0.085),
        fontsize=11.5, fontweight="bold", color=ACCENT_RED, ha="right",
        arrowprops=dict(arrowstyle="-", color=ACCENT_RED, lw=1.0),
        zorder=7)
    ax.text(base["Inference(ms)"] - 6, base["F1-Score"] + 0.045,
            f"F1 {base['F1-Score']:.3f}  ·  {base['Inference(ms)']:.1f} ms",
            fontsize=9.5, color=SUBINK, ha="right", zorder=7)

    ax.annotate(
        "ViT-Base/16",
        xy=(vit["Inference(ms)"], vit["F1-Score"]),
        xytext=(vit["Inference(ms)"] + 9, vit["F1-Score"] + 0.03),
        fontsize=12.5, fontweight="bold", color=ACCENT_BLUE, ha="left",
        arrowprops=dict(arrowstyle="-", color=ACCENT_BLUE, lw=1.0),
        zorder=7)
    ax.text(vit["Inference(ms)"] + 9, vit["F1-Score"] - 0.005,
            f"F1 {vit['F1-Score']:.3f}  ·  {vit['Inference(ms)']:.2f} ms",
            fontsize=10, color=SUBINK, ha="left", zorder=7)
    ax.text(vit["Inference(ms)"] + 9, vit["F1-Score"] - 0.04,
            "best speed × performance",
            fontsize=9, color=ACCENT_BLUE, ha="left", style="italic",
            zorder=7)

    callout_x = (base["Inference(ms)"] + vit["Inference(ms)"]) / 2 - 6
    callout_y = (base["F1-Score"] + vit["F1-Score"]) / 2 + 0.10
    ax.text(callout_x, callout_y,
            f"+{delta_f1:.2f}  Macro-F1\n{speedup:.1f}×  faster",
            fontsize=15, fontweight="bold", color=INK, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#fff5d6",
                      edgecolor=GOLD, linewidth=1.3),
            zorder=8)

    ax.text(0.015, 0.04,
            f"n = {len(others)} additional architectures benchmarked",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=SUBINK, style="italic",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG,
                      edgecolor="#dcd6c8", linewidth=0.8))

    ax.set_xlabel("Inference time per image (ms)   ←   lower is better",
                  fontsize=11, color=INK, fontweight="bold", labelpad=10)
    ax.set_ylabel("Macro-F1   →   higher is better",
                  fontsize=11, color=INK, fontweight="bold", labelpad=10)

    ax.grid(linestyle=":", color="#dcd6c8", alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9aa3b2")
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=SUBINK, labelsize=10)

    ax.set_xlim(-3, 95)
    ax.set_ylim(0.10, 1.00)

    # ---- Footer band ----------------------------------------------------
    ax_foot.plot([0.0, 1.0], [1.0, 1.0], color=INK, lw=1.0,
                 transform=ax_foot.transAxes, clip_on=False)
    ax_foot.text(0.0, 0.45,
                 "github.com/mertdurukan/tooth-resorption-detection",
                 fontsize=9, color=INK, fontweight="bold",
                 family="DejaVu Sans Mono",
                 transform=ax_foot.transAxes, va="center")
    ax_foot.text(1.0, 0.45,
                 "src: results/metrics/performance_table.csv  ·  "
                 "bubble size ∝ model size (MB)",
                 fontsize=8.5, color=SUBINK, style="italic",
                 transform=ax_foot.transAxes, va="center", ha="right")

    out = PLOTS_DIR / "linkedin_benchmark.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Slide-style — blueprint/mono aesthetic to match SEC.03 deck (1200x800, 3:2)
# --------------------------------------------------------------------------- #
def figure_slide(perf: pd.DataFrame) -> Path:
    df = perf.copy()
    df = df[df["Composite"] > 0].reset_index(drop=True)

    base = df[df["Model"] == "CNN_Classification"].iloc[0]
    vit = df[df["Model"] == "vit_base_16"].iloc[0]
    speedup = base["Inference(ms)"] / vit["Inference(ms)"]
    delta_f1 = vit["F1-Score"] - base["F1-Score"]

    INK = "#0a0a0a"
    SUBINK = "#5b6273"
    BLUE = "#2c3edb"
    BLUE_SOFT = "#dbe0ff"
    RED = "#c5283d"
    RED_SOFT = "#f5d6dc"
    DOT = "#dfe5f1"
    BG = "#ffffff"
    MONO = "DejaVu Sans Mono"

    fig = plt.figure(figsize=(12, 8), dpi=120)
    fig.patch.set_facecolor(BG)

    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_facecolor(BG)
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.axis("off")
    nx, ny = 96, 60
    xs = np.linspace(0.02, 0.98, nx)
    ys = np.linspace(0.02, 0.98, ny)
    XS, YS = np.meshgrid(xs, ys)
    ax_bg.scatter(XS.flatten(), YS.flatten(), s=0.6, c=DOT, zorder=0)

    ax_bg.plot([0.04, 0.96], [0.93, 0.93], color=INK, lw=0.6, zorder=2)
    ax_bg.text(0.04, 0.955, "● SEC.03 // METRICS & REPRODUCIBILITY",
               fontsize=10, color=INK, family=MONO, fontweight="bold",
               va="center")
    ax_bg.text(0.96, 0.955, "PLOT // COMPARISON",
               fontsize=10, color=INK, family=MONO, fontweight="bold",
               va="center", ha="right")

    ax_bg.text(0.04, 0.885, "BASELINE CNN  →  ViT-BASE/16",
               fontsize=22, color=INK, fontweight="black", va="center")

    ax_bg.text(0.04, 0.835,
               f"// +{delta_f1:.2f} macro-F1  ·  {speedup:.1f}× faster inference  ·  same protocol, different inductive bias",
               fontsize=10.5, color=BLUE, family=MONO, va="center")

    ax_bg.plot([0.04, 0.96], [0.07, 0.07], color=INK, lw=0.6, zorder=2)
    ax_bg.text(0.04, 0.04,
               "src: results/metrics/performance_table.csv",
               fontsize=9, color=SUBINK, family=MONO, va="center")
    ax_bg.text(0.96, 0.04,
               "github.com/mertdurukan/tooth-resorption-detection",
               fontsize=9, color=INK, family=MONO, fontweight="bold",
               va="center", ha="right")

    ax = fig.add_axes([0.085, 0.13, 0.875, 0.66])
    ax.set_facecolor(BG)

    others = df[~df["Model"].isin(["CNN_Classification", "vit_base_16"])]

    ax.scatter(others["Inference(ms)"], others["F1-Score"],
               s=others["Size(MB)"] * 0.9 + 30,
               facecolors="none", edgecolors="#9aa3b2",
               linewidths=0.9, alpha=0.9, zorder=3)

    ax.scatter(base["Inference(ms)"], base["F1-Score"],
               s=base["Size(MB)"] * 1.6 + 320,
               facecolor=RED_SOFT, edgecolor=RED, linewidths=2.2, zorder=5)
    ax.scatter(base["Inference(ms)"], base["F1-Score"],
               s=55, c=RED, zorder=6)

    ax.scatter(vit["Inference(ms)"], vit["F1-Score"],
               s=vit["Size(MB)"] * 1.6 + 320,
               facecolor=BLUE_SOFT, edgecolor=BLUE, linewidths=2.2, zorder=5)
    ax.scatter(vit["Inference(ms)"], vit["F1-Score"],
               s=55, c=BLUE, zorder=6)

    ax.annotate(
        "", xy=(vit["Inference(ms)"] + 0.5, vit["F1-Score"] - 0.01),
        xytext=(base["Inference(ms)"] - 0.5, base["F1-Score"] + 0.01),
        arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.45",
                        color=BLUE, lw=2.2,
                        connectionstyle="arc3,rad=-0.3"),
        zorder=4)

    ax.text(vit["Inference(ms)"] + 9, vit["F1-Score"] + 0.025,
            "[ViT-BASE/16]",
            fontsize=11, color=BLUE, family=MONO, fontweight="bold",
            ha="left", va="center", zorder=7)
    ax.text(vit["Inference(ms)"] + 9, vit["F1-Score"] - 0.01,
            f"f1={vit['F1-Score']:.3f}  ms={vit['Inference(ms)']:.2f}",
            fontsize=9.5, color=SUBINK, family=MONO, ha="left",
            va="center", zorder=7)
    ax.text(vit["Inference(ms)"] + 9, vit["F1-Score"] - 0.045,
            "// best speed × performance",
            fontsize=9, color=BLUE, family=MONO, ha="left", va="center",
            zorder=7)

    ax.text(base["Inference(ms)"] - 4, base["F1-Score"] + 0.07,
            "[BASELINE CNN]",
            fontsize=11, color=RED, family=MONO, fontweight="bold",
            ha="right", va="center", zorder=7)
    ax.text(base["Inference(ms)"] - 4, base["F1-Score"] + 0.035,
            f"f1={base['F1-Score']:.3f}  ms={base['Inference(ms)']:.1f}",
            fontsize=9.5, color=SUBINK, family=MONO, ha="right",
            va="center", zorder=7)

    cx = (base["Inference(ms)"] + vit["Inference(ms)"]) / 2 - 4
    cy = (base["F1-Score"] + vit["F1-Score"]) / 2 + 0.10
    ax.text(cx, cy + 0.025,
            f"+{delta_f1:.2f}  MACRO-F1",
            fontsize=15, fontweight="black", color=INK,
            ha="center", va="center", zorder=8)
    ax.text(cx, cy - 0.025,
            f"{speedup:.1f}×  FASTER",
            fontsize=15, fontweight="black", color=BLUE,
            ha="center", va="center", zorder=8)

    ax.text(0.985, 0.05,
            f"// n={len(others)} additional architectures benchmarked",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=SUBINK, family=MONO, zorder=7)

    ax.set_xlabel("INFERENCE TIME (MS/IMG)  //  LOWER IS BETTER",
                  fontsize=10, color=INK, family=MONO, fontweight="bold",
                  labelpad=10)
    ax.set_ylabel("MACRO-F1  //  HIGHER IS BETTER",
                  fontsize=10, color=INK, family=MONO, fontweight="bold",
                  labelpad=10)

    ax.grid(True, linestyle=":", color="#cdd3df", alpha=0.9, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=INK, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family(MONO)

    ax.set_xlim(-3, 95)
    ax.set_ylim(0.10, 1.00)

    out = PLOTS_DIR / "slide_benchmark.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    alias = PLOTS_DIR / "comparison.png"
    fig.savefig(alias, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main() -> None:
    perf = pd.read_csv(METRICS_DIR / "performance_table.csv")
    classwise = pd.read_csv(METRICS_DIR / "classwise_performance.csv")
    with (METRICS_DIR / "high_confidence_summary.json").open(
            "r", encoding="utf-8") as f:
        hc = json.load(f)

    print("A:", figure_a(perf))
    print("C:", figure_c(perf))
    print("D:", figure_d(classwise, perf))
    print("E:", figure_e(hc))
    print("LinkedIn:", figure_linkedin(perf))
    print("Slide:", figure_slide(perf))


if __name__ == "__main__":
    main()
