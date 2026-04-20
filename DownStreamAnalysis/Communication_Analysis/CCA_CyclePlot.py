# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# Config
# =========================
INPUT_CSV = "data/Communication/config/CCA_config_by_condition.csv"
OUT_DIR = "plot/Communication/SpearmanBetweenCNs_plot/CirclePlot"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CN = 10                  # 若为 None，则自动从数据推断
P_THRESHOLD = 0.05           # 只画显著边
MIN_RHO1 = 0.0               # 可改成 0.2 / 0.3 让图更干净
MIN_COMMON_SAMPLES = 3

# 主图参数
FIGSIZE = (8.4, 8.4)
NODE_RADIUS = 1.00
LABEL_RADIUS = 1.20
NODE_SIZE = 620
START_ANGLE_DEG = 90

# 边宽范围
LW_MIN = 1.0
LW_MAX = 7.5

# 节点颜色
NODE_CMAP = plt.get_cmap("tab10")

# condition 顺序与颜色
CONDITION_ORDER = ["condition0", "condition1", "condition2", "t0", "t1", "t2"]
CONDITION_COLORS = {
    "condition0": "#4C78A8",  # blue
    "condition1": "#54A24B",  # green
    "condition2": "#E45756",  # red
    "t0": "#4C78A8",
    "t1": "#54A24B",
    "t2": "#E45756",
}

# overlay 透明度
EDGE_ALPHA_OVERLAY = 0.45

# 同一对 CN 的不同 condition 边略微错开
OVERLAY_RAD_OFFSET = {
    "condition0": -0.14,
    "condition1":  0.00,
    "condition2":  0.14,
    "t0": -0.14,
    "t1":  0.00,
    "t2":  0.14,
}

# 图例文件名
MAIN_PDF = "CCA_circle_overlay_main.pdf"
RHO_LEGEND_PDF = "CCA_circle_overlay_legend_rho1.pdf"
COND_LEGEND_PDF = "CCA_circle_overlay_legend_condition.pdf"
COMBINED_LEGEND_PDF = "CCA_circle_overlay_legend_combined.pdf"


# =========================
# Helpers
# =========================
def parse_cn(x):
    s = str(x).strip().upper().replace("CN", "")
    return int(float(s))


def normalize_condition(cond):
    return str(cond).strip()


def get_node_color(cn):
    return NODE_CMAP((cn - 1) % 10)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find: {path}")

    df = pd.read_csv(path)

    required = ["Condition", "CN_A", "CN_B", "rho1", "pval1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Condition"] = df["Condition"].apply(normalize_condition)
    df["CN_A"] = df["CN_A"].apply(parse_cn)
    df["CN_B"] = df["CN_B"].apply(parse_cn)
    df["rho1"] = pd.to_numeric(df["rho1"], errors="coerce")
    df["pval1"] = pd.to_numeric(df["pval1"], errors="coerce")

    if "n_samples" in df.columns:
        df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")
    else:
        df["n_samples"] = np.nan

    df = df.dropna(subset=["Condition", "CN_A", "CN_B", "rho1", "pval1"]).copy()
    df = df[df["CN_A"] != df["CN_B"]].copy()

    # 只保留显著边
    df["rho1"] = df["rho1"].clip(lower=0)
    df = df[df["pval1"] < P_THRESHOLD].copy()
    df = df[df["rho1"] >= MIN_RHO1].copy()

    if "n_samples" in df.columns:
        df = df[(df["n_samples"].isna()) | (df["n_samples"] >= MIN_COMMON_SAMPLES)].copy()

    return df


def build_condition_order(df):
    present = list(pd.unique(df["Condition"]))
    ordered = [c for c in CONDITION_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def infer_cn_list(df, num_cn=None):
    if num_cn is not None:
        return list(range(1, num_cn + 1))
    cn_set = sorted(set(df["CN_A"]).union(set(df["CN_B"])))
    return cn_set


def build_circle_positions(cn_list, radius=1.0, start_angle_deg=90):
    n = len(cn_list)
    start_angle_rad = math.radians(start_angle_deg)
    pos = {}

    for i, cn in enumerate(cn_list):
        theta = start_angle_rad - 2.0 * math.pi * i / n
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        pos[cn] = (x, y, theta)

    return pos


def compute_base_rad(i, j, n_cn):
    d = abs(i - j)
    d = min(d, n_cn - d)

    base = 0.14 + 0.08 * max(d - 1, 0)
    sign = 1 if ((i + j) % 2 == 0) else -1
    return sign * min(base, 0.42)


def linewidth_from_rho(rho, rho_min, rho_max, lw_min=1.0, lw_max=7.5):
    if np.isclose(rho_min, rho_max):
        return 0.5 * (lw_min + lw_max)
    return lw_min + (rho - rho_min) / (rho_max - rho_min) * (lw_max - lw_min)


def add_edge(ax, x1, y1, x2, y2, rad, color, lw, alpha, zorder=1):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round"
    )
    ax.add_patch(patch)


def add_nodes(ax, cn_list, pos):
    for cn in cn_list:
        x, y, theta = pos[cn]

        ax.scatter(
            x, y,
            s=NODE_SIZE,
            color=get_node_color(cn),
            edgecolors="black",
            linewidths=0.7,
            zorder=4
        )

        lx = LABEL_RADIUS * math.cos(theta)
        ly = LABEL_RADIUS * math.sin(theta)

        ax.text(
            lx, ly, f"CN{cn}",
            ha="center", va="center",
            fontsize=14,
            zorder=5
        )


def finalize_axes(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 1.38
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)


def save_pdf(fig, path):
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


# =========================
# Legend handle builders
# =========================
def get_rho_legend_handles(rho_min, rho_max):
    if np.isclose(rho_min, rho_max):
        ticks = [round(float(rho_max), 2)]
    else:
        ticks = np.linspace(rho_min, rho_max, 4)
        ticks = [round(float(v), 2) for v in ticks]

    handles = []
    labels = []
    for rv in ticks:
        lw = linewidth_from_rho(rv, rho_min, rho_max, LW_MIN, LW_MAX)
        handles.append(
            Line2D(
                [0], [0],
                color="black",
                lw=lw,
                alpha=0.85
            )
        )
        labels.append(f"{rv:.2f}")
    return handles, labels


def get_condition_legend_handles(condition_order):
    handles = []
    labels = []
    for cond in condition_order:
        handles.append(
            Line2D(
                [0], [0],
                color=CONDITION_COLORS.get(cond, "#666666"),
                lw=3.2
            )
        )
        labels.append(cond)
    return handles, labels


# =========================
# Main plot
# =========================
def draw_overlay_main(df_all, condition_order, cn_list, pos, cn_to_idx, rho_min, rho_max):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    df = df_all.copy().sort_values("rho1", ascending=True)

    for _, row in df.iterrows():
        cond = row["Condition"]
        cn_a = int(row["CN_A"])
        cn_b = int(row["CN_B"])
        rho1 = float(row["rho1"])

        x1, y1, _ = pos[cn_a]
        x2, y2, _ = pos[cn_b]

        i = cn_to_idx[cn_a]
        j = cn_to_idx[cn_b]
        base_rad = compute_base_rad(i, j, len(cn_list))
        extra = OVERLAY_RAD_OFFSET.get(cond, 0.0)
        rad = np.clip(base_rad + extra, -0.55, 0.55)

        lw = linewidth_from_rho(rho1, rho_min, rho_max, LW_MIN, LW_MAX)
        color = CONDITION_COLORS.get(cond, "#666666")

        add_edge(
            ax, x1, y1, x2, y2,
            rad=rad,
            color=color,
            lw=lw,
            alpha=EDGE_ALPHA_OVERLAY,
            zorder=1
        )

    add_nodes(ax, cn_list, pos)
    finalize_axes(ax)

    out_path = os.path.join(OUT_DIR, MAIN_PDF)
    save_pdf(fig, out_path)


# =========================
# Separate legends
# =========================
def draw_rho_legend_pdf(rho_min, rho_max):
    handles, labels = get_rho_legend_handles(rho_min, rho_max)

    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.axis("off")

    leg = ax.legend(
        handles=handles,
        labels=labels,
        title="rho1",
        frameon=False,
        loc="center",
        handlelength=2.6,
        labelspacing=1.0,
        borderpad=0.2
    )
    leg._legend_box.align = "left"

    out_path = os.path.join(OUT_DIR, RHO_LEGEND_PDF)
    save_pdf(fig, out_path)


def draw_condition_legend_pdf(condition_order):
    handles, labels = get_condition_legend_handles(condition_order)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.axis("off")

    leg = ax.legend(
        handles=handles,
        labels=labels,
        title="Condition",
        frameon=False,
        loc="center",
        handlelength=2.6,
        labelspacing=1.0,
        borderpad=0.2
    )
    leg._legend_box.align = "left"

    out_path = os.path.join(OUT_DIR, COND_LEGEND_PDF)
    save_pdf(fig, out_path)


def draw_combined_legend_pdf(rho_min, rho_max, condition_order):
    rho_handles, rho_labels = get_rho_legend_handles(rho_min, rho_max)
    cond_handles, cond_labels = get_condition_legend_handles(condition_order)

    fig = plt.figure(figsize=(3.4, 5.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    leg1 = ax.legend(
        handles=rho_handles,
        labels=rho_labels,
        title="rho1",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        handlelength=2.6,
        labelspacing=1.0,
        borderpad=0.2
    )
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=cond_handles,
        labels=cond_labels,
        title="Condition",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.48),
        handlelength=2.6,
        labelspacing=1.0,
        borderpad=0.2
    )
    leg2._legend_box.align = "left"
    ax.add_artist(leg2)

    out_path = os.path.join(OUT_DIR, COMBINED_LEGEND_PDF)
    save_pdf(fig, out_path)


# =========================
# Main
# =========================
def main():
    df = load_data(INPUT_CSV)
    if df.empty:
        raise ValueError("No significant edges after filtering.")

    condition_order = build_condition_order(df)
    cn_list = infer_cn_list(df, NUM_CN)
    pos = build_circle_positions(
        cn_list,
        radius=NODE_RADIUS,
        start_angle_deg=START_ANGLE_DEG
    )
    cn_to_idx = {cn: i for i, cn in enumerate(cn_list)}

    rho_min = float(df["rho1"].min())
    rho_max = float(df["rho1"].max())

    draw_overlay_main(df, condition_order, cn_list, pos, cn_to_idx, rho_min, rho_max)
    draw_rho_legend_pdf(rho_min, rho_max)
    draw_condition_legend_pdf(condition_order)
    draw_combined_legend_pdf(rho_min, rho_max, condition_order)


if __name__ == "__main__":
    main()