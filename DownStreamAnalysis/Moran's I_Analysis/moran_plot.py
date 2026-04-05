# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Matplotlib style
# =========================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 13,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# =========================
# Input / Output
# =========================
GLOBAL_DIR = "data/Moran/global"
LOCAL_DIR = "data/Moran/local"
OUT_DIR = "data/Moran/Figures_Triptych"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Column names
# =========================
X_COL = "x_coordinate"
Y_COL = "y_coordinate"

# =========================
# Candidate filter
# =========================
P_THRESH = 0.05
MIN_GLOBAL_I = 0.5
MIN_POSITIVE_COUNT = 5
TOP_N_PER_SAMPLE = 8
ONLY_SIGNIFICANT_GLOBAL = True

# =========================
# Figure / point style
# =========================
FIG_W = 20.0
FIG_H = 6.8

POINT_SIZE_BG = 14
POINT_SIZE_TARGET = 24
POINT_SIZE_HH = 44

ALPHA_BG = 0.18
ALPHA_TARGET = 0.90
ALPHA_NS = 0.28

COLOR_BG = "#D9D9D9"
COLOR_TARGET = "#C44E52"
COLOR_HH = "#D62728"
COLOR_LL = "#4C72B0"
COLOR_HL = "#F28E2B"
COLOR_LH = "#76B7B2"
COLOR_NS = "#D3D3D3"
COLOR_HH_OVERLAY = "#B40426"

SHOW_TICKS = False

# =========================
# Helpers
# =========================
def sanitize_filename(s):
    s = str(s)
    s = re.sub(r'[\\/:*?"<>| ]+', "_", s)
    return s

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def set_equal_xy(ax, x, y, pad_ratio=0.03):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    xr = xmax - xmin
    yr = ymax - ymin
    if xr == 0:
        xr = 1.0
    if yr == 0:
        yr = 1.0

    side = max(xr, yr)
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    pad = side * pad_ratio

    ax.set_xlim(xmid - side / 2 - pad, xmid + side / 2 + pad)
    ax.set_ylim(ymid - side / 2 - pad, ymid + side / 2 + pad)
    ax.set_aspect("equal", adjustable="box")

def infer_sample_name_from_global(global_csv_path):
    base = os.path.basename(global_csv_path)
    return re.sub(r"_Global_Moran\.csv$", "", base)

def get_local_csv_from_sample(sample_name):
    return os.path.join(LOCAL_DIR, f"{sample_name}_Local_Moran.csv")

def pick_candidates_from_global(global_df, top_n=8):
    df = global_df.copy()
    df = df[df["Positive_count"] >= MIN_POSITIVE_COUNT].copy()
    df = df[df["Global_Moran_I"] > MIN_GLOBAL_I].copy()

    if ONLY_SIGNIFICANT_GLOBAL:
        df = df[df["Global_Moran_p_sim"] < P_THRESH].copy()

    if df.empty:
        return df

    df = df.sort_values(
        by=["Global_Moran_I", "Global_Moran_p_sim", "Positive_count"],
        ascending=[False, True, False]
    ).copy()

    return df.head(top_n).copy()

def add_spatial_association_category(df):
    df = df.copy()

    def _cat(row):
        sig = int(row["Significant"]) == 1
        q = str(row["Quadrant"])
        if not sig:
            return "NS"
        if q == "HH":
            return "HH"
        if q == "LL":
            return "LL"
        if q == "HL":
            return "HL"
        if q == "LH":
            return "LH"
        return "NS"

    df["Spatial_Association"] = df.apply(_cat, axis=1)
    return df

def style_axis(ax, x, y, show_ticks=False):
    remove_top_right_spines(ax)
    set_equal_xy(ax, x, y)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

def add_panel_label(ax, label):
    ax.text(
        0.01, 0.99, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=14, fontweight="bold"
    )

def add_global_stat_text(fig, sample, cn, celltype, global_i, global_p):
    fig.suptitle(
        f"{sample} | CN {cn} | {celltype}",
        y=0.98, fontsize=16
    )
    fig.text(
        0.5, 0.935,
        f"Global Moran's I = {global_i:.3f}    p = {global_p:.3g}",
        ha="center", va="center", fontsize=12
    )

# =========================
# Single panel plotters
# =========================
def draw_celltype_spatial_map(ax, df_sub, celltype):
    ax.scatter(
        df_sub[X_COL], df_sub[Y_COL],
        s=POINT_SIZE_BG,
        c=COLOR_BG,
        alpha=ALPHA_BG,
        linewidths=0,
        rasterized=True
    )

    target_mask = df_sub["Actual_CellType"].astype(str) == str(celltype)
    if target_mask.any():
        ax.scatter(
            df_sub.loc[target_mask, X_COL],
            df_sub.loc[target_mask, Y_COL],
            s=POINT_SIZE_TARGET,
            c=COLOR_TARGET,
            alpha=ALPHA_TARGET,
            linewidths=0,
            rasterized=True
        )

    ax.set_title("Cell-type spatial map", fontsize=14, pad=8)
    style_axis(ax, df_sub[X_COL].values, df_sub[Y_COL].values, show_ticks=SHOW_TICKS)
    add_panel_label(ax, "a")

def draw_spatial_association_map(ax, df_sub):
    color_map = {
        "NS": COLOR_NS,
        "HH": COLOR_HH,
        "LL": COLOR_LL,
        "HL": COLOR_HL,
        "LH": COLOR_LH,
    }

    order = ["NS", "HH", "LL", "HL", "LH"]

    for cat in order:
        m = df_sub["Spatial_Association"] == cat
        if m.any():
            ax.scatter(
                df_sub.loc[m, X_COL],
                df_sub.loc[m, Y_COL],
                s=POINT_SIZE_TARGET if cat != "NS" else POINT_SIZE_BG,
                c=color_map[cat],
                alpha=ALPHA_NS if cat == "NS" else 0.92,
                linewidths=0,
                rasterized=True,
                label=cat if cat != "NS" else "NS"
            )

    ax.set_title("Spatial association map", fontsize=14, pad=8)
    style_axis(ax, df_sub[X_COL].values, df_sub[Y_COL].values, show_ticks=SHOW_TICKS)
    add_panel_label(ax, "b")

    handles, labels = ax.get_legend_handles_labels()
    desired = ["HH", "LL", "HL", "LH", "NS"]
    order_idx = [labels.index(x) for x in desired if x in labels]
    if order_idx:
        handles = [handles[i] for i in order_idx]
        labels = [labels[i] for i in order_idx]
        ax.legend(
            handles, labels,
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            markerscale=1.1
        )

def draw_significant_hotspot_map(ax, df_sub, celltype):
    ax.scatter(
        df_sub[X_COL], df_sub[Y_COL],
        s=POINT_SIZE_BG,
        c=COLOR_BG,
        alpha=0.16,
        linewidths=0,
        rasterized=True
    )

    target_mask = df_sub["Actual_CellType"].astype(str) == str(celltype)
    if target_mask.any():
        ax.scatter(
            df_sub.loc[target_mask, X_COL],
            df_sub.loc[target_mask, Y_COL],
            s=POINT_SIZE_TARGET,
            c=COLOR_TARGET,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
            label=str(celltype)
        )

    hh_mask = (df_sub["Spatial_Association"] == "HH")
    if hh_mask.any():
        ax.scatter(
            df_sub.loc[hh_mask, X_COL],
            df_sub.loc[hh_mask, Y_COL],
            s=POINT_SIZE_HH,
            c=COLOR_HH_OVERLAY,
            alpha=0.98,
            linewidths=0,
            rasterized=True,
            label="Significant hotspot"
        )

    ax.set_title("Significant hotspot map", fontsize=14, pad=8)
    style_axis(ax, df_sub[X_COL].values, df_sub[Y_COL].values, show_ticks=SHOW_TICKS)
    add_panel_label(ax, "c")

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        markerscale=1.1
    )

# =========================
# Triptych plotter
# =========================
def plot_triptych(df_sub, sample, cn, celltype, outpath, global_i, global_p):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))

    draw_celltype_spatial_map(axes[0], df_sub, celltype)
    draw_spatial_association_map(axes[1], df_sub)
    draw_significant_hotspot_map(axes[2], df_sub, celltype)

    add_global_stat_text(fig, sample, cn, celltype, global_i, global_p)

    plt.subplots_adjust(
        left=0.03,
        right=0.90,
        top=0.86,
        bottom=0.06,
        wspace=0.18
    )

    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# =========================
# Main
# =========================
global_csvs = sorted(glob.glob(os.path.join(GLOBAL_DIR, "*_Global_Moran.csv")))

if len(global_csvs) == 0:
    raise FileNotFoundError(f"No global Moran csv found in: {GLOBAL_DIR}")

total_candidates = 0
saved_files = 0

for global_csv in global_csvs:
    sample_name = infer_sample_name_from_global(global_csv)
    local_csv = get_local_csv_from_sample(sample_name)

    if not os.path.exists(local_csv):
        print(f"[Skip] local csv not found for sample: {sample_name}")
        continue

    global_df = pd.read_csv(global_csv)
    local_df = pd.read_csv(local_csv)

    required_global = [
        "Sample", "Condition", "CN", "CellType",
        "Positive_count", "Global_Moran_I", "Global_Moran_p_sim"
    ]
    required_local = [
        "Sample", "Condition", "CN", "CellType",
        "Actual_CellType", X_COL, Y_COL,
        "Local_Moran_I", "Local_Moran_p_sim",
        "Quadrant", "Significant"
    ]

    miss_g = [c for c in required_global if c not in global_df.columns]
    miss_l = [c for c in required_local if c not in local_df.columns]

    if miss_g:
        print(f"[Skip] {sample_name}: missing global columns {miss_g}")
        continue
    if miss_l:
        print(f"[Skip] {sample_name}: missing local columns {miss_l}")
        continue

    cand_df = pick_candidates_from_global(global_df, top_n=TOP_N_PER_SAMPLE)
    if cand_df.empty:
        print(f"[Info] {sample_name}: no candidate passed the filter")
        continue

    total_candidates += len(cand_df)

    sample_out_dir = os.path.join(OUT_DIR, sanitize_filename(sample_name))
    ensure_dir(sample_out_dir)

    for _, row in cand_df.iterrows():
        cn = row["CN"]
        celltype = row["CellType"]
        global_i = float(row["Global_Moran_I"])
        global_p = float(row["Global_Moran_p_sim"])

        sub = local_df[
            (local_df["CN"].astype(str) == str(cn)) &
            (local_df["CellType"].astype(str) == str(celltype))
        ].copy()

        if sub.empty:
            print(f"[Skip] {sample_name} | CN={cn} | {celltype}: empty local subset")
            continue

        sub = add_spatial_association_category(sub)

        outname = f"CN_{sanitize_filename(cn)}__{sanitize_filename(celltype)}.pdf"
        outpath = os.path.join(sample_out_dir, outname)

        plot_triptych(sub, sample_name, cn, celltype, outpath, global_i, global_p)

        saved_files += 1
        print(f"[Saved] {sample_name} | CN={cn} | {celltype} | I={global_i:.3f}")

print(f"[Done] candidates={total_candidates}, saved_triptychs={saved_files}")