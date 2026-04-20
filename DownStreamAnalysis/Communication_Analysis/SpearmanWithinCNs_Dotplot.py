# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colorbar import ColorbarBase



# 路径
LONG_FILE = os.path.join("data", "Communication", "config", "EnrichScoreMatrix_long.csv")
OUT_DIR = os.path.join("plot", "Communication", "SpearmanWithinCNs_plot","Dotplot")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)



# 参数
MIN_SAMPLES = 3
N_PERM = 1000
RANDOM_SEED = 12345

CN_ORDER = [str(i) for i in range(1, 11)]
SIZE_LEVELS = [0.1, 0.5, 1.0, 1.3, 2.0]
SIZE_CAP = 2.0

PERM_P_STAR = 0.05
SPEARMAN_P_STAR = 0.05

DOT_SIZE_MIN = 20
DOT_SIZE_MAX = 220
STAR_MARKER_SIZE = 35

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



# 工具函数
def infer_condition(sample_name):
    parts = str(sample_name).strip().split("_")
    return str(parts[1]) if len(parts) >= 3 else "Unknown"


def stable_seed(*args):
    return RANDOM_SEED + sum(ord(c) for c in "_".join(map(str, args)))


def safe_spearman(x, y, min_samples=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (~np.isnan(x)) & (~np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < min_samples:
        return np.nan, np.nan
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    rho, p = spearmanr(x, y)
    if np.isnan(rho):
        return np.nan, np.nan
    return float(rho), (float(p) if not np.isnan(p) else np.nan)


def permutation_pvalue(x, y, labels, cond_a, cond_b, delta_obs,
                       n_perm=1000, min_samples=3, seed=12345):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = np.asarray(labels)

    valid = (~np.isnan(x)) & (~np.isnan(y)) & np.isin(labels, [cond_a, cond_b])
    x = x[valid]
    y = y[valid]
    labels = labels[valid]

    idx_a = np.where(labels == cond_a)[0]
    idx_b = np.where(labels == cond_b)[0]
    if len(idx_a) < min_samples or len(idx_b) < min_samples:
        return np.nan, 0

    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(labels))
    n_a = len(idx_a)

    extreme = 0
    valid_perm = 0
    max_try = n_perm * 20

    for _ in range(max_try):
        perm = rng.permutation(idx_all)
        p_a = perm[:n_a]
        p_b = perm[n_a:]

        rho_a_perm, _ = safe_spearman(x[p_a], y[p_a], min_samples=min_samples)
        rho_b_perm, _ = safe_spearman(x[p_b], y[p_b], min_samples=min_samples)
        if np.isnan(rho_a_perm) or np.isnan(rho_b_perm):
            continue

        valid_perm += 1
        delta_perm = rho_b_perm - rho_a_perm
        if abs(delta_perm) >= abs(delta_obs) - 1e-12:
            extreme += 1
        if valid_perm >= n_perm:
            break

    if valid_perm == 0:
        return np.nan, 0

    perm_p = (extreme + 1.0) / (valid_perm + 1.0)
    return perm_p, valid_perm


def size_map(values, vmax=2.0, smin=20, smax=220):
    values = np.clip(np.asarray(values, dtype=float), 0, vmax)
    if vmax <= 1e-12:
        return np.full_like(values, (smin + smax) / 2.0)
    return smin + (values / vmax) * (smax - smin)


def save_shared_colorbar(vmax, out_pdf):
    fig = plt.figure(figsize=(1.2, 3.6))
    ax = fig.add_axes([0.38, 0.08, 0.26, 0.84])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cb = ColorbarBase(ax, cmap=plt.get_cmap("coolwarm"), norm=norm, orientation="vertical")
    cb.set_label("delta rho")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")


def save_shared_size_legend(out_pdf, levels, vmax=2.0, smin=20, smax=220):
    fig, ax = plt.subplots(figsize=(1.9, 3.2))
    ax.axis("off")

    handles, labels = [], []
    for v in levels:
        handles.append(ax.scatter([], [], s=size_map([v], vmax=vmax, smin=smin, smax=smax)[0],
                                  color="gray", edgecolors="none"))
        labels.append(f"{v:.1f}")

    ax.legend(
        handles, labels,
        title="-log10(perm_p)",
        loc="center",
        frameon=False,
        labelspacing=1.1,
        handletextpad=1.0,
    )
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")


def prepare_wide_tables(scores_long, cn_order):
    wide_dict = {}
    for cn in cn_order:
        sub = scores_long[scores_long["CN"] == cn]
        if sub.empty:
            continue
        wide_dict[cn] = (
            sub.pivot_table(
                index=["Sample", "Condition"],
                columns="CellType",
                values="Score",
                aggfunc="mean",
            )
            .reset_index()
        )
    return wide_dict


def compute_pair_result(wide_dict, cond_a, cond_b, cn_order):
    rows = []

    for cn in cn_order:
        wide = wide_dict.get(cn)
        if wide is None:
            continue

        wide = wide[wide["Condition"].isin([cond_a, cond_b])].copy()
        if wide.empty:
            continue

        celltypes = [c for c in wide.columns if c not in ["Sample", "Condition"]]
        if len(celltypes) < 2:
            continue

        for ct1, ct2 in itertools.combinations(celltypes, 2):
            tmp = wide[["Sample", "Condition", ct1, ct2]].dropna().copy()
            if tmp.empty:
                continue

            grp_a = tmp[tmp["Condition"] == cond_a]
            grp_b = tmp[tmp["Condition"] == cond_b]
            if len(grp_a) < MIN_SAMPLES or len(grp_b) < MIN_SAMPLES:
                continue

            rho_a, p_a = safe_spearman(grp_a[ct1].values, grp_a[ct2].values, MIN_SAMPLES)
            rho_b, p_b = safe_spearman(grp_b[ct1].values, grp_b[ct2].values, MIN_SAMPLES)
            if np.isnan(rho_a) or np.isnan(rho_b):
                continue

            delta_rho = rho_b - rho_a
            perm_p, valid_perm = permutation_pvalue(
                x=tmp[ct1].values,
                y=tmp[ct2].values,
                labels=tmp["Condition"].values,
                cond_a=cond_a,
                cond_b=cond_b,
                delta_obs=delta_rho,
                n_perm=N_PERM,
                min_samples=MIN_SAMPLES,
                seed=stable_seed(cond_a, cond_b, cn, ct1, ct2),
            )

            rows.append({
                "ConditionA": cond_a,
                "ConditionB": cond_b,
                "CN": str(cn),
                "CellType1": ct1,
                "CellType2": ct2,
                "Pair": f"{ct1}-{ct2}",
                "rho_A": rho_a,
                "p_A": p_a,
                "rho_B": rho_b,
                "p_B": p_b,
                "delta_rho": delta_rho,
                "abs_delta_rho": abs(delta_rho),
                "n_A": len(grp_a),
                "n_B": len(grp_b),
                "perm_p": perm_p,
                "neglog10_perm_p": -np.log10(perm_p) if pd.notna(perm_p) and perm_p > 0 else np.nan,
                "valid_perm": valid_perm,
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["abs_delta_rho", "CN", "Pair"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def plot_pair(diff_df, cond_a, cond_b, pair_order, global_color_vmax):
    x_map = {p: i for i, p in enumerate(pair_order)}
    y_map = {cn: i for i, cn in enumerate(CN_ORDER)}
    norm = TwoSlopeNorm(vmin=-global_color_vmax, vcenter=0.0, vmax=global_color_vmax)

    plot_df = diff_df[diff_df["Pair"].isin(pair_order)].copy()
    if plot_df.empty:
        return

    plot_df["x"] = plot_df["Pair"].map(x_map)
    plot_df["y"] = plot_df["CN"].map(y_map)
    plot_df["neglog10_plot"] = plot_df["neglog10_perm_p"].clip(lower=0, upper=SIZE_CAP)
    plot_df["dot_size"] = size_map(
        plot_df["neglog10_plot"].values,
        vmax=SIZE_CAP,
        smin=DOT_SIZE_MIN,
        smax=DOT_SIZE_MAX,
    )

    star_df = plot_df[
        (plot_df["perm_p"] < PERM_P_STAR) &
        ((plot_df["p_A"] < SPEARMAN_P_STAR) | (plot_df["p_B"] < SPEARMAN_P_STAR))
    ].copy()

    fig_w = max(10, 0.30 * len(pair_order) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    ax.scatter(
        plot_df["x"], plot_df["y"],
        s=plot_df["dot_size"],
        c=plot_df["delta_rho"],
        cmap="coolwarm",
        norm=norm,
        edgecolors="none",
        zorder=2,
    )

    ax.scatter(
        star_df["x"], star_df["y"],
        marker="*",
        s=STAR_MARKER_SIZE,
        c="black",
        edgecolors="none",
        zorder=3,
    )

    ax.set_xlim(-0.5, len(pair_order) - 0.5)
    ax.set_ylim(-0.5, len(CN_ORDER) - 0.5)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, rotation=90, ha="center", va="top")
    ax.set_yticks(range(len(CN_ORDER)))
    ax.set_yticklabels([f"CN{i}" for i in range(1, 11)])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_pdf = os.path.join(FIG_DIR, f"DotPlot_condition{cond_a}-condition{cond_b}_perm_shared.pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")



# 1) 读取长表
if not os.path.exists(LONG_FILE):
    raise FileNotFoundError(f"找不到 {LONG_FILE}")

scores_long = pd.read_csv(LONG_FILE)
required_cols = {"Sample", "CN", "CellType", "Score"}
if not required_cols.issubset(scores_long.columns):
    raise ValueError(
        f"{LONG_FILE} 至少应包含列: {required_cols}，当前列为: {list(scores_long.columns)}"
    )

scores_long["Sample"] = scores_long["Sample"].astype(str)
scores_long["CN"] = scores_long["CN"].astype(str)
scores_long["CellType"] = scores_long["CellType"].astype(str)
scores_long["Score"] = pd.to_numeric(scores_long["Score"], errors="coerce")
scores_long = scores_long.dropna(subset=["Score"]).copy()
scores_long["Condition"] = scores_long["Sample"].apply(infer_condition)

conditions = sorted([c for c in scores_long["Condition"].unique() if c != "Unknown"])
if len(conditions) < 2:
    raise ValueError(f"有效 condition 数不足 2，当前识别到: {conditions}")

wide_dict = prepare_wide_tables(scores_long, CN_ORDER)



# 2) 逐个 condition pair 重新计算并保存
pair_results = []
all_diff_list = []

for cond_a, cond_b in itertools.combinations(conditions, 2):
    diff_df = compute_pair_result(wide_dict, cond_a, cond_b, CN_ORDER)
    if diff_df.empty:
        print(f"[WARN] condition{cond_a} vs condition{cond_b} 无可用结果")
        continue

    out_fp = os.path.join(TABLE_DIR, f"CommunicationDiff_condition{cond_a}_vs_{cond_b}_perm.csv")
    diff_df.to_csv(out_fp, index=False)
    print(f"[OK] Saved {out_fp}")

    pair_results.append((cond_a, cond_b, diff_df))
    all_diff_list.append(diff_df)

if not all_diff_list:
    raise ValueError("没有生成任何可用于作图的差异结果。")

all_diff_df = pd.concat(all_diff_list, axis=0, ignore_index=True)
pair_order = (
    all_diff_df.groupby("Pair")["abs_delta_rho"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()
)

global_color_vmax = float(np.nanmax(np.abs(all_diff_df["delta_rho"].values)))
global_color_vmax = max(global_color_vmax, 0.3)



# 3) 共享图例 / 色卡：每次都重新生
save_shared_colorbar(
    global_color_vmax,
    os.path.join(FIG_DIR, "Shared_Colorbar_delta_rho.pdf"),
)

save_shared_size_legend(
    os.path.join(FIG_DIR, "Shared_SizeLegend_neglog10_perm_p.pdf"),
    levels=SIZE_LEVELS,
    vmax=SIZE_CAP,
    smin=DOT_SIZE_MIN,
    smax=DOT_SIZE_MAX,
)



# 4) 画三张主图
for cond_a, cond_b, diff_df in pair_results:
    plot_pair(diff_df, cond_a, cond_b, pair_order, global_color_vmax)

print("\n[Done]")
