# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind

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
INPUT_DIR = "data/Moran/global"
OUT_DIR = "data/Moran/DiffAggregation"
os.makedirs(OUT_DIR, exist_ok=True)

PAIR_LIST = [
    ("condition0", "condition1"),
    ("condition0", "condition2"),
    ("condition1", "condition2"),
]

CAP_MLOG10 = 5.0
SIZE_MIN = 40.0
SIZE_MAX = 1200.0
PVAL_MIN = 1e-300
CMAP_NAME = "coolwarm"


# =========================
# Helpers
# =========================
def cn_sort_key(x):
    s = str(x)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return (0, int(digits), s)
    return (1, s)


def safe_one_sided_ttest(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    delta = mean2 - mean1

    try:
        res = ttest_ind(x2, x1, equal_var=False, nan_policy="omit")
        t_stat = float(res.statistic)
        p_two = float(res.pvalue)
    except Exception:
        return np.nan, delta, mean1, mean2

    if np.isnan(t_stat) or np.isnan(p_two):
        return np.nan, delta, mean1, mean2

    if delta >= 0:
        p_one = p_two / 2.0 if t_stat >= 0 else 1.0 - p_two / 2.0
    else:
        p_one = p_two / 2.0 if t_stat <= 0 else 1.0 - p_two / 2.0

    return p_one, delta, mean1, mean2


def build_pair_mats(df, cond_a, cond_b, ct_order, cn_order):
    n_ct = len(ct_order)
    n_cn = len(cn_order)

    ct_to_i = {ct: i for i, ct in enumerate(ct_order)}
    cn_to_j = {cn: j for j, cn in enumerate(cn_order)}

    delta_mat = np.full((n_ct, n_cn), np.nan, dtype=float)
    pval_mat = np.ones((n_ct, n_cn), dtype=float)

    sub = df[df["Condition"].isin([cond_a, cond_b])].copy()

    for (cn, ct), g in sub.groupby(["CN", "CellType"]):
        if ct not in ct_to_i or cn not in cn_to_j:
            continue

        vals_a = g.loc[g["Condition"] == cond_a, "Global_Moran_I"].dropna().to_numpy()
        vals_b = g.loc[g["Condition"] == cond_b, "Global_Moran_I"].dropna().to_numpy()

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        pval, delta, _, _ = safe_one_sided_ttest(vals_a, vals_b)
        if np.isnan(delta):
            continue

        i = ct_to_i[ct]
        j = cn_to_j[cn]

        delta_mat[i, j] = delta
        if np.isnan(pval):
            pval_mat[i, j] = 1.0
        else:
            pval_mat[i, j] = np.clip(pval, PVAL_MIN, 1.0)

    return delta_mat, pval_mat


def compute_global_vmax(df, pair_list, ct_order, cn_order, percentile=98, hard_cap=None):
    all_abs = []

    for cond_a, cond_b in pair_list:
        delta_mat, _ = build_pair_mats(df, cond_a, cond_b, ct_order, cn_order)
        finite = delta_mat[np.isfinite(delta_mat)]
        if finite.size > 0:
            all_abs.append(np.abs(finite))

    if len(all_abs) == 0:
        vmax = 1.0
    else:
        merged = np.concatenate(all_abs, axis=0)
        vmax = np.nanpercentile(merged, percentile)
        vmax = max(vmax, 1e-6)

    if hard_cap is not None:
        vmax = min(vmax, hard_cap)

    return vmax


def build_size_mapper(cap_mlog10=CAP_MLOG10, size_min=SIZE_MIN, size_max=SIZE_MAX):
    cap_mlog10 = max(float(cap_mlog10), 1e-12)
    size_min = float(size_min)
    size_max = float(size_max)

    def map_size(m):
        m = np.asarray(m, dtype=float)
        m = np.clip(m, 0.0, cap_mlog10)
        out = size_min + (m / cap_mlog10) * (size_max - size_min)
        out = np.clip(out, size_min, size_max)
        if out.ndim == 0:
            return float(out)
        return out

    return map_size


# =========================
# Plotting (same presentation as Pvalue_dotplot)
# =========================
def plot_dot_only_pdf(delta_mat, pval_mat, celltypes, cn_order, out_pdf, norm):
    n_ct, n_cn = delta_mat.shape

    pval_mat = np.clip(pval_mat, PVAL_MIN, 1.0)
    mlog10 = -np.log10(pval_mat)
    mlog10 = np.minimum(mlog10, CAP_MLOG10)
    mlog10 = np.nan_to_num(mlog10, nan=0.0, posinf=CAP_MLOG10, neginf=0.0)

    map_size = build_size_mapper()

    xs, ys, cs, ss = [], [], [], []
    for i in range(n_ct):
        for j in range(n_cn):
            xs.append(i)
            ys.append(j)
            cs.append(delta_mat[i, j])
            ss.append(map_size(mlog10[i, j]))

    xs = np.array(xs)
    ys = np.array(ys)
    cs = np.array(cs, dtype=float)
    ss = np.array(ss, dtype=float)
    ss = np.nan_to_num(ss, nan=SIZE_MIN, posinf=SIZE_MAX, neginf=SIZE_MIN)
    ss = np.clip(ss, SIZE_MIN, SIZE_MAX)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(False)

    ax.scatter(
        xs,
        ys,
        s=ss,
        c=cs,
        cmap=CMAP_NAME,
        norm=norm,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95
    )

    sig_mask = (mlog10.flatten() >= 1.3) & np.isfinite(cs)
    if np.any(sig_mask):
        ax.scatter(
            xs[sig_mask],
            ys[sig_mask],
            marker="*",
            s=85,
            c="black",
            alpha=0.65,
            linewidths=0.0,
            zorder=5
        )

    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(celltypes, rotation=90, ha="center", va="top")
    ax.set_yticks(range(n_cn))
    ax.set_yticklabels([f"CN{k}" for k in range(1, n_cn + 1)])

    ax.tick_params(axis="both", which="both", direction="out", length=4, width=0.8)
    ax.set_xlim(-0.5, n_ct - 0.5)
    ax.set_ylim(-0.5, n_cn - 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_pdf, bbox_inches="tight", dpi=1200)
    plt.close(fig)

    return map_size


def save_colorbar_pdf(norm, out_pdf, cmap_name=CMAP_NAME, label="Δ mean Global Moran's I"):
    fig = plt.figure(figsize=(1.4, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    cax = fig.add_axes([0.35, 0.12, 0.25, 0.80])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title(label, pad=6)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(out_pdf, bbox_inches="tight", dpi=1200)
    plt.close(fig)


def save_size_legend_pdf(map_size, out_pdf, cap_mlog10=CAP_MLOG10, cmap_name=CMAP_NAME):
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    lax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    lax.axis("off")

    ref_vals = np.array([0.1, 0.5, 1.3, 2], dtype=float)
    ref_vals = ref_vals[ref_vals <= cap_mlog10]

    warm_color = plt.get_cmap(cmap_name)(0.9)

    handles = []
    for r in ref_vals:
        ms = np.sqrt(float(map_size(r)))
        h = Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=warm_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=ms
        )
        handles.append(h)

    labels = [f"{r:g}" for r in ref_vals]

    lax.legend(
        handles,
        labels,
        title="-log10(pvalue)",
        frameon=False,
        loc="upper left",
        borderaxespad=0.0,
        labelspacing=1.3,
        handletextpad=1.0,
        numpoints=1
    )

    fig.savefig(out_pdf, bbox_inches="tight", dpi=1200)
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Global_Moran.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No global Moran csv files found in {INPUT_DIR}")

    all_df = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_df.append(df)

    df = pd.concat(all_df, axis=0, ignore_index=True)

    required_cols = ["Condition", "CN", "CellType", "Global_Moran_I"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["CN"] = df["CN"].astype(str).str.strip()
    df["CellType"] = df["CellType"].astype(str).str.strip()
    df["Global_Moran_I"] = pd.to_numeric(df["Global_Moran_I"], errors="coerce")
    df = df.dropna(subset=["Global_Moran_I"]).copy()

    ct_order = sorted(df["CellType"].dropna().astype(str).unique().tolist())
    cn_order = sorted(df["CN"].dropna().astype(str).unique().tolist(), key=cn_sort_key)

    vmax_global = compute_global_vmax(df, PAIR_LIST, ct_order, cn_order, percentile=98, hard_cap=None)
    norm = TwoSlopeNorm(vmin=-vmax_global, vcenter=0.0, vmax=vmax_global)
    print(f"[OK] Global colorbar vmax = {vmax_global:.4g}")

    legend_map_size = build_size_mapper()
    any_done = False

    for cond_a, cond_b in PAIR_LIST:
        delta_mat, pval_mat = build_pair_mats(df, cond_a, cond_b, ct_order, cn_order)

        out_pdf = os.path.join(
            OUT_DIR,
            f"DotPlot_{cond_a}-{cond_b}_GlobalMoran.pdf"
        )
        plot_dot_only_pdf(delta_mat, pval_mat, ct_order, cn_order, out_pdf, norm)

        any_done = True
        print(f"[OK] Saved {out_pdf}")

    if any_done:
        out_cbar = os.path.join(OUT_DIR, "DotPlot_COLORBAR_GlobalMoran.pdf")
        out_sleg = os.path.join(OUT_DIR, "DotPlot_SIZELEGEND_GlobalMoran.pdf")
        save_colorbar_pdf(norm, out_cbar, label="Δ mean Global Moran's I")
        save_size_legend_pdf(legend_map_size, out_sleg)
        print(f"[OK] Saved legends: {out_cbar} + {out_sleg}")
    else:
        print("[WARN] No dotplot generated; legends not created.")


if __name__ == "__main__":
    main()
