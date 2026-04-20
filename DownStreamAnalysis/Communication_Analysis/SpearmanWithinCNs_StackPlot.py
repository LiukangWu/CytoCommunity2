# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# =========================
# Path
# =========================
LONG_FILE = os.path.join("data", "Communication", "config", "EnrichScoreMatrix_long.csv")
OUT_DIR = os.path.join("plot", "Communication", "SpearmanWithinCNs_plot","Stackplot")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =========================
# Parameters
# =========================
MIN_SAMPLES = 3
N_PERM = 1000
RANDOM_SEED = 12345

CN_ORDER = [str(i) for i in range(1, 11)]
PERM_P_SIG = 0.05
SPEARMAN_P_DRAW = 0.05

BAR_WIDTH = 0.22
CN_SPACING = 1.25
SEGMENT_EDGE_LW = 0.85
OUTER_EDGE_LW = 0.95
ZERO_LW = 1.0
STAR_MARKER_SIZE = 70
PAIR_GAP_FACTOR = 0.03
Y_MARGIN_RATIO = 0.06
CONDITION_LABEL_X = -0.040
CENTER_DIVIDER_LW = 1.0


mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Utilities
# =========================
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


def build_color_map(celltypes):
    celltypes = sorted([str(x) for x in celltypes])

    palette = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set1", "Dark2", "Accent", "Paired"]:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend(list(cmap.colors))
        else:
            palette.extend([cmap(i / 255.0) for i in range(256)])

    if len(celltypes) > len(palette):
        extra = plt.get_cmap("hsv")
        need = len(celltypes) - len(palette)
        palette.extend([extra(i / max(need, 1)) for i in range(need)])

    return {ct: palette[i] for i, ct in enumerate(celltypes)}


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

            abs_rho_a = abs(rho_a)
            abs_rho_b = abs(rho_b)
            delta_rho = rho_b - rho_a
            delta_abs_rho = abs_rho_b - abs_rho_a

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

            # 只要置换检验显著，就打星
            star_flag = bool(pd.notna(perm_p) and (perm_p < PERM_P_SIG))

            rows.append({
                "ConditionA": cond_a,
                "ConditionB": cond_b,
                "CN": str(cn),
                "CellType1": ct1,
                "CellType2": ct2,
                "Pair": f"{ct1}-{ct2}",
                "rho_A": rho_a,
                "abs_rho_A": abs_rho_a,
                "p_A": p_a,
                "rho_B": rho_b,
                "abs_rho_B": abs_rho_b,
                "p_B": p_b,
                "delta_rho": delta_rho,
                "delta_abs_rho": delta_abs_rho,
                "abs_delta_rho": abs(delta_rho),
                "perm_p": perm_p,
                "neglog10_perm_p": -np.log10(perm_p) if pd.notna(perm_p) and perm_p > 0 else np.nan,
                "valid_perm": valid_perm,
                "draw_A": bool(pd.notna(p_a) and (p_a < SPEARMAN_P_DRAW)),
                "draw_B": bool(pd.notna(p_b) and (p_b < SPEARMAN_P_DRAW)),
                "star_flag": star_flag,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["CN", "Pair"], ascending=[True, True]).reset_index(drop=True)


def make_stack_layout(diff_df, cn_order):
    pos_segments = []
    neg_segments = []
    pos_totals = {}
    neg_totals = {}

    for cn in cn_order:
        cn_df = diff_df[diff_df["CN"] == str(cn)].copy()

        pos_df = cn_df[cn_df["draw_A"]].copy()
        pos_df = pos_df.sort_values(["abs_rho_A", "Pair"], ascending=[False, True]).reset_index(drop=True)

        current = 0.0
        for _, row in pos_df.iterrows():
            h = float(row["abs_rho_A"])
            if not np.isfinite(h) or h <= 0:
                continue
            y0 = current
            y1 = current + h
            rec = row.to_dict()
            rec.update({
                "stack_side": "A",
                "y_bottom": y0,
                "y_top": y1,
                "height": h,
                "y_center": 0.5 * (y0 + y1),
            })
            pos_segments.append(rec)
            current = y1
        pos_totals[str(cn)] = current

        neg_df = cn_df[cn_df["draw_B"]].copy()
        neg_df = neg_df.sort_values(["abs_rho_B", "Pair"], ascending=[False, True]).reset_index(drop=True)

        current = 0.0
        for _, row in neg_df.iterrows():
            h = float(row["abs_rho_B"])
            if not np.isfinite(h) or h <= 0:
                continue
            y0 = current
            y1 = current - h
            rec = row.to_dict()
            rec.update({
                "stack_side": "B",
                "y_bottom": y0,
                "y_top": y1,
                "height": h,
                "y_center": 0.5 * (y0 + y1),
            })
            neg_segments.append(rec)
            current = y1
        neg_totals[str(cn)] = abs(current)

    pos_seg_df = pd.DataFrame(pos_segments)
    neg_seg_df = pd.DataFrame(neg_segments)
    return pos_seg_df, neg_seg_df, pos_totals, neg_totals


def get_segment_draw_range(y_bottom, y_top):
    height = y_top - y_bottom
    if abs(height) <= 1e-12:
        return None

    shrink = min(abs(height) * PAIR_GAP_FACTOR, 0.02)
    if height > 0:
        yb = y_bottom + shrink * 0.5
        yt = y_top - shrink * 0.5
    else:
        yb = y_bottom - shrink * 0.5
        yt = y_top + shrink * 0.5
    return yb, yt


def draw_segment_bar(ax, x, width, y_bottom, y_top, color):
    draw_range = get_segment_draw_range(y_bottom, y_top)
    if draw_range is None:
        return

    yb, yt = draw_range
    ax.add_patch(Rectangle(
        (x - width / 2.0, min(yb, yt)),
        width,
        abs(yt - yb),
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        zorder=3,
    ))


def _collect_boundary_y(seg_df):
    boundaries = []
    if seg_df.empty:
        return boundaries
    for _, row in seg_df.iterrows():
        draw_range = get_segment_draw_range(row["y_bottom"], row["y_top"])
        if draw_range is None:
            continue
        yb, yt = draw_range
        boundaries.extend([yb, yt])
    return boundaries


def _draw_cn_boundaries(ax, x, cn, pos_cn_df, neg_cn_df, pos_totals, neg_totals):
    y_top = pos_totals.get(str(cn), 0.0)
    y_bottom = -neg_totals.get(str(cn), 0.0)
    if abs(y_top - y_bottom) <= 1e-12:
        return

    left_x = x - BAR_WIDTH
    right_x = x + BAR_WIDTH

    ax.plot([x, x], [y_bottom, y_top], color="white", linewidth=CENTER_DIVIDER_LW,
            solid_capstyle="butt", zorder=4)

    y_values = []
    y_values.extend(_collect_boundary_y(pos_cn_df))
    y_values.extend(_collect_boundary_y(neg_cn_df))
    y_values.extend([0.0])

    unique_y = []
    for val in sorted(y_values):
        if not unique_y or abs(val - unique_y[-1]) > 1e-9:
            unique_y.append(val)

    for y in unique_y:
        ax.plot([left_x, right_x], [y, y], color="black", linewidth=SEGMENT_EDGE_LW,
                solid_capstyle="butt", zorder=5)

    ax.plot([left_x, left_x], [y_bottom, y_top], color="black", linewidth=OUTER_EDGE_LW,
            solid_capstyle="butt", zorder=6)
    ax.plot([right_x, right_x], [y_bottom, y_top], color="black", linewidth=OUTER_EDGE_LW,
            solid_capstyle="butt", zorder=6)


def plot_stacked_twinbars(diff_df, cond_a, cond_b, cn_order, color_map):
    pos_seg_df, neg_seg_df, pos_totals, neg_totals = make_stack_layout(diff_df, cn_order)

    if pos_seg_df.empty and neg_seg_df.empty:
        print(f"[WARN] condition{cond_a} vs condition{cond_b}: nothing to plot")
        return

    x_positions = {str(cn): i * CN_SPACING for i, cn in enumerate(cn_order)}
    max_pos = max([pos_totals.get(str(cn), 0.0) for cn in cn_order] + [0.0])
    max_neg = max([neg_totals.get(str(cn), 0.0) for cn in cn_order] + [0.0])
    y_abs = max(max_pos, max_neg, 0.5)
    y_margin = max(0.05, y_abs * Y_MARGIN_RATIO)

    fig_w = max(8.0, 0.95 * len(cn_order) + 2.0)
    fig_h = 6.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if not pos_seg_df.empty:
        for _, row in pos_seg_df.iterrows():
            x = x_positions[row["CN"]]
            draw_segment_bar(
                ax=ax,
                x=x - BAR_WIDTH / 2.0,
                width=BAR_WIDTH,
                y_bottom=row["y_bottom"],
                y_top=row["y_top"],
                color=color_map[row["CellType1"]],
            )
            draw_segment_bar(
                ax=ax,
                x=x + BAR_WIDTH / 2.0,
                width=BAR_WIDTH,
                y_bottom=row["y_bottom"],
                y_top=row["y_top"],
                color=color_map[row["CellType2"]],
            )

            if bool(row.get("star_flag", False)):
                ax.scatter(
                    [x],
                    [row["y_center"]],
                    marker="*",
                    s=STAR_MARKER_SIZE,
                    c="black",
                    linewidths=0.0,
                    zorder=7,
                )

    if not neg_seg_df.empty:
        for _, row in neg_seg_df.iterrows():
            x = x_positions[row["CN"]]
            draw_segment_bar(
                ax=ax,
                x=x - BAR_WIDTH / 2.0,
                width=BAR_WIDTH,
                y_bottom=row["y_bottom"],
                y_top=row["y_top"],
                color=color_map[row["CellType1"]],
            )
            draw_segment_bar(
                ax=ax,
                x=x + BAR_WIDTH / 2.0,
                width=BAR_WIDTH,
                y_bottom=row["y_bottom"],
                y_top=row["y_top"],
                color=color_map[row["CellType2"]],
            )

            if bool(row.get("star_flag", False)):
                ax.scatter(
                    [x],
                    [row["y_center"]],
                    marker="*",
                    s=STAR_MARKER_SIZE,
                    c="black",
                    linewidths=0.0,
                    zorder=7,
                )

    for cn in cn_order:
        x = x_positions[str(cn)]
        total_pos = pos_totals.get(str(cn), 0.0)

        pos_cn_df = pos_seg_df[pos_seg_df["CN"] == str(cn)].copy() if not pos_seg_df.empty else pd.DataFrame()
        neg_cn_df = neg_seg_df[neg_seg_df["CN"] == str(cn)].copy() if not neg_seg_df.empty else pd.DataFrame()

        _draw_cn_boundaries(ax, x, cn, pos_cn_df, neg_cn_df, pos_totals, neg_totals)

        label_y = total_pos + max(y_margin * 0.35, 0.04)
        ax.text(
            x,
            label_y,
            f"CN{cn}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            zorder=7,
        )

    ax.axhline(0.0, color="black", linewidth=ZERO_LW, zorder=8)

    xs = [x_positions[str(cn)] for cn in cn_order]
    ax.set_xticks([])
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(-(y_abs + y_margin), y_abs + y_margin)

    from matplotlib.ticker import MultipleLocator, FuncFormatter

    ax.set_ylim(-(y_abs + y_margin), y_abs + y_margin)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{abs(v):.0f}"))

    ax.tick_params(axis="y", which="major", length=5, width=0.8)
    ax.tick_params(axis="y", which="minor", length=3, width=0.6)

    ax.set_ylabel(r"Absolute Spearman correlation, $|\rho|$")
    ax.set_xlabel("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=3)

    ax.text(
        CONDITION_LABEL_X,
        0.84,
        f"condition{cond_a}",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
    )
    ax.text(
        CONDITION_LABEL_X,
        0.16,
        f"condition{cond_b}",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
    )

    out_pdf = os.path.join(FIG_DIR, f"StackedTwinBars_absRho_condition{cond_a}_vs_{cond_b}.pdf")
    out_png = os.path.join(FIG_DIR, f"StackedTwinBars_absRho_condition{cond_a}_vs_{cond_b}.png")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")
    print(f"[OK] Saved {out_png}")

    if not pos_seg_df.empty:
        pos_seg_df.to_csv(
            os.path.join(TABLE_DIR, f"PlotSegments_positive_absRho_condition{cond_a}_vs_{cond_b}.csv"),
            index=False,
        )
    if not neg_seg_df.empty:
        neg_seg_df.to_csv(
            os.path.join(TABLE_DIR, f"PlotSegments_negative_absRho_condition{cond_a}_vs_{cond_b}.csv"),
            index=False,
        )


def save_celltype_legend(color_map, out_pdf):
    items = sorted(color_map.items(), key=lambda x: x[0])
    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=c, markeredgecolor="none",
               markersize=8, label=ct)
        for ct, c in items
    ]

    n_items = len(handles)
    ncol = 1 if n_items <= 14 else 2
    nrow = int(np.ceil(n_items / ncol))

    fig_h = max(2.2, 0.28 * nrow + 0.5)
    fig_w = 3.8 if ncol == 1 else 7.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=ncol,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.2,
        labelspacing=0.7,
        title="Cell types",
    )
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")


def save_annotation_legend(out_pdf):
    fig, ax = plt.subplots(figsize=(3.8, 1.6))
    ax.axis("off")

    ax.scatter([0.12], [0.50], marker="*", s=110, c="black")
    ax.text(0.24, 0.50, "Permutation significant CT pair (perm p < 0.05)",
            va="center", ha="left")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_pdf}")


# =========================
# Main
# =========================
if not os.path.exists(LONG_FILE):
    raise FileNotFoundError(f"Cannot find {LONG_FILE}")

scores_long = pd.read_csv(LONG_FILE)
required_cols = {"Sample", "CN", "CellType", "Score"}
if not required_cols.issubset(scores_long.columns):
    raise ValueError(
        f"{LONG_FILE} must contain columns {required_cols}, current columns are {list(scores_long.columns)}"
    )

scores_long["Sample"] = scores_long["Sample"].astype(str)
scores_long["CN"] = scores_long["CN"].astype(str)
scores_long["CellType"] = scores_long["CellType"].astype(str)
scores_long["Score"] = pd.to_numeric(scores_long["Score"], errors="coerce")
scores_long = scores_long.dropna(subset=["Score"]).copy()
scores_long["Condition"] = scores_long["Sample"].apply(infer_condition)

conditions = sorted([c for c in scores_long["Condition"].unique() if c != "Unknown"])
if len(conditions) < 2:
    raise ValueError(f"Valid conditions are fewer than 2: {conditions}")

wide_dict = prepare_wide_tables(scores_long, CN_ORDER)
all_celltypes = sorted(scores_long["CellType"].dropna().astype(str).unique().tolist())
color_map = build_color_map(all_celltypes)

color_table = pd.DataFrame({
    "CellType": list(color_map.keys()),
    "Color": [mpl.colors.to_hex(v) for v in color_map.values()],
})
color_table.to_csv(os.path.join(TABLE_DIR, "CellType_ColorMap.csv"), index=False)
print(f"[OK] Saved {os.path.join(TABLE_DIR, 'CellType_ColorMap.csv')}")

save_celltype_legend(color_map, os.path.join(FIG_DIR, "Shared_CellType_Legend.pdf"))
save_annotation_legend(os.path.join(FIG_DIR, "Shared_Annotation_Legend.pdf"))

for cond_a, cond_b in itertools.combinations(conditions, 2):
    diff_df = compute_pair_result(wide_dict, cond_a, cond_b, CN_ORDER)
    if diff_df.empty:
        print(f"[WARN] condition{cond_a} vs condition{cond_b}: no usable result")
        continue

    out_csv = os.path.join(TABLE_DIR, f"CommunicationPairStats_absRho_condition{cond_a}_vs_{cond_b}.csv")
    diff_df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")

    plot_stacked_twinbars(diff_df, cond_a, cond_b, CN_ORDER, color_map)

print("\n[Done]")