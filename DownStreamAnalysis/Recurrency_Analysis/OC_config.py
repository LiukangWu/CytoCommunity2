# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind

# =========================
# Config
# =========================
INPUT_FOLDER = "data/EnrichScoreMatrix"
IMAGE_LIST_FILE = "TNBC_Input/ImageNameList.txt"
CELLTYPE_FILE = os.path.join(INPUT_FOLDER, "CellTypeVec_List.csv")

OUT_DIR = "data/OverlapCoefficient"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_OUT = os.path.join(OUT_DIR, "config.csv")
REAL_OUT   = os.path.join(OUT_DIR, "RealDistribution_sameCN.csv")
BG_OUT     = os.path.join(OUT_DIR, "BackgroundDistribution_randomCN.csv")
TTEST_OUT  = os.path.join(OUT_DIR, "TTest_Results.csv")
DOM_OUT    = os.path.join(OUT_DIR, "Sample_CN_DominantCellTypes.csv")

SCORE_THRESH = -math.log10(0.05)
N_RANDOM = 100000
SEED = 1234

# True: 背景中两个随机 CN 不能相同
# False: 背景中允许相同 CN
EXCLUDE_SAME_CN_IN_BG = False


# =========================
# Helpers
# =========================
def overlap_coefficient(set1, set2):
    min_size = min(len(set1), len(set2))
    if min_size == 0:
        return 0.0
    return len(set1.intersection(set2)) / float(min_size)


def one_sided_welch_ttest_greater(x, y):
    """
    H0: mean(x) <= mean(y)
    H1: mean(x) > mean(y)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    res = ttest_ind(x, y, equal_var=False, alternative="greater")
    return float(res.statistic), float(res.pvalue)


def cn_sort_key(cn_name):
    s = str(cn_name).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 10**9


# =========================
# 1. Load sample names
# =========================
region_df = pd.read_csv(
    IMAGE_LIST_FILE,
    sep="\t",
    header=None,
    names=["Image"]
)
samples = region_df["Image"].astype(str).tolist()

if len(samples) < 2:
    raise RuntimeError("样本数少于 2，无法计算跨样本 OC。")

# =========================
# 2. Load cell type names
# =========================
celltypes_df = pd.read_csv(CELLTYPE_FILE, header=0)
celltypes = celltypes_df.iloc[:, 0].astype(str).tolist()
n_types = len(celltypes)

# =========================
# 3. Load per-sample enrich score matrix
# =========================
sample_to_mat = {}
n_cns = None

for sample in samples:
    fp = os.path.join(INPUT_FOLDER, f"{sample}_EnrichScoreMatrix.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"缺少文件: {fp}")

    mat = pd.read_csv(fp, header=None).values

    if mat.shape[0] != n_types:
        raise ValueError(
            f"{os.path.basename(fp)} 的行数 ({mat.shape[0]}) != cell type 数 ({n_types})"
        )

    if n_cns is None:
        n_cns = mat.shape[1]
    elif mat.shape[1] != n_cns:
        raise ValueError(
            f"{os.path.basename(fp)} 的 CN 列数 ({mat.shape[1]}) 与前面不一致 ({n_cns})"
        )

    sample_to_mat[sample] = mat

print(f"[INFO] samples={len(samples)}, celltypes={n_types}, CNs={n_cns}")

# =========================
# 4. Build dominant cell-type sets
# =========================
dominant_sets = {}
dominant_rows = []

for sample in samples:
    dominant_sets[sample] = {}
    mat = sample_to_mat[sample]

    for cn_idx in range(n_cns):
        enrich_vec = mat[:, cn_idx]
        selected_idx = np.where(enrich_vec > SCORE_THRESH)[0]
        selected_cts = [celltypes[i] for i in selected_idx]
        selected_set = set(selected_cts)

        dominant_sets[sample][cn_idx] = selected_set

        dominant_rows.append({
            "Sample": sample,
            "CN": f"TCN{cn_idx + 1}",
            "NumDominantCellTypes": len(selected_cts),
            "DominantCellTypes": ", ".join(selected_cts)
        })

dominant_df = pd.DataFrame(dominant_rows)
dominant_df.to_csv(DOM_OUT, index=False)

# =========================
# 5. Real distribution
#    same CN across sample pairs
# =========================
real_rows = []

for cn_idx in range(n_cns):
    cn_name = f"TCN{cn_idx + 1}"
    for sample_a, sample_b in combinations(samples, 2):
        set_a = dominant_sets[sample_a][cn_idx]
        set_b = dominant_sets[sample_b][cn_idx]
        oc = overlap_coefficient(set_a, set_b)

        real_rows.append({
            "TCN": cn_name,          # 保持兼容旧代码
            "CN": cn_name,           # 新代码统一用 CN
            "Region_A": sample_a,
            "Region_B": sample_b,
            "Sample_A": sample_a,
            "Sample_B": sample_b,
            "OverlapCoefficient": oc
        })

real_df = pd.DataFrame(real_rows)

# 兼容你原来的 plot 读取方式
config_df = real_df[["TCN", "Region_A", "Region_B", "OverlapCoefficient"]].copy()
config_df.to_csv(CONFIG_OUT, index=False)

# 完整 real 分布
real_df[["CN", "Sample_A", "Sample_B", "OverlapCoefficient"]].to_csv(REAL_OUT, index=False)

# =========================
# 6. Background distribution
# =========================
rng = np.random.default_rng(SEED)
sample_pairs = list(combinations(samples, 2))

bg_rows = []
for i in range(N_RANDOM):
    sample_a, sample_b = sample_pairs[rng.integers(0, len(sample_pairs))]
    cn_a = int(rng.integers(0, n_cns))

    if EXCLUDE_SAME_CN_IN_BG:
        candidates = [x for x in range(n_cns) if x != cn_a]
        cn_b = int(rng.choice(candidates))
    else:
        cn_b = int(rng.integers(0, n_cns))

    set_a = dominant_sets[sample_a][cn_a]
    set_b = dominant_sets[sample_b][cn_b]
    oc = overlap_coefficient(set_a, set_b)

    bg_rows.append({
        "Iter": i + 1,
        "Sample_A": sample_a,
        "CN_A": f"TCN{cn_a + 1}",
        "Sample_B": sample_b,
        "CN_B": f"TCN{cn_b + 1}",
        "OverlapCoefficient": oc
    })

bg_df = pd.DataFrame(bg_rows)
bg_df.to_csv(BG_OUT, index=False)

# =========================
# 7. T-test: per CN real vs background
# =========================
bg_vals = bg_df["OverlapCoefficient"].values
ttest_rows = []

cn_names = sorted(real_df["CN"].unique(), key=cn_sort_key)

for cn_name in cn_names:
    real_vals = real_df.loc[real_df["CN"] == cn_name, "OverlapCoefficient"].values
    t_stat, p_val = one_sided_welch_ttest_greater(real_vals, bg_vals)

    ttest_rows.append({
        "CN": cn_name,
        "n_real": len(real_vals),
        "n_background": len(bg_vals),
        "real_mean": np.mean(real_vals) if len(real_vals) > 0 else np.nan,
        "real_std": np.std(real_vals, ddof=1) if len(real_vals) > 1 else np.nan,
        "background_mean": np.mean(bg_vals) if len(bg_vals) > 0 else np.nan,
        "background_std": np.std(bg_vals, ddof=1) if len(bg_vals) > 1 else np.nan,
        "delta_mean": (
            np.mean(real_vals) - np.mean(bg_vals)
            if len(real_vals) > 0 and len(bg_vals) > 0 else np.nan
        ),
        "t_statistic": t_stat,
        "p_value_one_sided": p_val
    })

# overall
overall_real = real_df["OverlapCoefficient"].values
t_stat_all, p_val_all = one_sided_welch_ttest_greater(overall_real, bg_vals)

ttest_rows.append({
    "CN": "Overall",
    "n_real": len(overall_real),
    "n_background": len(bg_vals),
    "real_mean": np.mean(overall_real) if len(overall_real) > 0 else np.nan,
    "real_std": np.std(overall_real, ddof=1) if len(overall_real) > 1 else np.nan,
    "background_mean": np.mean(bg_vals) if len(bg_vals) > 0 else np.nan,
    "background_std": np.std(bg_vals, ddof=1) if len(bg_vals) > 1 else np.nan,
    "delta_mean": (
        np.mean(overall_real) - np.mean(bg_vals)
        if len(overall_real) > 0 and len(bg_vals) > 0 else np.nan
    ),
    "t_statistic": t_stat_all,
    "p_value_one_sided": p_val_all
})

ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(TTEST_OUT, index=False)

print("[DONE]")
print(CONFIG_OUT)
print(REAL_OUT)
print(BG_OUT)
print(TTEST_OUT)
print(DOM_OUT)