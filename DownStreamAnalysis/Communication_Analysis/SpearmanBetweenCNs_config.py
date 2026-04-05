# -*- coding: utf-8 -*-
import os
import pandas as pd
from scipy.stats import spearmanr

CCA_DIR = "data/Communication/config"
OUT_DIR = CCA_DIR

# ===== 1. Read enrichment long table =====
scores_long = pd.read_csv(os.path.join(CCA_DIR, "EnrichScoreMatrix_long.csv"))
scores_long = scores_long.dropna(subset=["Sample", "CN", "CellType", "Score"])
scores_long["CN"] = scores_long["CN"].astype(str).str.strip()
scores_long["CellType"] = scores_long["CellType"].astype(str).str.strip()

# ===== 2. Read top CN pairs from CCA_config =====
pairs = pd.read_csv(os.path.join(CCA_DIR, "CCA_config.csv"))

required_pair_cols = {"CN_A", "CN_B", "rho1", "pval1"}
missing_pair_cols = required_pair_cols - set(pairs.columns)
if missing_pair_cols:
    raise ValueError(f"CCA_config.csv 缺少列: {missing_pair_cols}")

pairs["CN_A"] = pairs["CN_A"].astype(str).str.strip()
pairs["CN_B"] = pairs["CN_B"].astype(str).str.strip()

# 按 |rho1| 取前10
pairs = pairs.reindex(pairs["rho1"].abs().sort_values(ascending=False).index).head(10)

rows = []

# ===== 3. For each CN pair, compute Spearman =====
for _, r in pairs.iterrows():
    cnA = r["CN_A"]
    cnB = r["CN_B"]

    coord_file = os.path.join(CCA_DIR, f"CCA_coordinates_CN{cnA}_vs_CN{cnB}.csv")
    if not os.path.exists(coord_file):
        print("⚠️ 缺少文件:", coord_file)
        continue

    coord = pd.read_csv(coord_file)

    required_coord_cols = {
        "CellType",
        "CN_A_coord_can1", "CN_A_coord_can2",
        "CN_B_coord_can1", "CN_B_coord_can2"
    }
    missing_coord_cols = required_coord_cols - set(coord.columns)
    if missing_coord_cols:
        print("⚠️ 文件列缺失:", coord_file, "| missing:", sorted(missing_coord_cols))
        continue

    coord["CellType"] = coord["CellType"].astype(str).str.strip()

    # ===== 3.1 Select representative cell types =====
    # 方案A：按 |can1 coordinate| 最大选代表 cell type
    coord_A = coord.dropna(subset=["CN_A_coord_can1"]).copy()
    coord_B = coord.dropna(subset=["CN_B_coord_can1"]).copy()

    if coord_A.empty or coord_B.empty:
        print(f"⚠️ CN{cnA} vs CN{cnB} 没有有效 coordinate")
        continue

    topA = coord_A.loc[coord_A["CN_A_coord_can1"].abs().idxmax(), "CellType"]
    topB = coord_B.loc[coord_B["CN_B_coord_can1"].abs().idxmax(), "CellType"]

    # ===== 3.2 Extract scores from long table =====
    subA = scores_long.loc[
        (scores_long["CN"] == cnA) & (scores_long["CellType"] == topA),
        ["Sample", "Score"]
    ].rename(columns={"Score": "Score_A"})

    subB = scores_long.loc[
        (scores_long["CN"] == cnB) & (scores_long["CellType"] == topB),
        ["Sample", "Score"]
    ].rename(columns={"Score": "Score_B"})

    merged = pd.merge(subA, subB, on="Sample", how="inner")

    print(f"🔹 CN{cnA}({topA}) vs CN{cnB}({topB}) overlap={len(merged)}")

    if len(merged) < 3:
        print("⚠️ 样本重叠不足")
        continue

    if merged["Score_A"].nunique() < 2 or merged["Score_B"].nunique() < 2:
        print("⚠️ 其中一侧几乎常数")
        continue

    rho, p = spearmanr(merged["Score_A"], merged["Score_B"], nan_policy="omit")

    rows.append({
        "CN_A": cnA,
        "CN_B": cnB,
        "rho1_CCA": r["rho1"],
        "pval1_CCA": r["pval1"],
        "rho2_CCA": r["rho2"] if "rho2" in pairs.columns else None,
        "pval2_CCA": r["pval2"] if "pval2" in pairs.columns else None,
        "Top_CellType_A": topA,
        "Top_CellType_B": topB,
        "TopA_coord_can1": coord_A.loc[coord_A["CN_A_coord_can1"].abs().idxmax(), "CN_A_coord_can1"],
        "TopB_coord_can1": coord_B.loc[coord_B["CN_B_coord_can1"].abs().idxmax(), "CN_B_coord_can1"],
        "Spearman_rho": rho,
        "Spearman_p": p,
        "N_overlap": len(merged)
    })

# ===== 4. Output =====
res_df = pd.DataFrame(rows)

if not res_df.empty:
    res_df = res_df.sort_values("rho1_CCA", key=lambda s: s.abs(), ascending=False)

out_csv = os.path.join(OUT_DIR, "Spearman_TopCCA_Pairs.csv")
res_df.to_csv(out_csv, index=False)

print("\n✅ 结果已保存:", out_csv)