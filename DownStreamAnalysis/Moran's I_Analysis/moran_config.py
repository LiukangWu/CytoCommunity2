# -*- coding: utf-8 -*-
import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local

# =========================
# Config
# =========================
INPUT_DIR = "Step4_Output/ResultTable_File"
OUT_DIR = "data/Moran"
GLOBAL_DIR = os.path.join(OUT_DIR, "global")
LOCAL_DIR = os.path.join(OUT_DIR, "local")
os.makedirs(GLOBAL_DIR, exist_ok=True)
os.makedirs(LOCAL_DIR, exist_ok=True)

X_COL = "x_coordinate"
Y_COL = "y_coordinate"
CN_COL = "CN_Label"
CELLTYPE_COL = "Cell_Type"

DIST_THRESHOLD = 50.0
PERMUTATIONS = 99
MIN_CELLS_PER_CN = 5
MIN_POSITIVE_COUNT = 2
ALPHA = 0.05


def infer_sample_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def infer_condition(sample_name):
    s = str(sample_name).lower()
    m = re.search(r'_(\d+)_', s)
    if m:
        return f"condition{m.group(1)}"
    return "Unknown"


def safe_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


def quadrant_to_label(q):
    # Moran_Local.q: 1=HH, 2=LH, 3=LL, 4=HL
    mapping = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    try:
        return mapping.get(int(q), "NA")
    except Exception:
        return "NA"


def build_distance_weights(coords, threshold=50.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = DistanceBand(
            coords,
            threshold=threshold,
            binary=True,
            silence_warnings=True
        )
    w.transform = "R"
    return w


csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"esda\.moran")

for csv_path in csv_files:
    sample_name = infer_sample_name(csv_path)
    condition = infer_condition(sample_name)
    print(f"[Processing] {sample_name}")

    df = pd.read_csv(csv_path)

    required_cols = [X_COL, Y_COL, CN_COL, CELLTYPE_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[Skip] {sample_name}: missing columns {missing}")
        continue

    df = df.copy()
    df[X_COL] = safe_numeric_series(df[X_COL])
    df[Y_COL] = safe_numeric_series(df[Y_COL])
    df[CN_COL] = df[CN_COL].astype(str).str.strip()
    df[CELLTYPE_COL] = df[CELLTYPE_COL].astype(str).str.strip()
    df = df.dropna(subset=[X_COL, Y_COL, CN_COL, CELLTYPE_COL]).copy()

    if df.empty:
        print(f"[Skip] {sample_name}: empty after cleanup")
        continue

    sample_global_rows = []
    sample_local_rows = []

    for cn, sub in df.groupby(CN_COL):
        sub = sub.copy().reset_index(drop=True)
        n_cells = len(sub)

        if n_cells < MIN_CELLS_PER_CN:
            continue

        coords = sub[[X_COL, Y_COL]].to_numpy(float)
        w = build_distance_weights(coords, threshold=DIST_THRESHOLD)

        if len(w.neighbors) == 0 or sum(len(v) for v in w.neighbors.values()) == 0:
            continue

        for ct in sorted(sub[CELLTYPE_COL].unique()):
            x = (sub[CELLTYPE_COL] == ct).astype(int).to_numpy()
            pos_count = int(x.sum())
            neg_count = int(len(x) - pos_count)

            if pos_count < MIN_POSITIVE_COUNT or neg_count < 1:
                continue

            # ---------- Global Moran ----------
            gm = Moran(x, w, permutations=PERMUTATIONS, two_tailed=False)

            sample_global_rows.append({
                "Sample": sample_name,
                "Condition": condition,
                "CN": cn,
                "CellType": ct,
                "N_cells_in_CN": n_cells,
                "Positive_count": pos_count,
                "Positive_fraction": pos_count / n_cells,
                "Global_Moran_I": float(gm.I),
                "Global_Moran_p_sim": float(gm.p_sim) if gm.p_sim is not None else np.nan
            })

            # ---------- Local Moran ----------
            lm = Moran_Local(x, w, permutations=PERMUTATIONS)

            for i in range(n_cells):
                quad = quadrant_to_label(lm.q[i])
                pval = float(lm.p_sim[i]) if hasattr(lm, "p_sim") else np.nan
                Ii = float(lm.Is[i])

                sample_local_rows.append({
                    "Sample": sample_name,
                    "Condition": condition,
                    "CN": cn,
                    "CellType": ct,
                    X_COL: sub.loc[i, X_COL],
                    Y_COL: sub.loc[i, Y_COL],
                    "Actual_CellType": sub.loc[i, CELLTYPE_COL],
                    "Local_Moran_I": Ii,
                    "Local_Moran_p_sim": pval,
                    "Quadrant": quad,
                    "Significant": int(pval < ALPHA) if not np.isnan(pval) else 0,
                    "Significant_HH": int((quad == "HH") and (pval < ALPHA))
                })

    sample_global_df = pd.DataFrame(sample_global_rows)
    sample_local_df = pd.DataFrame(sample_local_rows)

    global_path = os.path.join(GLOBAL_DIR, f"{sample_name}_Global_Moran.csv")
    local_path = os.path.join(LOCAL_DIR, f"{sample_name}_Local_Moran.csv")

    sample_global_df.to_csv(global_path, index=False)
    sample_local_df.to_csv(local_path, index=False)

    print(f"[Saved] Global -> {global_path}")
    print(f"[Saved] Local  -> {local_path}")

print("[Done] All samples processed.")