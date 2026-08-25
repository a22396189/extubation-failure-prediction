#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sensitivity_leave_first_bin_out.py

Sensitivity analysis addressing committee comment on label circularity in the
early-phenotype-classifier pipeline (Section 3.5.6 / 5.4 / 6.3).

── Background / concern ──────────────────────────────────────────────────────
Phenotypes are derived (Section 3.5.5) by K-means clustering on Transformer
encoder embeddings computed from ALL twelve 4-hour bins (t = -52h .. -8h).
The early classifier (Section 3.5.6) is then trained to predict this cluster
label using ONLY the earliest bin (t = -52h to -48h). Because that earliest
bin is one of the twelve inputs used to build the embedding that defined the
labels in the first place, part of the classifier's apparent early
predictive power could mechanically reflect this overlap rather than
independent early physiological signal.

── What this script does ─────────────────────────────────────────────────────
1. Re-extracts patient embeddings from the SAME frozen, already-trained
   Transformer checkpoint (no retraining), but with the earliest bin
   (t = -52h) masked out exactly the way the model already treats any
   missing bin: step_present=0 for that position, so it is excluded from
   the encoder's key_padding_mask and contributes nothing to attention or
   pooling. This yields a "clean" embedding built only from the remaining
   11 bins (t = -48h .. -8h), using the identical model weights and
   pipeline as the original analysis (only the input differs).
2. Re-runs K-means (k=4) on this clean embedding and compares the resulting
   cluster assignments against the original (all-12-bin) assignments via
   Adjusted Rand Index (ARI) and a contingency table.
3. Trains a fresh XGBoost multiclass classifier on the excluded first-bin
   raw clinical features to predict the CLEAN cluster labels (i.e. labels
   that by construction contain no mechanical contribution from that bin),
   and recomputes one-versus-rest AUROC with bootstrap 95% CI.
4. Saves an appendix-ready table and figure summarizing (2) and (3), to be
   referenced as a robustness check against the original Figure 14 result.

Note on appendix numbering: the current manuscript already uses Appendix
Table A1-A5 and Appendix Figure A1-A7 (see 4.3.7 / Section 5). This
sensitivity analysis is therefore saved/labeled as Appendix Table A6 and
Appendix Figure A8 (the next free slots), not A5/A6 as in the raw reviewer
note.

Outputs (in OUTPUT_DIR):
  - clean_embedding_cluster_assignments.csv   (stay_id, cluster_clean)
  - crosstab_old_vs_clean_clusters.csv        (contingency table)
  - table_A6_ari_and_auroc_comparison.csv     (Appendix Table A6)
  - figure_A8_crosstab_and_auroc.png          (Appendix Figure A8)
  - early_bin_predict_clean_labels_merged.csv
  - model_xgb_multiclass_clean.json
"""

import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


# Fix: Windows threadpoolctl / MKL DLL compatibility (OSError 0xc06d007f)
# Must be set before importing sklearn / xgboost.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader

# ── Fix import path ──────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "model training"))

from transformer_pre_extubation_risk_trajectory import (
    ExtubationTransformer, ExtubationSeqDataset,
    STATIC_COLS, DYNAMIC_COLS, TARGET, SEQ_TIME_BINS, split_by_stay_id
)
from extubation_failure_phenotyping import extract_combined_features

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy.optimize import linear_sum_assignment
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =========================
# Paths / constants
# =========================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
BASE          = rf"{EXTUBATION_ROOT}"
DATA_CSV      = rf"{BASE}\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
MODEL_PATH    = rf"{BASE}\results\transformer\best_transformer.pt"
CLUSTER_CSV   = rf"{BASE}\results\phenotyping\cluster_assignments.csv"
ORIG_AUC_CSV  = rf"{BASE}\results\phenotyping\early_cluster_pred_timebin_minus52\metrics_auc_ovr.csv"
OUTPUT_DIR    = rf"{BASE}\results\phenotyping\sensitivity_leave_first_bin_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model architecture (must match training exactly)
D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 64, 4, 3, 128
DROPOUT, PE_FACTOR = 0.2, 1.0

N_CLUSTERS   = 4
SEED         = 42
TEST_SIZE    = 0.30
TIME_BIN_TARGET = -52   # earliest bin, t = -52h to -48h

# Same phenotype-mortality-based labeling convention as the original analysis
PHENOTYPE_LABELS = {
    1: "Phenotype 1 (Critical)",
    0: "Phenotype 2 (High Risk)",
    2: "Phenotype 3 (Moderate Risk)",
    3: "Phenotype 4 (Low Risk)",
}
PHENOTYPE_COLORS = {1: "#c0392b", 0: "#e67e22", 2: "#2980b9", 3: "#27ae60"}

STATIC_COLS_EARLY = ["age", "sex", "BMI", "Charlson_Score"]
DYNAMIC_COLS_EARLY = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day",
    "pH", "PaO2", "PaCO2", "BE", "OI",
    "Cr", "WBC", "Hb", "PLT", "AnionGap", "Lactate", "Glucose",
    "io_balance", "Vasopressor_use", "Hemodialysis_use",
]
FEATURE_COLS_EARLY = STATIC_COLS_EARLY + DYNAMIC_COLS_EARLY


# =========================
# Helpers
# =========================
def mask_first_bin(dataset: ExtubationSeqDataset) -> ExtubationSeqDataset:
    """
    Return a deep-copied dataset in which the earliest time bin
    (index 0, t = -52h) is treated as NOT present, i.e. exactly the way
    the model already handles any missing bin for other patients:
      - x_dyn[0, :] -> 0   (feature values + missingness mask both zeroed)
      - step_present[0] -> 0.0

    Because step_present feeds directly into the Transformer's
    src_key_padding_mask, this removes bin -52h from self-attention (and
    therefore from pooling) for every patient, without retraining the model
    and without changing anything about how the remaining 11 bins are
    represented.
    """
    ds_masked = copy.deepcopy(dataset)
    for s in ds_masked.samples:
        s["x_dyn"][0, :] = 0.0
        s["step_present"][0] = 0.0
    return ds_masked


def bootstrap_auc(y_true_bin, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true_bin)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_true_bin[idx], y_prob[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) < 10:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    w = np.array([1.0 / freq[yi] for yi in y], dtype=float)
    return w * (len(y) / w.sum())


def align_clean_to_old(crosstab: pd.DataFrame):
    """
    Hungarian matching between clean-cluster IDs (rows) and old
    phenotype IDs (columns) that maximizes overlap, purely to make the
    clean cluster IDs human-readable in the AUROC table
    (e.g. "clean cluster 2 corresponds to old Phenotype 1: Critical").
    Does NOT affect ARI (which is permutation-invariant) or AUROC.
    """
    cost = -crosstab.values.astype(float)
    row_idx, col_idx = linear_sum_assignment(cost)
    mapping = {crosstab.index[r]: crosstab.columns[c] for r, c in zip(row_idx, col_idx)}
    return mapping


# =========================
# Main
# =========================
def main():
    # Force CPU: running Transformer inference on CUDA and then calling
    # sklearn KMeans in the same process crashes on this Windows/CUDA/MKL
    # combination (native abort, no Python traceback). Inference over ~1000
    # patients through this small Transformer is fast enough on CPU.
    device = torch.device("cpu")
    print(f"[Device] {device}")

    # ── Load data & split (identical seed/ratio to the original phenotyping run) ──
    df = pd.read_csv(DATA_CSV)
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)

    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.70, seed=SEED)
    print(f"[Split] train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
                  if c in df.columns and c not in {"sex", "Vasopressor_use", "Hemodialysis_use"}]
    scaler = StandardScaler().fit(df[df["stay_id"].isin(train_ids)][scale_cols])

    ds_test_full = ExtubationSeqDataset(
        df[df["stay_id"].isin(test_ids)], test_ids, scaler, scale_cols, SEQ_TIME_BINS)

    # ── Load frozen Transformer (no retraining) ──
    model = ExtubationTransformer(
        dyn_dim=52, stat_dim=len(STATIC_COLS),
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        dropout=DROPOUT, pe_factor=PE_FACTOR,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    print(f"✓ Model loaded: {MODEL_PATH}")

    # ── Step 1a: original (all 12 bins) embedding — used only to get the fail mask ──
    loader_full = DataLoader(ds_test_full, batch_size=256, shuffle=False)
    _, orig_labels, sids_full, _ = extract_combined_features(
        model, loader_full, device, pooling="last", embed_source="encoder", use_static=False)

    # ── Step 1b: "clean" embedding — earliest bin masked as not-present ──
    ds_test_masked = mask_first_bin(ds_test_full)
    loader_masked = DataLoader(ds_test_masked, batch_size=256, shuffle=False)
    combined_clean, labels_masked, sids_masked, _ = extract_combined_features(
        model, loader_masked, device, pooling="last", embed_source="encoder", use_static=False)

    assert np.array_equal(sids_full, sids_masked), \
        "Patient order mismatch between full and masked extraction — check DataLoader shuffle settings."
    assert np.array_equal(orig_labels, labels_masked), \
        "Extubation_failure labels differ between full and masked extraction — should be identical."

    fail_mask = orig_labels == 1
    combined_clean_fail = combined_clean[fail_mask]
    sids_fail = sids_full[fail_mask]
    print(f"[Filter] Extubation Failure patients: {combined_clean_fail.shape[0]}")

    # ── Step 2: K-means (k=4) on the clean embedding ──
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=50)
    clean_labels = km.fit_predict(combined_clean_fail)
    clean_sil = silhouette_score(combined_clean_fail, clean_labels)
    print(f"[KMeans, clean embedding] k={N_CLUSTERS}  Silhouette={clean_sil:.4f}")
    for cid in range(N_CLUSTERS):
        print(f"  Cluster {cid}: {(clean_labels == cid).sum()} patients")

    df_clean = pd.DataFrame({"stay_id": sids_fail.tolist(), "cluster_clean": clean_labels.astype(int)})
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "clean_embedding_cluster_assignments.csv"),
                     index=False, encoding="utf-8-sig")

    # ── Compare against the ORIGINAL (all-12-bin) cluster assignments ──
    df_old = pd.read_csv(CLUSTER_CSV)[["stay_id", "cluster"]].rename(columns={"cluster": "cluster_old"})
    df_cmp = df_clean.merge(df_old, on="stay_id", how="inner")
    n_matched = len(df_cmp)
    n_expected = len(df_clean)
    if n_matched != n_expected:
        print(f"[WARN] Only {n_matched}/{n_expected} patients matched between clean and "
              f"original cluster assignments — check that CLUSTER_CSV corresponds to the "
              f"same run/split.")

    ari = adjusted_rand_score(df_cmp["cluster_old"].values, df_cmp["cluster_clean"].values)
    print(f"\n[Robustness] ARI(original 12-bin clusters, clean 11-bin clusters) = {ari:.4f}  (n={n_matched})")

    crosstab = pd.crosstab(df_cmp["cluster_clean"], df_cmp["cluster_old"])
    crosstab.index.name = "cluster_clean"
    crosstab.columns.name = "cluster_old"
    crosstab.to_csv(os.path.join(OUTPUT_DIR, "crosstab_old_vs_clean_clusters.csv"), encoding="utf-8-sig")
    print("\n[Contingency table] clean cluster (rows) vs original cluster (cols):")
    print(crosstab.to_string())

    # Human-readable mapping: which clean cluster mostly corresponds to which original phenotype
    clean_to_old_map = align_clean_to_old(crosstab)
    clean_to_phenotype_name = {
        cc: PHENOTYPE_LABELS.get(int(oc), f"Cluster {oc}") for cc, oc in clean_to_old_map.items()
    }
    print("\n[Best-match mapping] clean cluster -> nearest original phenotype:")
    for cc, name in clean_to_phenotype_name.items():
        print(f"  clean cluster {cc}  ~  {name}")

    # ── Step 3: predict the CLEAN labels from the excluded first bin's raw features ──
    print("\n[Early prediction] Training XGBoost on t = -52h features -> clean labels...")
    feats = pd.read_csv(DATA_CSV)
    df_bin = feats[feats["time_bin"] == TIME_BIN_TARGET].copy()
    df_bin = df_bin.sort_values("stay_id").drop_duplicates("stay_id", keep="last")
    if "sex" in df_bin.columns and df_bin["sex"].dtype == object:
        df_bin["sex"] = (df_bin["sex"].astype(str).str.lower() == "male").astype(int)

    df_bin = df_bin.merge(df_clean, on="stay_id", how="inner")
    avail_cols = [c for c in FEATURE_COLS_EARLY if c in df_bin.columns]
    for c in avail_cols:
        df_bin[c] = pd.to_numeric(df_bin[c], errors="coerce")

    merged_path = os.path.join(OUTPUT_DIR, "early_bin_predict_clean_labels_merged.csv")
    df_bin.to_csv(merged_path, index=False, encoding="utf-8-sig")

    X = df_bin[avail_cols].fillna(df_bin[avail_cols].median(numeric_only=True))
    y = df_bin["cluster_clean"].astype(int).values
    classes = np.sort(np.unique(y))
    n_classes = len(classes)
    print(f"  n={len(X)}  classes={classes.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    w_train = compute_class_weights(y_train)

    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=n_classes,
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=1,
        random_state=SEED, n_jobs=1, eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train, sample_weight=w_train)
    clf.save_model(os.path.join(OUTPUT_DIR, "model_xgb_multiclass_clean.json"))

    proba = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    clean_auc_rows = []
    for i, c in enumerate(classes):
        col = y_test_bin[:, i]
        n_pos, n_test = int(col.sum()), len(col)
        if n_pos == 0 or n_pos == n_test:
            auc, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            auc = roc_auc_score(col, proba[:, i])
            ci_lo, ci_hi = bootstrap_auc(col, proba[:, i], n_boot=1000, seed=SEED)
        phenotype_name = clean_to_phenotype_name.get(int(c), f"Clean cluster {int(c)}")
        clean_auc_rows.append({
            "clean_cluster": int(c), "nearest_original_phenotype": phenotype_name,
            "n_test": n_test, "n_pos": n_pos,
            "auc_clean": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            "ci_lo_clean": round(float(ci_lo), 4) if not np.isnan(ci_lo) else np.nan,
            "ci_hi_clean": round(float(ci_hi), 4) if not np.isnan(ci_hi) else np.nan,
        })
    df_auc_clean = pd.DataFrame(clean_auc_rows)
    print("\n[AUROC, clean labels]")
    print(df_auc_clean.to_string(index=False))

    # ── Merge with the ORIGINAL Figure 14 AUROC values for side-by-side comparison ──
    df_auc_orig = pd.read_csv(ORIG_AUC_CSV)  # columns: class, phenotype, n_test, n_pos, auc, ci_lo, ci_hi
    df_auc_orig = df_auc_orig.rename(columns={
        "class": "orig_cluster_id", "auc": "auc_original",
        "ci_lo": "ci_lo_original", "ci_hi": "ci_hi_original",
    })
    df_auc_clean["orig_cluster_id"] = df_auc_clean["clean_cluster"].map(clean_to_old_map)
    table_A6 = df_auc_clean.merge(
        df_auc_orig[["orig_cluster_id", "phenotype", "auc_original", "ci_lo_original", "ci_hi_original"]],
        on="orig_cluster_id", how="left")
    table_A6.insert(0, "ARI_old_vs_clean", round(float(ari), 4))
    table_A6_path = os.path.join(OUTPUT_DIR, "table_A6_ari_and_auroc_comparison.csv")
    table_A6.to_csv(table_A6_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ Appendix Table A6 saved: {table_A6_path}")
    print(table_A6.to_string(index=False))

    # ── Figure A8: crosstab heatmap + AUROC comparison bar chart ──
    fig = plt.figure(figsize=(13, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    ax0 = fig.add_subplot(gs[0])
    old_col_labels = [PHENOTYPE_LABELS.get(int(c), str(c)).replace("Phenotype ", "P")
                      for c in crosstab.columns]
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues",
                xticklabels=old_col_labels,
                yticklabels=[f"Clean {r}" for r in crosstab.index],
                cbar=False, ax=ax0, linewidths=0.5, linecolor="grey")
    ax0.set_xlabel("Original phenotype (12-bin embedding)")
    ax0.set_ylabel("Clean cluster (11-bin embedding, t = -52h masked)")
    ax0.set_title(f"Cluster agreement\nARI = {ari:.3f}  (n = {n_matched})", fontweight="bold")

    ax1 = fig.add_subplot(gs[1])
    plot_order = sorted(table_A6["clean_cluster"].tolist(),
                         key=lambda cc: table_A6.loc[table_A6["clean_cluster"] == cc, "auc_original"].values[0]
                         if not pd.isna(table_A6.loc[table_A6["clean_cluster"] == cc, "auc_original"].values[0])
                         else -1, reverse=True)
    t = table_A6.set_index("clean_cluster").loc[plot_order].reset_index()
    x = np.arange(len(t))
    w = 0.35
    colors = [PHENOTYPE_COLORS.get(int(oc), "#888") if not pd.isna(oc) else "#888"
              for oc in t["orig_cluster_id"]]
    ax1.bar(x - w / 2, t["auc_original"], width=w, color=colors, alpha=0.5, label="Original (12-bin, Figure 14)",
            yerr=[t["auc_original"] - t["ci_lo_original"], t["ci_hi_original"] - t["auc_original"]],
            error_kw=dict(elinewidth=1, ecolor="black", capsize=3))
    ax1.bar(x + w / 2, t["auc_clean"], width=w, color=colors, alpha=0.95, label="Clean (11-bin, sensitivity)",
            yerr=[t["auc_clean"] - t["ci_lo_clean"], t["ci_hi_clean"] - t["auc_clean"]],
            error_kw=dict(elinewidth=1, ecolor="black", capsize=3))
    ax1.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("Phenotype ", "P") for n in t["nearest_original_phenotype"]],
                         rotation=15, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("One-vs-Rest AUROC")
    ax1.set_title("Early prediction AUROC:\noriginal vs. leave-first-bin-out sensitivity check",
                   fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle("Appendix Figure A8. Sensitivity analysis for label circularity in early phenotype prediction",
                  fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "figure_A8_crosstab_and_auroc.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Appendix Figure A8 saved: {fig_path}")

    print(f"\n✅ Sensitivity analysis complete. All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # =========================================================================
    # Run (same model checkpoint as the primary phenotyping pipeline;
    # <EXTUBATION_PROJECT_ROOT> is a placeholder, set the environment variable
    # first, see .env.example):
    #
    # python "<EXTUBATION_PROJECT_ROOT>/clustering/sensitivity_leave_first_bin_out.py"
    # =========================================================================
    main()
