#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
early_cluster_prediction_shap_timebin_minus52.py

Goal:
  用拔管前最早的時間點 (time_bin = -52h) 的臨床特徵，
  訓練 XGBoost 多分類模型預測病人屬於哪個 extubation failure cluster。

  用途（模型可解釋性）：
  - 若能在 52h 前就區分亞型，代表 Transformer 學到的 phenotype 有早期辨別力
  - SHAP 解釋各 cluster 的早期辨別特徵

Outputs (saved in OUTPUT_DIR):
  - early_cluster_timebin-52_merged.csv
  - metrics_auc_ovr.csv
  - auc_ovr_bar.png
  - shap_summary_phenotype{n}.png   （n = Phenotype 編號 1–4，依死亡率排序）
  - shap_top20_phenotype{n}.csv
  - model_xgb_multiclass.json
"""

import os

# Fix: Windows threadpoolctl / MKL DLL 相容性問題（OSError 0xc06d007f）
# 必須在 import xgboost / sklearn 之前設定
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
import shap

# =========================
# Paths
# =========================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
BASE         = r"C:\Users\your-username\Desktop\extubation_failure_prediction"
FEATURES_CSV = rf"{BASE}\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
CLUSTER_CSV  = rf"{BASE}\results\phenotyping\cluster_assignments.csv"
OUTPUT_DIR   = rf"{BASE}\results\phenotyping\early_cluster_pred_timebin_minus52"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_BIN_TARGET = -52
RANDOM_STATE    = 42
TEST_SIZE       = 0.30

# 與 plot_km_poster.py / replot_umap_with_labels.py 保持一致的表型標籤
# 格式：{ KMeans cluster id → 顯示標籤 }（依 28 天死亡率排序）
PHENOTYPE_LABELS = {
    1: "Phenotype 1\n(Critical)",       # C1 → 83.3%
    0: "Phenotype 2\n(High Risk)",      # C0 → 72.8%
    2: "Phenotype 3\n(Moderate Risk)",  # C2 → 50.0%
    3: "Phenotype 4\n(Low Risk)",       # C3 → 40.2%
}
PHENOTYPE_COLORS = {
    1: "#c0392b", 0: "#e67e22", 2: "#2980b9", 3: "#27ae60",
}

# KMeans cluster ID → Phenotype 編號（依 28 天死亡率排序）
# 用於統一檔名與圖標題，與 KM poster / UMAP 圖保持一致
CLUSTER_TO_PHENOTYPE_NUM = {
    1: 1,   # KMeans C1 (83.3%) → Phenotype 1
    0: 2,   # KMeans C0 (72.8%) → Phenotype 2
    2: 3,   # KMeans C2 (50.0%) → Phenotype 3
    3: 4,   # KMeans C3 (40.2%) → Phenotype 4
}

# ⚠️  Data availability note at time_bin = -52h（pre-imputation missingness）：
#   Vital signs / ventilator settings : < 11% missing  → informative
#   GCS                                : 33% missing
#   Blood gas (pH, PaO2, OI, BE)       : 64–68% missing → largely imputed at this time point
#   Labs (Cr, Lactate, PLT, WBC)       : 72–79% missing → largely imputed at this time point
#   Interpretation：features with >50% missingness at -52h should be interpreted cautiously

# =========================
# Feature columns
# =========================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]

DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day",
    "pH", "PaO2", "PaCO2", "BE", "OI",
    "Cr", "WBC", "Hb", "PLT", "AnionGap", "Lactate", "Glucose",
    "io_balance", "Vasopressor_use", "Hemodialysis_use",
]

FEATURE_COLS = STATIC_COLS + DYNAMIC_COLS

FEATURE_DISPLAY_NAME = {
    "age": "Age", "sex": "Sex (Male)", "BMI": "BMI", "Charlson_Score": "Charlson",
    "heart_rate": "HR", "resp_rate": "RR", "spo2": "SpO2", "mbp": "MBP",
    "temperature": "Temp", "GCS": "GCS", "FiO2": "FiO2", "MAP": "Paw_mean",
    "PEEP": "PEEP", "TV_per_kg": "TV/kg", "MV_day": "MV Day",
    "pH": "pH", "PaO2": "PaO2", "PaCO2": "PaCO2", "BE": "BE", "OI": "OI",
    "Cr": "Cr", "WBC": "WBC", "Hb": "Hb", "PLT": "PLT",
    "AnionGap": "AG", "Lactate": "Lactate", "Glucose": "Glucose",
    "io_balance": "IO Balance", "Vasopressor_use": "Vasopressor",
    "Hemodialysis_use": "Hemodialysis",
}

def map_feature_names(cols):
    return [FEATURE_DISPLAY_NAME.get(c, c) for c in cols]

# =========================
# Helpers
# =========================
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def handle_sex(df):
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    return df

def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    w    = np.array([1.0 / freq[yi] for yi in y], dtype=float)
    return w * (len(y) / w.sum())

def bootstrap_auc(y_true_bin, y_prob, n_boot=1000, seed=42):
    """Bootstrap 95% CI for a single OvR AUC."""
    rng  = np.random.default_rng(seed)
    n    = len(y_true_bin)
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


def compute_threshold_metrics(y_true_bin, y_prob):
    """
    Youden's J 最適閾值下的分類效能指標（OvR）。
    回傳 dict: threshold, sensitivity, specificity, PPV, NPV, F1, TP, FP, FN, TN
    """
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob)
    j_idx  = np.argmax(tpr - fpr)
    thresh = float(thresholds[j_idx])
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1   = f1_score(y_true_bin, y_pred, zero_division=0)
    return dict(threshold=round(thresh, 4),
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                PPV=round(ppv, 4), NPV=round(npv, 4), F1=round(f1, 4),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def save_confusion_matrices(cm_data_list, out_png):
    """
    4 個表型 OvR confusion matrix，2×2 排版，附各格數值與百分比。
    cm_data_list: list of dict，每個包含 class、phenotype_num、label、TP/FP/FN/TN、threshold
    """
    PLOT_ORDER  = [1, 0, 2, 3]  # 依死亡率排序
    ordered     = sorted(cm_data_list, key=lambda x: PLOT_ORDER.index(x["class"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes      = axes.flatten()

    for ax, d in zip(axes, ordered):
        cm    = np.array([[d["TN"], d["FP"]], [d["FN"], d["TP"]]])
        total = cm.sum()
        color = PHENOTYPE_COLORS.get(int(d["class"]), "#888888")

        im = ax.imshow(cm, interpolation="nearest",
                       cmap=plt.cm.Blues, vmin=0, vmax=total)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"], fontsize=10)
        ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"], fontsize=10)

        thresh_color = cm.max() / 2.0
        for row in range(2):
            for col in range(2):
                val = cm[row, col]
                pct = 100 * val / total if total > 0 else 0
                ax.text(col, row, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=11,
                        color="white" if val > thresh_color else "black")

        pnum  = d["phenotype_num"]
        label = d["label"]
        sens  = d["sensitivity"]
        spec  = d["specificity"]
        f1    = d["F1"]
        thr   = d["threshold"]
        ax.set_title(
            f"Phenotype {pnum} ({label})\n"
            f"Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}  (thr={thr:.3f})",
            fontsize=10, fontweight="bold", color=color, pad=6)

    fig.suptitle(
        "Confusion Matrices — Early Phenotype Prediction (OvR, Youden Threshold)\n"
        "XGBoost at t = −52 h",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrices saved: {out_png}")


def save_classification_summary_figure(cls_df, out_png):
    """
    水平分組條形圖：4 個表型 × 4 指標（Sensitivity / Specificity / PPV / F1）。
    """
    PLOT_ORDER   = [1, 0, 2, 3]
    LABELS_FIG   = {1: "P1 Critical", 0: "P2 High Risk",
                    2: "P3 Moderate", 3: "P4 Low Risk"}
    metrics      = ["sensitivity", "specificity", "PPV", "F1"]
    metric_label = ["Sensitivity", "Specificity", "PPV", "F1-score"]
    n_metrics    = len(metrics)
    n_pheno      = len(PLOT_ORDER)

    ordered = cls_df.set_index("class").reindex(PLOT_ORDER).reset_index()
    labels  = [LABELS_FIG.get(int(r["class"]), str(r["class"])) for _, r in ordered.iterrows()]
    colors  = [PHENOTYPE_COLORS.get(int(r["class"]), "#888") for _, r in ordered.iterrows()]

    x        = np.arange(n_metrics)
    bar_w    = 0.18
    offsets  = np.linspace(-(n_pheno - 1) / 2, (n_pheno - 1) / 2, n_pheno) * bar_w

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(ordered.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width=bar_w,
                      label=labels[i], color=colors[i], alpha=0.85)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_label, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Classification Performance at Youden Threshold\n"
        "Early Phenotype Prediction (OvR, XGBoost at t = −52 h)",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Classification summary figure saved: {out_png}")


def save_auc_barplot_publication(auc_df, out_png):
    """
    論文品質水平條形圖：
      - 水平排列（高風險 → 低風險，由上到下）
      - AUC 數值標注在 CI 上限右側（不被誤差線遮擋）
      - Bootstrap 95% CI 誤差線
      - 樣本數 (n_pos / n_test) 標注於 y 軸標籤
      - 虛線標示 chance level (0.5)
    """
    # y 軸標籤對應（統一大小寫格式）
    LABELS_FOR_FIGURE = {
        1: "Phenotype 1 (Critical)",
        0: "Phenotype 2 (High risk)",
        2: "Phenotype 3 (Moderate risk)",
        3: "Phenotype 4 (Lower risk)",
    }

    # 依死亡率排序（由上到下：Critical → Lower Risk）
    plot_order = [1, 0, 2, 3]
    df = auc_df.set_index("class").reindex(plot_order).reset_index()

    labels = [
        LABELS_FOR_FIGURE.get(int(row["class"]), f"Cluster {int(row['class'])}")
        + f"\n(n = {int(row['n_pos'])} / {int(row['n_test'])})"
        for _, row in df.iterrows()
    ]
    aucs   = df["auc"].values
    ci_lo  = df["ci_lo"].values
    ci_hi  = df["ci_hi"].values
    colors = [PHENOTYPE_COLORS.get(int(c), "#888") for c in df["class"]]

    xerr_lo = np.where(np.isnan(ci_lo), 0, aucs - ci_lo)
    xerr_hi = np.where(np.isnan(ci_hi), 0, ci_hi - aucs)

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos   = np.arange(len(labels))

    bars = ax.barh(y_pos, aucs, xerr=[xerr_lo, xerr_hi],
                   color=colors, alpha=0.85, height=0.52,
                   error_kw=dict(elinewidth=1.5, ecolor="black", capsize=4))

    # AUC 數值標注：放在 CI 上限右側 + 0.025，避免被誤差線遮擋
    for i, (auc_val, ci_h, bar) in enumerate(zip(aucs, ci_hi, bars)):
        if not np.isnan(auc_val):
            x_pos = (ci_h if not np.isnan(ci_h) else auc_val) + 0.025
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f"{auc_val:.3f}", va="center", ha="left",
                    fontsize=11, fontweight="bold")

    ax.axvline(0.5, color="grey", ls="--", lw=1.5,
               label="Chance (AUROC = 0.50)", zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.15)   # 加寬右側空間讓 AUC 數值不被截斷
    ax.set_xlabel("One-vs-Rest AUROC", fontsize=12)

    # 標題與副標：副標放在主標下方
    # 使用 fig.text 在 figure 座標定位，避免 ax.set_title 與 ax.text 重疊
    plt.tight_layout()   # 先 tight_layout 再取圖形尺寸
    fig.text(0.5, 0.995,
             "Early Phenotype Prediction at $t$ = −52 h",
             ha="center", va="top",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.955,
             "XGBoost using the earliest pre-extubation 4-hour bin",
             ha="center", va="top",
             fontsize=10, color="#555555")
    fig.subplots_adjust(top=0.88)   # 在上方留出主標+副標的空間

    ax.legend(loc="lower right", fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()   # 由上到下：Critical → Lower Risk
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Publication-quality AUC figure saved: {out_png}")

# =========================
# Main
# =========================
def main():
    print("[1] Load data...")
    feats = pd.read_csv(FEATURES_CSV)
    clu   = pd.read_csv(CLUSTER_CSV)

    for c in ["stay_id", "time_bin"]:
        if c not in feats.columns:
            raise ValueError(f"Features CSV missing required column: {c}")
    if "stay_id" not in clu.columns or "cluster" not in clu.columns:
        raise ValueError("Cluster CSV must contain columns: stay_id, cluster")

    # Filter time_bin = -52
    print(f"[2] Filter time_bin == {TIME_BIN_TARGET} ...")
    df = feats[feats["time_bin"] == TIME_BIN_TARGET].copy()
    df = df.sort_values("stay_id").drop_duplicates("stay_id", keep="last")

    # Merge cluster label (inner join → only failure cases with cluster assignment)
    df = df.merge(clu[["stay_id", "cluster"]], on="stay_id", how="inner")
    print(f"  Matched: {len(df)} patients")

    # Check features
    avail_cols  = [c for c in FEATURE_COLS if c in df.columns]
    missing     = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Missing feature columns (will skip): {missing}")

    df = handle_sex(df)
    df = ensure_numeric(df, avail_cols)

    # Save merged table
    merged_path = os.path.join(OUTPUT_DIR, "early_cluster_timebin-52_merged.csv")
    df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {merged_path}")

    # Prepare X / y
    X       = df[avail_cols].copy()
    y       = df["cluster"].astype(int).values
    classes = np.sort(np.unique(y))
    n_classes = len(classes)
    print(f"\n  Clusters in data: {classes.tolist()}  (n_classes={n_classes})")

    # Median imputation
    X = X.fillna(X.median(numeric_only=True))

    # Train / test split（sample 數少，stratify 保持 cluster 比例）
    print("[3] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"  Train={len(X_train)}  Test={len(X_test)}")

    w_train = compute_class_weights(y_train)

    # Train XGBoost multi-class
    print("[4] Train XGBoost multiclass...")
    model = xgb.XGBClassifier(
        objective        = "multi:softprob",
        num_class        = n_classes,
        n_estimators     = 600,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        reg_lambda       = 1.0,
        min_child_weight = 1,
        random_state     = RANDOM_STATE,
        n_jobs           = 1,    # Windows threadpool 相容性（避免 OSError 0xc06d007f）
        eval_metric      = "mlogloss",
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    model_path = os.path.join(OUTPUT_DIR, "model_xgb_multiclass.json")
    model.save_model(model_path)
    print(f"✓ Model saved: {model_path}")

    # One-vs-Rest AUC + Bootstrap 95% CI
    print("[5] Compute One-vs-Rest AUC with Bootstrap 95% CI (n_boot=1000)...")
    proba      = model.predict_proba(X_test)               # (N, K)
    y_test_bin = label_binarize(y_test, classes=classes)   # (N, K)

    auc_rows = []
    for i, c in enumerate(classes):
        col    = y_test_bin[:, i]
        n_pos  = int(col.sum())
        n_test = len(col)
        if n_pos == 0 or n_pos == n_test:
            auc, ci_lo, ci_hi = np.nan, np.nan, np.nan
        else:
            auc   = roc_auc_score(col, proba[:, i])
            ci_lo, ci_hi = bootstrap_auc(col, proba[:, i], n_boot=1000, seed=RANDOM_STATE)
        phenotype = PHENOTYPE_LABELS.get(int(c), f"Cluster {int(c)}").replace("\n", " ")
        auc_rows.append({
            "class":     int(c),
            "phenotype": phenotype,
            "n_test":    n_test,
            "n_pos":     n_pos,
            "auc":       round(float(auc),   4) if not np.isnan(auc)   else np.nan,
            "ci_lo":     round(float(ci_lo), 4) if not np.isnan(ci_lo) else np.nan,
            "ci_hi":     round(float(ci_hi), 4) if not np.isnan(ci_hi) else np.nan,
        })

    auc_df  = pd.DataFrame(auc_rows)
    auc_csv = os.path.join(OUTPUT_DIR, "metrics_auc_ovr.csv")
    auc_df.to_csv(auc_csv, index=False, encoding="utf-8-sig")
    print(f"✓ AUC saved: {auc_csv}")
    print(auc_df[["phenotype","n_test","n_pos","auc","ci_lo","ci_hi"]].to_string(index=False))

    auc_png = os.path.join(OUTPUT_DIR, "auc_ovr_bar.png")
    save_auc_barplot_publication(auc_df, auc_png)

    # Classification metrics at Youden threshold (OvR)
    print("[5b] Compute threshold-dependent metrics (Youden's J)...")
    cls_rows  = []
    cm_inputs = []
    for i, c in enumerate(classes):
        col = y_test_bin[:, i]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        m = compute_threshold_metrics(col, proba[:, i])
        pnum  = CLUSTER_TO_PHENOTYPE_NUM.get(int(c), int(c))
        label = PHENOTYPE_LABELS.get(int(c), f"Cluster {int(c)}").replace("\n", " ")
        cls_rows.append({"class": int(c), "phenotype_num": pnum,
                         "phenotype": label, **m})
        cm_inputs.append({"class": int(c), "phenotype_num": pnum,
                          "label": label.replace("Phenotype . ", "").replace(
                              "Phenotype 1 (Critical)", "Critical").replace(
                              "Phenotype 2 (High Risk)", "High Risk").replace(
                              "Phenotype 3 (Moderate Risk)", "Moderate").replace(
                              "Phenotype 4 (Low Risk)", "Low Risk"),
                          **m})

    cls_df  = pd.DataFrame(cls_rows)
    cls_csv = os.path.join(OUTPUT_DIR, "metrics_classification_youden.csv")
    cls_df.to_csv(cls_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Classification metrics saved: {cls_csv}")
    print(cls_df[["phenotype", "threshold", "sensitivity", "specificity",
                  "PPV", "NPV", "F1"]].to_string(index=False))

    cm_png  = os.path.join(OUTPUT_DIR, "confusion_matrices_ovr.png")
    save_confusion_matrices(cm_inputs, cm_png)

    cls_summary_png = os.path.join(OUTPUT_DIR, "classification_summary_bar.png")
    save_classification_summary_figure(cls_df, cls_summary_png)

    # SHAP (per class)
    print("[6] Compute SHAP values on test set...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Normalize SHAP output format
    if isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

    display_names = map_feature_names(avail_cols)

    print("[7] Save SHAP summary plots...")
    for i, c in enumerate(classes):
        sv = shap_list[i]

        # 用 Phenotype 編號命名檔案與標題，與 KM / UMAP 圖保持一致
        phenotype_num   = CLUSTER_TO_PHENOTYPE_NUM.get(int(c), int(c))
        phenotype_label = PHENOTYPE_LABELS.get(int(c), f"Cluster {int(c)}").replace("\n", " ")

        out_png = os.path.join(OUTPUT_DIR, f"shap_summary_phenotype{phenotype_num}.png")
        plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, X_test.values, feature_names=display_names,
                          show=False, plot_type="dot")
        # 標題：符合 "early prediction of Phenotype X membership" 的命名慣例
        plt.title(
            f"SHAP: Early Prediction of Phenotype {phenotype_num} Membership\n"
            f"({phenotype_label}, Transformer-derived, K-means defined)",
            fontsize=11, fontweight="bold", pad=10
        )
        plt.xlabel(f"SHAP value  (Phenotype {phenotype_num} vs Rest)", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out_png}")

        # Top 20 features by mean |SHAP|
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:20]
        top_df   = pd.DataFrame({
            "feature_raw":     np.array(avail_cols)[top_idx],
            "feature_display": np.array(display_names)[top_idx],
            "mean_abs_shap":   mean_abs[top_idx].round(5),
        })
        out_csv = os.path.join(OUTPUT_DIR, f"shap_top20_phenotype{phenotype_num}.csv")
        top_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n✅ Done. Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
