#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================
# rf_baseline_pre_extubation_risk_trajectory.py
#
# 【目的】
#   以 Random Forest 預測 ICU 拔管失敗風險（label=1），
#   作為 Transformer 模型的 baseline 比較對象。
#
# 【與 Transformer 的對齊設計】
#   - 相同 Cohort：extubation_features_imputed_gap4_52to4.csv（N=6,761）
#   - 相同 train/val/test split：stay_id 切分，seed=42，70/15/15
#   - 相同特徵來源：26 dynamic features（12 time bins）+ 4 static features
#   - 相同 imputation：讀取預先填補的 CSV，不在模型內部重新填補
#   - 相同 threshold 選擇：Youden Index on val set
#   - 相同輸出指標格式：AUROC / AUPRC / Sensitivity / Specificity / F1 / Brier
#
# 【RF 特徵工程（對應 transformer_pre_extubation_risk_trajectory.py 的 12 bins）】
#   RF 直接使用 ALL 12 time bins 展開為 per-bin flat vector，與 XGB 及
#   Transformer 完全可比較。共 628 個特徵：
#     - 4 個靜態特徵
#     - 26 動態特徵 × 12 bins = 312 個動態特徵
#     - 26 遮罩特徵 × 12 bins = 312 個遮罩特徵
#   （不再使用 mean/last/std 聚合；此版本對應 XGB flat-vector 設計）
#
# 【輸入】
#   - extubation_features_imputed_gap4_52to4.csv：已填補的完整特徵表
#     （由 impute_extubation_features_gap4_52to4.py 產生）
#
# 【輸出（在 --output_dir 下）】
#   - best_rf.joblib：訓練好的模型
#   - test_roc_cm.png：ROC + 混淆矩陣
#   - calibration_curve.png：Calibration Curve + 機率分布
#   - rf_shap_summary.png：SHAP feature importance
#   - performance_metrics.csv：完整指標（與 Transformer 格式一致）
#
# 【執行步驟】
#   Step 1：讀取預填補 CSV，sex 編碼，確認欄位
#   Step 2：stay_id 切分（與 Transformer 完全相同的 split）
#   Step 3：（選用）輸出 / 儲存 test stay IDs
#   Step 4：將 12 bins 展開為 per-bin flat feature vector（628 features）
#   Step 5：訓練 Random Forest
#   Step 6：Youden threshold on val set
#   Step 7：Test set 評估（統一輸出格式）
#   Step 8：Calibration curve + Brier Score
#   Step 9：SHAP analysis
# =============================================================

import os
import argparse
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, average_precision_score,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

import shap
import joblib

warnings.filterwarnings("ignore")

# =============================================================
# 1. Feature Config（與 Transformer / XGB 完全一致）
# =============================================================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]

DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]

SEQ_TIME_BINS = list(range(-52, -4, 4))  # 12 bins: -52, -48, ..., -8
TARGET        = "Extubation_failure"

# Flat feature naming (identical to XGB convention)
STATIC_FEAT_NAMES = STATIC_COLS                                                          # 4
DYN_FEAT_NAMES    = [f"{col}_t{tb:+d}" for tb in SEQ_TIME_BINS for col in DYNAMIC_COLS] # 312
MASK_FEAT_NAMES   = [f"mask_{col}_t{tb:+d}" for tb in SEQ_TIME_BINS for col in DYNAMIC_COLS]  # 312
ALL_FLAT_FEAT     = STATIC_FEAT_NAMES + DYN_FEAT_NAMES + MASK_FEAT_NAMES                # 628


# =============================================================
# 2. Utility
# =============================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================================================
# 3. Load Data
# =============================================================
def load_data(filepath: str) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1：讀取資料")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"  讀取完成：{len(df):,} 列，{df['stay_id'].nunique():,} 位病人")
    print(f"   欄位數：{df.shape[1]}")

    # sex 編碼（與 Transformer 一致）
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
        print("   sex 欄位已編碼：male=1, female=0")

    df[TARGET] = df[TARGET].astype(int)

    # 篩選 SEQ_TIME_BINS
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
    print(f"   篩選 SEQ_TIME_BINS 後：{len(df):,} 列")

    # 確認必要欄位存在
    missing = [c for c in STATIC_COLS + DYNAMIC_COLS + ["stay_id", "time_bin", TARGET]
               if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}")

    return df


# =============================================================
# 4. Split（與 Transformer 完全相同）
# =============================================================
def split_by_stay_id(df: pd.DataFrame, train_ratio=0.7, seed=42):
    if "subject_id" in df.columns:
        subj_stay_count = df.groupby("subject_id")["stay_id"].nunique()
        multi_stay_count = int((subj_stay_count > 1).sum())
        if multi_stay_count > 0:
            print(f"[WARNING] {multi_stay_count} 個 subject_id 對應多個 stay_id！")
        else:
            print(f"[OK] Subject-level leakage 驗證通過：每個 subject_id 僅對應 1 個 stay_id（共 {len(subj_stay_count)} 位病人）。")
    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    train_ids, temp_ids = train_test_split(stay_labels["stay_id"], test_size=(1-train_ratio), stratify=stay_labels[TARGET], random_state=seed)
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(temp_labels["stay_id"], test_size=0.5, stratify=temp_labels[TARGET], random_state=seed)
    train_df = df[df["stay_id"].isin(train_ids)].copy()
    val_df   = df[df["stay_id"].isin(val_ids)].copy()
    test_df  = df[df["stay_id"].isin(test_ids)].copy()
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = d["stay_id"].nunique()
        fr = d.groupby("stay_id")[TARGET].first().mean()
        print(f"   {name}: {n:,} stays (failure rate = {fr:.3f})")
    return train_df, val_df, test_df


# =============================================================
# 5. Build flat features — 12 time bins → flat vector (628 dims)
# =============================================================
def build_flat_features(df: pd.DataFrame, stay_ids) -> pd.DataFrame:
    """12 time bins -> flat feature vector (628 dims) per patient.

    Feature layout (identical to XGB):
      - 4  static features
      - 26 dynamic features x 12 bins = 312
      - 26 mask   features  x 12 bins = 312
      Total: 628
    """
    has_precomputed_masks = all(f"mask_{c}" in df.columns for c in DYNAMIC_COLS)
    records = []
    for sid in stay_ids:
        d = df[df["stay_id"] == sid].sort_values("time_bin")
        if d.empty:
            continue
        row = {"stay_id": int(sid), TARGET: int(d[TARGET].iloc[0])}
        for col in STATIC_COLS:
            row[col] = float(d[col].iloc[0]) if col in d.columns else 0.0
        for tb in SEQ_TIME_BINS:
            bin_tag = f"t{tb:+d}"
            trow = d[d["time_bin"] == tb]
            if trow.empty:
                for col in DYNAMIC_COLS:
                    row[f"{col}_{bin_tag}"] = 0.0
                    row[f"mask_{col}_{bin_tag}"] = 0.0
            else:
                trow0 = trow.iloc[0]
                for col in DYNAMIC_COLS:
                    val = trow0[col] if col in trow0.index else np.nan
                    row[f"{col}_{bin_tag}"] = 0.0 if pd.isna(val) else float(val)
                    if has_precomputed_masks:
                        mval = trow0.get(f"mask_{col}", 0.0)
                        row[f"mask_{col}_{bin_tag}"] = float(mval)
                    else:
                        row[f"mask_{col}_{bin_tag}"] = 0.0 if pd.isna(val) else 1.0
        records.append(row)
    result = pd.DataFrame(records)
    for fn in ALL_FLAT_FEAT:
        if fn in result.columns:
            result[fn] = result[fn].fillna(0.0)
    return result


# =============================================================
# 6. Train Random Forest
# =============================================================
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        seed=42, n_estimators=500,
                        max_depth=10, min_samples_leaf=5) -> RandomForestClassifier:
    print("\n" + "=" * 60)
    print("Step 5：訓練 Random Forest")
    print("=" * 60)
    print(f"   特徵數：{X_train.shape[1]}  |  訓練樣本：{X_train.shape[0]:,}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  訓練完成")
    return model


# =============================================================
# 7. Threshold Selection（Youden on val，與 Transformer 一致）
# =============================================================
def find_best_threshold(y_val, p_val, mode="youden") -> float:
    fpr, tpr, thr = roc_curve(y_val, p_val)
    if mode == "youden":
        idx = np.argmax(tpr - fpr)
        return float(thr[idx])
    elif mode == "f1":
        prec, rec, thr_f = precision_recall_curve(y_val, p_val)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        return float(thr_f[np.argmax(f1[:-1])])
    elif mode == "f2":
        prec, rec, thr_f = precision_recall_curve(y_val, p_val)
        f2 = 5 * prec * rec / (4 * prec + rec + 1e-8)
        return float(thr_f[np.argmax(f2[:-1])])
    else:
        return 0.5


# =============================================================
# 8. Evaluate（輸出格式與 Transformer 完全一致）
# =============================================================
def evaluate_model(y_true, y_prob, threshold, output_dir, model_name="Random Forest"):
    y_hat = (y_prob >= threshold).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc   = accuracy_score(y_true, y_hat)
    f1v   = f1_score(y_true, y_hat)

    cm = confusion_matrix(y_true, y_hat)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0

    metrics = {
        "AUROC":       float(auroc),
        "AUPRC":       float(auprc),
        "Accuracy":    float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Precision":   float(prec),
        "F1_score":    float(f1v),
        "Recall":      float(sens),
    }

    # ROC + 混淆矩陣圖
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr_c, tpr_c, color="#1f77b4", lw=3,
                 label=f"{model_name} (AUROC={auroc:.4f})")
    axes[0].fill_between(fpr_c, tpr_c, alpha=0.2, color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    axes[0].set_title("ROC Curve", fontweight="bold", fontsize=14)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_xticklabels(["Pred: Success (0)", "Pred: Failure (1)"])
    axes[1].set_yticklabels(["True: Success (0)", "True: Failure (1)"], va="center")
    axes[1].set_title(f"Confusion Matrix\n(threshold={threshold:.4f})",
                      fontweight="bold", fontsize=14)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "test_roc_cm.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ROC + Confusion Matrix 已儲存：{save_path}")

    return metrics


# =============================================================
# 9. Calibration Curve + Brier Score（與 Transformer 一致）
# =============================================================
def plot_calibration_brier(y_true, y_prob, model_name, output_dir, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob,
                                            n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mean_pred, frac_pos, "s-", color="#1f77b4", lw=2,
                 label=f"{model_name} (Brier={brier:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=12)
    axes[0].set_ylabel("Fraction of Positives", fontsize=12)
    axes[0].set_title("Calibration Curve", fontweight="bold", fontsize=13)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.6,
                 color="steelblue", label="True: Success (0)")
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.6,
                 color="tomato", label="True: Failure (1)")
    axes[1].set_xlabel("Predicted Probability P(label=1)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Predicted Probability Distribution",
                      fontweight="bold", fontsize=13)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.suptitle(f"{model_name} — Calibration Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Brier Score: {brier:.4f}  (儲存至: {out_path})")
    return {"Brier_score": float(brier)}


# =============================================================
# 10. SHAP Analysis
# =============================================================
def analyze_shap(model, X_val: np.ndarray, feature_names: List[str],
                 output_dir: str, max_display: int = 20):
    print("\n[SHAP] 計算 feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)

    # RF 回傳 [class0, class1]，取 class1
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_val, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    out = os.path.join(output_dir, "rf_shap_summary.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary 已儲存：{out}")


# =============================================================
# 11. Main
# =============================================================
def main(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Step 1: 讀取資料
    df = load_data(args.data_csv)

    # Step 2: Split（與 Transformer 相同）
    print("\n" + "=" * 60)
    print("Step 2：資料切分")
    print("=" * 60)
    train_df, val_df, test_df = split_by_stay_id(df, train_ratio=0.7, seed=args.seed)

    # Step 3: (optional) save / list test IDs
    if getattr(args, "list_test_ids", 0) == 1 or getattr(args, "save_test_ids_csv", 0) == 1:
        rows = []
        test_labels = test_df.groupby("stay_id")[TARGET].first()
        for sid, lbl in test_labels.items():
            rows.append({"stay_id": int(sid), "label": int(lbl)})
        df_test_ids = pd.DataFrame(rows).sort_values(["label", "stay_id"]).reset_index(drop=True)
        fail_ids = df_test_ids[df_test_ids["label"] == 1]["stay_id"].tolist()
        succ_ids = df_test_ids[df_test_ids["label"] == 0]["stay_id"].tolist()
        print(f"[TEST] stay_ids: {len(df_test_ids)} (fail={len(fail_ids)}, success={len(succ_ids)})")
        if getattr(args, "save_test_ids_csv", 0) == 1:
            out_csv = os.path.join(args.output_dir, "test_stay_ids.csv")
            df_test_ids.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"  Saved: {out_csv}")

    if getattr(args, "list_only", 0) == 1:
        print("[list_only=1] Test ID 輸出完成，跳過訓練。")
        return

    # Step 4: 12 bins → flat feature vector (628 features)
    print("\n" + "=" * 60)
    print("Step 4：建立 per-bin flat feature vector（628 features）")
    print("=" * 60)
    print(f"   特徵結構：4 static + 26 dynamic × 12 bins + 26 mask × 12 bins = {len(ALL_FLAT_FEAT)}")

    train_ids = train_df["stay_id"].unique()
    val_ids   = val_df["stay_id"].unique()
    test_ids  = test_df["stay_id"].unique()

    train_flat = build_flat_features(train_df, train_ids)
    val_flat   = build_flat_features(val_df,   val_ids)
    test_flat  = build_flat_features(test_df,  test_ids)

    # Ensure ALL_FLAT_FEAT columns are present (fill missing with 0)
    for flat_df in [train_flat, val_flat, test_flat]:
        for fn in ALL_FLAT_FEAT:
            if fn not in flat_df.columns:
                flat_df[fn] = 0.0

    X_train = train_flat[ALL_FLAT_FEAT].values
    y_train = train_flat[TARGET].values.astype(int)
    X_val   = val_flat[ALL_FLAT_FEAT].values
    y_val   = val_flat[TARGET].values.astype(int)
    X_test  = test_flat[ALL_FLAT_FEAT].values
    y_test  = test_flat[TARGET].values.astype(int)

    print(f"   Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Step 5: 訓練
    model = train_random_forest(
        X_train, y_train, seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf
    )

    # 儲存模型
    model_path = os.path.join(args.output_dir, "best_rf.joblib")
    joblib.dump(model, model_path)
    print(f"  模型已儲存：{model_path}")

    # Step 6: Threshold on val
    print("\n" + "=" * 60)
    print("Step 6：Threshold selection on Val Set")
    print("=" * 60)
    p_val = model.predict_proba(X_val)[:, 1]
    best_thr = find_best_threshold(y_val, p_val, mode=args.threshold_mode)
    val_auroc = roc_auc_score(y_val, p_val)
    print(f"[VAL] AUROC={val_auroc:.4f} | threshold_mode={args.threshold_mode}"
          f" -> best_thr={best_thr:.4f}")

    # Step 7: Test 評估
    print("\n" + "=" * 60)
    print("Step 7：Test Set 評估")
    print("=" * 60)
    p_test = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, p_test, best_thr, args.output_dir)

    # Step 8: Calibration + Brier
    cal_metrics = plot_calibration_brier(
        y_test, p_test, model_name="Random Forest", output_dir=args.output_dir
    )
    metrics.update(cal_metrics)

    # 輸出格式（對齊 Transformer）
    print(f"\n{'='*30}")
    print(f"測試集性能指標 (Threshold={best_thr:.4f}):")
    print(f"AUROC       : {metrics['AUROC']:.4f}")
    print(f"AUPRC       : {metrics['AUPRC']:.4f}")
    print(f"Accuracy    : {metrics['Accuracy']:.4f}")
    print(f"Sensitivity : {metrics['Sensitivity']:.4f}")
    print(f"Specificity : {metrics['Specificity']:.4f}")
    print(f"Precision   : {metrics['Precision']:.4f}")
    print(f"F1 score    : {metrics['F1_score']:.4f}")
    print(f"Brier Score : {metrics['Brier_score']:.4f}")
    print('='*30)

    # 儲存 performance_metrics.csv（格式與 Transformer 一致）
    res_df = pd.DataFrame([metrics])
    csv_out = os.path.join(args.output_dir, "performance_metrics.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"  完整指標已儲存至：{csv_out}")

    # 儲存 test set 預測值（供多模型 ROC/PR 曲線比較用）
    if getattr(args, "save_predictions", 0) == 1:
        pred_df = pd.DataFrame({
            "stay_id": test_flat["stay_id"].values,   # 加入 stay_id 確保跨模型對齊
            "y_true":  y_test.astype(int),
            "y_prob":  p_test,
        })
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"  Test predictions 已儲存至：{pred_path}")

    # Step 9: SHAP
    if args.run_shap:
        analyze_shap(model, X_val, ALL_FLAT_FEAT, args.output_dir, args.shap_max_display)


# =============================================================
# 12. Args
# =============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Random Forest baseline — 628-feature flat vector, comparable with XGB and Transformer"
    )
    p.add_argument("--data_csv", type=str, required=True,
                   help="Path to extubation_features_imputed_gap4_52to4.csv")
    p.add_argument("--output_dir", type=str, default="results_rf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1", "f2"])
    # RF hyperparameters
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--min_samples_leaf", type=int, default=5)
    # SHAP
    p.add_argument("--run_shap", type=int, default=1, choices=[0, 1])
    p.add_argument("--shap_max_display", type=int, default=20)
    # Test ID management (same as Transformer script)
    p.add_argument("--list_test_ids", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_test_ids_csv", type=int, default=0, choices=[0, 1])
    p.add_argument("--list_only", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_predictions", type=int, default=0, choices=[0, 1],
                   help="Save test_predictions.csv (y_true, y_prob) for multi-model curve comparison")
    return p.parse_args()


if __name__ == "__main__":
    """
    Run example:

    python "C:/Users/your-username/Desktop/extubation_project_code_review/model training/rf_baseline_pre_extubation_risk_trajectory.py" \
      --data_csv "C:/Users/your-username/Desktop/extubation_project_code_review/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
      --output_dir "C:/Users/your-username/Desktop/extubation_project_code_review/results/rf" \
      --threshold_mode youden \
      --n_estimators 500 --max_depth 10 \
      --run_shap 1
    """
    main(parse_args())
