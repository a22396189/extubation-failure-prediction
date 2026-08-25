#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xgb_baseline_pre_extubation_risk_trajectory.py

XGBoost Baseline Model for Pre-Extubation Risk Trajectory Prediction
- Uses ALL 12 time bins (-52 to -8), flattened into a 628-dim feature vector per patient
- Comparable with transformer_pre_extubation_risk_trajectory.py (same split, same metrics)

Feature structure:
  4 static + 26 dynamic × 12 bins + 26 mask × 12 bins = 628 features total

Split: stay_id-level stratified, seed=42, 70/15/15
Metrics: AUROC, AUPRC, Accuracy, Sensitivity, Specificity, Precision, F1_score, Brier_score
         (keys aligned with Transformer output for direct comparison)

Run example:
    python "C:/Users/your-username/Desktop/extubation_failure_prediction/model training/xgb_baseline_pre_extubation_risk_trajectory.py"
    --data_csv "C:/Users/your-username/Desktop/extubation_failure_prediction/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv"
    --output_dir "C:/Users/your-username/Desktop/extubation_failure_prediction/results/xgb"
    --threshold_mode youden
    --n_estimators 500
    --max_depth 5
    --learning_rate 0.05
    --run_shap 0
"""

import os
import argparse
import warnings
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, average_precision_score, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")


# ==================== 1) Feature Config ====================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]
DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]
TARGET = "Extubation_failure"

# 12 time bins: -52, -48, -44, ..., -8  (step=4)
SEQ_TIME_BINS: List[int] = list(range(-52, -4, 4))  # [-52, -48, ..., -8]

# Flat feature names (628 total)
STATIC_FEAT_NAMES: List[str] = ["age", "sex", "BMI", "Charlson_Score"]
DYN_FEAT_NAMES: List[str] = [
    f"{col}_t{tb:+d}" for tb in SEQ_TIME_BINS for col in DYNAMIC_COLS
]  # 12 bins × 26 cols = 312
MASK_FEAT_NAMES: List[str] = [
    f"mask_{col}_t{tb:+d}" for tb in SEQ_TIME_BINS for col in DYNAMIC_COLS
]  # 12 bins × 26 cols = 312
ALL_FLAT_FEAT: List[str] = STATIC_FEAT_NAMES + DYN_FEAT_NAMES + MASK_FEAT_NAMES  # 628


# ==================== Utility ====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


# ==================== 2) Load & Basic Check ====================
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    print("\n" + "=" * 80)
    print("資料載入")
    print("=" * 80)
    print(f"原始資料形狀: {df.shape}")
    if "stay_id" in df.columns:
        print(f"Stay IDs 數量: {df['stay_id'].nunique()}")
    if "time_bin" in df.columns:
        print(f"Time bins 範圍: {df['time_bin'].min()} to {df['time_bin'].max()}")
        print(f"Time bins 唯一值數量: {df['time_bin'].nunique()}")

    # required columns
    required_cols = ["stay_id", "time_bin", TARGET] + STATIC_COLS + DYNAMIC_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要欄位: {missing_cols}")

    # label consistency per stay
    label_consistency = df.groupby("stay_id")[TARGET].nunique()
    if (label_consistency > 1).any():
        inconsistent = label_consistency[label_consistency > 1].index.tolist()
        print(f"警告: 有 {len(inconsistent)} 個 stay_id 的 label 不一致")
        print(f"   範例 stay_ids: {inconsistent[:10]}")
    else:
        print("✓ 所有 stay_id 的 label 一致")

    # overall failure rate by stay
    stay_y = df.groupby("stay_id")[TARGET].first()
    print(f"拔管失敗率 (label=1): {stay_y.mean():.2%}")
    print(f"  失敗案例: {(stay_y == 1).sum()}")
    print(f"  成功案例: {(stay_y == 0).sum()}")

    return df


# ==================== 2.5) Encode categorical ====================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("編碼類別變數")
    print("=" * 80)

    df = df.copy()

    # sex: Male=1, Female=0 (if object)
    if "sex" in df.columns and df["sex"].dtype == "object":
        print("編碼 'sex' 欄位:")
        print(f"  原始值: {df['sex'].unique()[:10]}")
        df["sex"] = (df["sex"] == "Male").astype(int)
        print("  編碼後: Male=1, Female=0")

    # check other object features
    obj_cols = df[STATIC_COLS + DYNAMIC_COLS].select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print(f"警告: 發現其他 object 類別特徵: {obj_cols}")
        for c in obj_cols:
            print(f"  {c} unique: {df[c].unique()[:10]}")
        print("  建議先手動處理（one-hot / mapping）再訓練。")

    return df


# ==================== 3) Split by stay_id ====================
def split_by_stay_id(df: pd.DataFrame, train_ratio: float = 0.7, seed: int = 42):
    """
    Stratified stay_id split: 70% train / 15% val / 15% test.
    Includes subject-level leakage guard if subject_id column is present.
    Returns (train_ids, val_ids, test_ids) as numpy arrays of stay_id values.
    """
    print("\n" + "=" * 80)
    print("資料切分 (按 stay_id 避免 data leakage)")
    print("=" * 80)

    if "subject_id" in df.columns:
        subj_stay_count = df.groupby("subject_id")["stay_id"].nunique()
        multi_stay_count = int((subj_stay_count > 1).sum())
        if multi_stay_count > 0:
            print(f"[WARNING] {multi_stay_count} 個 subject_id 對應多個 stay_id！")
        else:
            print(f"[OK] Subject-level leakage 驗證通過：每個 subject_id 僅對應 1 個 stay_id（共 {len(subj_stay_count)} 位病人）。")

    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()

    print(f"總 stays: {len(stay_labels)}")
    print(f"  失敗(label=1): {(stay_labels[TARGET] == 1).sum()}")
    print(f"  成功(label=0): {(stay_labels[TARGET] == 0).sum()}")

    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"],
        test_size=(1 - train_ratio),
        stratify=stay_labels[TARGET],
        random_state=seed
    )
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"],
        test_size=0.5,
        stratify=temp_labels[TARGET],
        random_state=seed
    )

    def stay_rate(ids) -> float:
        return float(stay_labels[stay_labels["stay_id"].isin(ids)][TARGET].mean())

    print("\n切分結果:")
    print(f"  Train: {len(train_ids):4d} stays, 失敗率={stay_rate(train_ids):.2%}")
    print(f"  Val:   {len(val_ids):4d} stays, 失敗率={stay_rate(val_ids):.2%}")
    print(f"  Test:  {len(test_ids):4d} stays, 失敗率={stay_rate(test_ids):.2%}")

    return train_ids.values, val_ids.values, test_ids.values


# ==================== 4) Build flat features (628-dim) ====================
def build_flat_features(df: pd.DataFrame, stay_ids) -> pd.DataFrame:
    """
    12 time bins (-52 to -8) -> flat feature vector (628 dims) per patient.

    For each stay:
      - 4 static features (taken from first row)
      - 26 dynamic × 12 bins (0.0 when bin absent or value NaN)
      - 26 mask  × 12 bins  (1.0 if value observed, 0.0 if absent/NaN)

    If the CSV already contains precomputed mask_{col} columns (e.g. from
    external imputation), those mask values are used directly; otherwise the
    mask is inferred from NaN presence before imputation.
    """
    has_precomputed_masks = all(f"mask_{c}" in df.columns for c in DYNAMIC_COLS)

    records = []
    for sid in stay_ids:
        d = df[df["stay_id"] == sid].sort_values("time_bin")
        if d.empty:
            continue
        row: Dict = {"stay_id": int(sid), TARGET: int(d[TARGET].iloc[0])}

        # static features
        for col in STATIC_COLS:
            row[col] = float(d[col].iloc[0]) if col in d.columns else 0.0

        # dynamic + mask features per time bin
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

    # fill any remaining NaN in feature columns
    for fn in ALL_FLAT_FEAT:
        if fn in result.columns:
            result[fn] = result[fn].fillna(0.0)

    return result


# ==================== 5) Train XGBoost ====================
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    n_estimators: int = 500,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50
) -> xgb.XGBClassifier:

    print("\n" + "=" * 80)
    print("訓練 XGBoost 模型")
    print("=" * 80)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    print("Class distribution (TRAIN):")
    print(f"  Negative(label=0): {n_neg} ({n_neg/len(y_train)*100:.1f}%)")
    print(f"  Positive(label=1): {n_pos} ({n_pos/len(y_train)*100:.1f}%)")
    print(f"  scale_pos_weight:  {scale_pos_weight:.3f}")

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        tree_method="hist"
    )

    print("\n模型超參數:")
    for k, v in params.items():
        print(f"  {k:22s}: {v}")
    print(f"  early_stopping_rounds: {early_stopping_rounds}")

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
        early_stopping_rounds=early_stopping_rounds
    )

    print("\n✓ 訓練完成")
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"  Best iteration: {model.best_iteration}")
    if hasattr(model, "best_score") and model.best_score is not None:
        print(f"  Best VAL AUROC (during training): {model.best_score:.4f}")

    return model


# ==================== 6) Threshold selection on VAL ====================
def pick_threshold_on_val(
    y_val: np.ndarray,
    p_val: np.ndarray,
    mode: str = "youden",
    min_sens: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Choose threshold on validation set.

    mode:
      - "youden"                  : maximize (tpr - fpr)
      - "f1"                      : maximize F1
      - "f2"                      : maximize F2 (recall-weighted)
      - "min_sens_then_best_spec" : require sensitivity >= min_sens, then maximize specificity

    Returns:
      best_th (float), detail_dict (Dict)
    """
    fpr, tpr, thr = roc_curve(y_val, p_val)

    # remove inf thresholds
    keep = np.isfinite(thr)
    fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    spec = 1 - fpr

    def get_detail(th: float) -> Dict:
        y_hat = (p_val >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spe = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1v = f1_score(y_val, y_hat) if len(np.unique(y_val)) > 1 else 0.0
        return dict(
            threshold=float(th), sens=float(sens), spec=float(spe),
            ppv=float(ppv), npv=float(npv),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), f1=float(f1v)
        )

    if mode == "youden":
        j = tpr - fpr
        idx = int(np.argmax(j))
        best = float(thr[idx])
        det = get_detail(best)
        det["criterion"] = "youden"
        det["youden_J"] = float(j[idx])
        return best, det

    if mode in ("f1", "f2"):
        beta = 1.0 if mode == "f1" else 2.0
        best_score, best_th = -1.0, 0.5
        for th in thr:
            y_hat = (p_val >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_hat).ravel()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            if (prec + rec) == 0:
                score = 0.0
            else:
                score = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
            if score > best_score:
                best_score, best_th = float(score), float(th)
        det = get_detail(best_th)
        det["criterion"] = f"f{int(beta)}"
        det["fbeta"] = float(best_score)
        return best_th, det

    if mode == "min_sens_then_best_spec":
        if min_sens is None:
            raise ValueError("mode=min_sens_then_best_spec requires --min_sens")
        ok = np.where(tpr >= min_sens)[0]
        if len(ok) == 0:
            j = tpr - fpr
            idx = int(np.argmax(j))
            best = float(thr[idx])
            det = get_detail(best)
            det["criterion"] = f"fallback_youden(no tpr >= {min_sens})"
            det["youden_J"] = float(j[idx])
            return best, det
        best_idx = ok[int(np.argmax(spec[ok]))]
        best = float(thr[best_idx])
        det = get_detail(best)
        det["criterion"] = f"min_sens_then_best_spec(min_sens={min_sens})"
        return best, det

    raise ValueError("Unknown mode. Use: youden, f1, f2, min_sens_then_best_spec")


# ==================== 7) Evaluate model on TEST ====================
def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    save_dir: str,
    prefix: str = "test"
) -> Dict:
    """
    Evaluate and produce ROC/PR/CM plots.
    Returns dict with Transformer-aligned keys:
      AUROC, AUPRC, Accuracy, Sensitivity, Specificity, Precision, F1_score, Recall
    """
    print("\n" + "=" * 80)
    print(f"模型評估 ({prefix.upper()} Set, threshold={threshold:.3f})")
    print("=" * 80)

    p = model.predict_proba(X_test)[:, 1]
    y_hat = (p >= threshold).astype(int)

    auroc = roc_auc_score(y_test, p)
    auprc = average_precision_score(y_test, p)
    acc = accuracy_score(y_test, y_hat)
    f1v = f1_score(y_test, y_hat)

    cm = confusion_matrix(y_test, y_hat)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    print("\n主要指標結果:")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {sens:.4f}")
    print(f"  F1 score:    {f1v:.4f}")

    # Plots: CM / ROC / PR
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Pred: Success (0)", "Pred: Failure (1)"],
        yticklabels=["True: Success (0)", "True: Failure (1)"],
        cbar_kws={"label": "Count"}
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title(f"Confusion Matrix\n(threshold={threshold:.3f})", fontweight="bold")

    fpr_c, tpr_c, _ = roc_curve(y_test, p)
    axes[1].plot(fpr_c, tpr_c, color="#1f77b4", lw=3, label=f"XGBoost (AUROC={auroc:.4f})")
    axes[1].fill_between(fpr_c, tpr_c, alpha=0.2, color="#1f77b4")
    axes[1].plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Random")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower right")

    precision_pts, recall_pts, _ = precision_recall_curve(y_test, p)
    axes[2].plot(recall_pts, precision_pts, color="#2ca02c", lw=3, label=f"XGBoost (AUPRC={auprc:.4f})")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Precision-Recall Curve", fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_evaluation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ 評估圖表已儲存: {save_path}")

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Accuracy": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "Precision": float(prec),
        "F1_score": float(f1v),
        "Recall": float(sens),
        # internal extras (not written to performance_metrics.csv directly)
        "_threshold": float(threshold),
        "_npv": float(npv),
        "_confusion_matrix": cm,
        "_tp": int(tp), "_fp": int(fp), "_fn": int(fn), "_tn": int(tn),
        "_p_test": p,
    }


# ==================== 8) Calibration Curve + Brier Score ====================
def plot_calibration_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_dir: str,
    n_bins: int = 10
) -> Dict:
    """
    Plot Calibration Curve (Reliability Diagram) and compute Brier Score.
    Returns {"Brier_score": float}.
    """
    brier = brier_score_loss(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mean_pred, frac_pos, "s-", color="#1f77b4", lw=2,
                 label=f"{model_name} (Brier={brier:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=12)
    axes[0].set_ylabel("Fraction of Positives", fontsize=12)
    axes[0].set_title("Calibration Curve (Reliability Diagram)", fontweight="bold", fontsize=13)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.6, color="steelblue", label="True: Success (0)")
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.6, color="tomato", label="True: Failure (1)")
    axes[1].set_xlabel("Predicted Probability P(label=1)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Predicted Probability Distribution", fontweight="bold", fontsize=13)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, linestyle="--")

    plt.suptitle(f"{model_name} — Calibration Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(save_dir, "calibration_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Brier Score:   {brier:.4f}")
    print(f"✓ Calibration Curve 已儲存: {out_path}")

    return {"Brier_score": float(brier)}


# ==================== 9) SHAP (Explain=VAL, Background=TRAIN) ====================
def analyze_shap(
    model: xgb.XGBClassifier,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    max_display: int,
    bg_samples: int,
    explain_samples: int,
    save_dir: str,
    prefix: str = "val"
) -> Tuple[np.ndarray, pd.DataFrame]:

    print("\n" + "=" * 80)
    print(f"SHAP 分析 (Explain={prefix.upper()}, Background=TRAIN)")
    print("=" * 80)

    def subsample(X: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
        if len(X) > n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), n, replace=False)
            return X[idx]
        return X

    X_bg = subsample(X_background, bg_samples)
    X_exp = subsample(X_explain, explain_samples)

    print(f"Background shape: {X_bg.shape} (sampled to {len(X_bg)})")
    print(f"Explain shape:    {X_exp.shape} (sampled to {len(X_exp)})")
    print("計算 SHAP values（可能需要數分鐘）...")

    explainer = shap.TreeExplainer(
        model.get_booster(),
        data=X_bg,
        feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X_exp)

    print("✓ SHAP 計算完成")
    print(f"  shap_values shape: {np.array(shap_values).shape}")

    shap_arr = np.array(shap_values)  # (n_samples, n_features)

    # top-k by mean(|SHAP|)
    mean_abs_all = np.abs(shap_arr).mean(axis=0)
    topk = min(max_display, len(feature_names))
    top_idx = np.argsort(mean_abs_all)[::-1][:topk]

    feat_top = [feature_names[i] for i in top_idx]
    shap_top = shap_arr[:, top_idx]
    X_top = X_exp[:, top_idx]
    mean_top = mean_abs_all[top_idx]

    # (A) Beeswarm summary
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_top, X_top, feature_names=feat_top, show=False, max_display=topk)
    plt.title("SHAP Summary Plot (Top Features Impact)", fontweight="bold")
    plt.xlabel("SHAP value")
    plt.tight_layout()
    save_path1 = os.path.join(save_dir, f"shap_summary_{prefix}.png")
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP Summary plot 已儲存: {save_path1}")

    # (B) Bar importance
    plt.figure(figsize=(8, 7))
    order = np.argsort(mean_top)
    plt.barh(np.array(feat_top)[order], mean_top[order])
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontweight="bold")
    plt.xlabel("mean(|SHAP|)")
    plt.grid(axis="x", alpha=0.2, linestyle="--")
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f"shap_importance_{prefix}.png")
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP Importance bar 已儲存: {save_path2}")

    # importance table (all features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    csv_path = os.path.join(save_dir, f"shap_feature_importance_{prefix}.csv")
    shap_importance.to_csv(csv_path, index=False)
    print(f"✓ SHAP 重要性已儲存: {csv_path}")

    print("\nTop 15 features:")
    print(shap_importance.head(15).to_string(index=False))

    return shap_values, shap_importance


# ==================== 10) parse_args ====================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="XGBoost baseline: all 12 time bins flattened to 628-dim vector, "
                    "threshold on VAL, metrics aligned with Transformer."
    )

    p.add_argument("--data_csv", type=str, required=True,
                   help="Path to input CSV (already imputed externally).")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Output directory.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_ratio", type=float, default=0.70)

    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1", "f2", "min_sens_then_best_spec"])
    p.add_argument("--min_sens", type=float, default=None,
                   help="Required when threshold_mode=min_sens_then_best_spec, e.g. 0.85")

    # model hyperparameters
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--early_stopping_rounds", type=int, default=50)

    # SHAP
    p.add_argument("--run_shap", type=int, default=0, choices=[0, 1])
    p.add_argument("--shap_max_display", type=int, default=20)
    p.add_argument("--shap_bg_samples", type=int, default=1000)
    p.add_argument("--shap_explain_samples", type=int, default=1000)

    # test ID management
    p.add_argument("--list_test_ids", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_test_ids_csv", type=int, default=0, choices=[0, 1])
    p.add_argument("--list_only", type=int, default=0, choices=[0, 1])
    p.add_argument("--save_predictions", type=int, default=0, choices=[0, 1],
                   help="Save test_predictions.csv (y_true, y_prob) for multi-model curve comparison")

    return p.parse_args()


# ==================== 11) Main ====================
def main(args: argparse.Namespace):
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    print("=" * 80)
    print("XGBoost Baseline — All 12 Time Bins (628-dim flat feature vector)")
    print("=" * 80)
    print("\n設定:")
    print(f"  SEQ_TIME_BINS: {SEQ_TIME_BINS[0]} to {SEQ_TIME_BINS[-1]} ({len(SEQ_TIME_BINS)} bins)")
    print(f"  特徵總數: {len(ALL_FLAT_FEAT)}  "
          f"(static={len(STATIC_FEAT_NAMES)}, "
          f"dynamic={len(DYN_FEAT_NAMES)}, "
          f"mask={len(MASK_FEAT_NAMES)})")
    print(f"  目標變數: {TARGET} (1=失敗, 0=成功)")
    print(f"  threshold_mode: {args.threshold_mode}, min_sens={args.min_sens}")
    print(f"  output_dir: {args.output_dir}")

    # ── 1) Load ──────────────────────────────────────────────────────────────
    df = load_data(args.data_csv)

    # ── 2) Encode categorical ─────────────────────────────────────────────────
    df = encode_categorical_features(df)

    # ── 3) Filter to SEQ_TIME_BINS rows only ──────────────────────────────────
    before = df["stay_id"].nunique()
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
    after = df["stay_id"].nunique()
    print(f"\n[Filter] Kept only SEQ_TIME_BINS rows. "
          f"Stays before={before}, after={after}")

    # ── 4) Split ──────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = split_by_stay_id(
        df, train_ratio=args.train_ratio, seed=args.seed
    )

    # ── 5) (Optional) List / save test IDs ───────────────────────────────────
    if getattr(args, "list_test_ids", 0) == 1 or getattr(args, "save_test_ids_csv", 0) == 1:
        rows = []
        test_labels = df[df["stay_id"].isin(test_ids)].groupby("stay_id")[TARGET].first()
        for sid, lbl in test_labels.items():
            rows.append({"stay_id": int(sid), "label": int(lbl)})
        df_test_ids = (
            pd.DataFrame(rows)
            .sort_values(["label", "stay_id"])
            .reset_index(drop=True)
        )
        fail_ids = df_test_ids[df_test_ids["label"] == 1]["stay_id"].tolist()
        succ_ids = df_test_ids[df_test_ids["label"] == 0]["stay_id"].tolist()
        print(f"[TEST] stay_ids: {len(df_test_ids)} (fail={len(fail_ids)}, success={len(succ_ids)})")
        if getattr(args, "save_test_ids_csv", 0) == 1:
            out_csv = os.path.join(args.output_dir, "test_stay_ids.csv")
            df_test_ids.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"✓ Saved: {out_csv}")

    if getattr(args, "list_only", 0) == 1:
        print("[list_only=1] Test ID 輸出完成，跳過訓練。")
        return

    # ── 6) Build flat feature matrices ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("建立 flat feature matrices (628-dim)")
    print("=" * 80)

    flat_train = build_flat_features(df, train_ids)
    flat_val = build_flat_features(df, val_ids)
    flat_test = build_flat_features(df, test_ids)

    print(f"  flat_train: {flat_train.shape}  失敗率={flat_train[TARGET].mean():.2%}")
    print(f"  flat_val:   {flat_val.shape}  失敗率={flat_val[TARGET].mean():.2%}")
    print(f"  flat_test:  {flat_test.shape}  失敗率={flat_test[TARGET].mean():.2%}")

    X_train = flat_train[ALL_FLAT_FEAT].values.astype(np.float32)
    y_train = flat_train[TARGET].values.astype(int)

    X_val = flat_val[ALL_FLAT_FEAT].values.astype(np.float32)
    y_val = flat_val[TARGET].values.astype(int)

    X_test = flat_test[ALL_FLAT_FEAT].values.astype(np.float32)
    y_test = flat_test[TARGET].values.astype(int)

    # ── 7) Train XGBoost ──────────────────────────────────────────────────────
    model = train_xgboost(
        X_train, y_train, X_val, y_val,
        seed=args.seed,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        early_stopping_rounds=args.early_stopping_rounds
    )

    # ── 8) Choose threshold on VAL ────────────────────────────────────────────
    p_val = model.predict_proba(X_val)[:, 1]
    best_thr, best_thr_detail = pick_threshold_on_val(
        y_val, p_val, mode=args.threshold_mode, min_sens=args.min_sens
    )
    print("\n" + "=" * 80)
    print("VAL threshold selection")
    print("=" * 80)
    for k, v in best_thr_detail.items():
        print(f"  {k:18s}: {v}")

    # ── 9) Evaluate on TEST ───────────────────────────────────────────────────
    metrics = evaluate_model(
        model, X_test, y_test,
        threshold=best_thr,
        save_dir=args.output_dir,
        prefix="test"
    )

    # ── 10) Calibration + Brier Score ────────────────────────────────────────
    p_test = metrics["_p_test"]
    cal_metrics = plot_calibration_brier(
        y_test, p_test,
        model_name="XGBoost",
        save_dir=args.output_dir
    )
    metrics["Brier_score"] = cal_metrics["Brier_score"]

    # ── 11) Print performance summary ────────────────────────────────────────
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
    print("=" * 30)

    # ── 12) Save performance_metrics.csv ─────────────────────────────────────
    perf_keys = [
        "AUROC", "AUPRC", "Accuracy", "Sensitivity",
        "Specificity", "Precision", "F1_score", "Brier_score"
    ]
    perf_rows = [{"Metric": k, "Value": metrics[k]} for k in perf_keys]
    # append threshold and confusion matrix details
    perf_rows += [
        {"Metric": "Threshold", "Value": best_thr},
        {"Metric": "TP", "Value": metrics["_tp"]},
        {"Metric": "FP", "Value": metrics["_fp"]},
        {"Metric": "FN", "Value": metrics["_fn"]},
        {"Metric": "TN", "Value": metrics["_tn"]},
    ]
    perf_df = pd.DataFrame(perf_rows)
    perf_path = os.path.join(args.output_dir, "performance_metrics.csv")
    perf_df.to_csv(perf_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 性能指標已儲存: {perf_path}")

    # ── 12b) Save test predictions（供多模型 ROC/PR 曲線比較用）
    if getattr(args, "save_predictions", 0) == 1:
        pred_df = pd.DataFrame({
            "stay_id": flat_test["stay_id"].values,   # 加入 stay_id 確保跨模型對齊
            "y_true":  y_test.astype(int),
            "y_prob":  p_test,
        })
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"✓ Test predictions 已儲存: {pred_path}")

    # ── 13) Save model ────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "xgboost_baseline_model.json")
    model.save_model(model_path)
    print(f"✓ 模型已儲存: {model_path}")

    # ── 14) SHAP (optional) ───────────────────────────────────────────────────
    if args.run_shap == 1:
        analyze_shap(
            model=model,
            X_background=X_train,
            X_explain=X_val,
            feature_names=ALL_FLAT_FEAT,
            max_display=args.shap_max_display,
            bg_samples=args.shap_bg_samples,
            explain_samples=args.shap_explain_samples,
            save_dir=args.output_dir,
            prefix="val"
        )

    # ── 15) Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("執行完成！輸出檔案：")
    print("=" * 80)
    print(f"  - test_evaluation.png")
    print(f"  - calibration_curve.png")
    print(f"  - performance_metrics.csv")
    print(f"  - xgboost_baseline_model.json")
    if args.run_shap == 1:
        print(f"  - shap_summary_val.png")
        print(f"  - shap_importance_val.png")
        print(f"  - shap_feature_importance_val.csv")
    if getattr(args, "save_test_ids_csv", 0) == 1:
        print(f"  - test_stay_ids.csv")
    print("=" * 80)

    return model, metrics


if __name__ == "__main__":
    # Example:
    #   python "C:/Users/your-username/Desktop/extubation_failure_prediction/model training/xgb_baseline_pre_extubation_risk_trajectory.py"
    #   --data_csv "C:/Users/your-username/Desktop/extubation_failure_prediction/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv"
    #   --output_dir "C:/Users/your-username/Desktop/extubation_failure_prediction/results/xgb"
    #   --threshold_mode youden --n_estimators 500 --max_depth 5 --learning_rate 0.05
    #   --save_predictions 1 --run_shap 0

    args = parse_args()
    try:
        main(args)
        print("\n✓ All tasks completed successfully.")
    except Exception as e:
        print("\n✗ 程式執行失敗:")
        print(f"  錯誤訊息: {e}")
        import traceback
        traceback.print_exc()
