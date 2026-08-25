#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_model_comparison_curves.py

將四個模型的 ROC 曲線與 Precision-Recall 曲線繪製在同一張圖上，
供論文使用。

【前置條件】
  執行以下指令，產生各模型的 test_predictions.csv：

  # Random Forest
  python "model training/rf_baseline_pre_extubation_risk_trajectory.py" \
    --data_csv "data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --output_dir "results/rf" --save_predictions 1 --run_shap 0

  # XGBoost
  python "model training/xgb_baseline_pre_extubation_risk_trajectory.py" \
    --data_csv "data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --output_dir "results/xgb" --save_predictions 1 --run_shap 0

  # LSTM
  python "model training/lstm_pre_extubation_risk_trajectory.py" \
    --data_csv "data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --output_dir "results/lstm" --save_predictions 1

  # Transformer（若尚未產生，加 --save_predictions 1）
  python "model training/transformer_pre_extubation_risk_trajectory.py" \
    --data_csv "data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --output_dir "results/transformer" --save_predictions 1 \
    --use_time_weights 1 --use_causal_mask 1

【執行方式】
  # 使用預設路徑（results/rf、results/xgb、results/lstm、results/transformer）
  python "model training/plot_model_comparison_curves.py"

  # 自訂各模型結果路徑與輸出位置
  python "model training/plot_model_comparison_curves.py" \
    --rf_dir        results/rf \
    --xgb_dir       results/xgb \
    --lstm_dir      results/lstm \
    --transformer_dir results/transformer \
    --output_dir    results/comparison

【輸出】
  - roc_comparison.png          四模型 ROC 曲線
  - pr_comparison.png           四模型 Precision-Recall 曲線
  - roc_pr_comparison.png       合圖（ROC + PR 並排，論文用）
  - model_performance_summary.csv  四模型指標摘要
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# =====================================================================
# 預設路徑（相對於此腳本所在的 model training/ 資料夾）
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RF_DIR          = os.path.join(BASE_DIR, "results", "rf")
DEFAULT_XGB_DIR         = os.path.join(BASE_DIR, "results", "xgb")
DEFAULT_LSTM_DIR        = os.path.join(BASE_DIR, "results", "lstm")
DEFAULT_TRANSFORMER_DIR = os.path.join(BASE_DIR, "results", "transformer")
DEFAULT_OUTPUT_DIR      = os.path.join(BASE_DIR, "results", "comparison")

# =====================================================================
# 模型顯示設定（名稱、顏色、線型）
# =====================================================================
MODEL_STYLES = [
    {
        "key":       "rf",
        "label":     "Random Forest",
        "color":     "#2CA02C",   # green
        "linestyle": "--",
        "linewidth": 2.0,
        "zorder":    2,
    },
    {
        "key":       "xgb",
        "label":     "XGBoost",
        "color":     "#FF7F0E",   # orange
        "linestyle": "-.",
        "linewidth": 2.0,
        "zorder":    3,
    },
    {
        "key":       "lstm",
        "label":     "LSTM",
        "color":     "#9467BD",   # purple
        "linestyle": ":",
        "linewidth": 2.2,
        "zorder":    4,
    },
    {
        "key":       "transformer",
        "label":     "Transformer",
        "color":     "#D62728",   # red
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder":    5,
    },
]


# =====================================================================
# 讀取預測值
# =====================================================================
def load_predictions(pred_dir: str, model_key: str) -> dict | None:
    """
    從 pred_dir/test_predictions.csv 讀取 y_true 與 y_prob。
    若檔案不存在則回傳 None 並印出警告。
    """
    csv_path = os.path.join(pred_dir, "test_predictions.csv")
    if not os.path.exists(csv_path):
        print(f"  ⚠️  [{model_key}] 找不到 test_predictions.csv：{csv_path}")
        print(f"       請先執行對應的訓練腳本並加上 --save_predictions 1")
        return None

    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        print(f"  ⚠️  [{model_key}] test_predictions.csv 缺少 y_true 或 y_prob 欄位")
        return None

    y_true = df["y_true"].values.astype(int)
    y_prob = df["y_prob"].values.astype(float)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    print(f"  ✅ [{model_key:12s}] N={len(y_true)}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return {"y_true": y_true, "y_prob": y_prob, "auroc": auroc, "auprc": auprc}


# =====================================================================
# 繪圖函數
# =====================================================================

def plot_roc(ax, models_data: list, title: str = "ROC Curves") -> None:
    """在 ax 上繪製多模型 ROC 曲線。"""
    # Random guess baseline
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            linewidth=1.2, alpha=0.6, label="Random (AUROC = 0.50)", zorder=1)

    for item in models_data:
        if item["data"] is None:
            continue
        style = item["style"]
        d     = item["data"]
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_prob"])
        label = f"{style['label']} (AUROC = {d['auroc']:.3f})"
        ax.plot(fpr, tpr,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label,
                zorder=style["zorder"])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("1 – Specificity (False Positive Rate)", fontsize=12)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_aspect("equal")


def plot_pr(ax, models_data: list, prevalence: float,
            title: str = "Precision-Recall Curves") -> None:
    """在 ax 上繪製多模型 PR 曲線。"""
    # No-skill baseline（水平虛線 = 陽性率）
    ax.axhline(y=prevalence, color="grey", linestyle="--",
               linewidth=1.2, alpha=0.6,
               label=f"No Skill (AUPRC = {prevalence:.2f})", zorder=1)

    for item in models_data:
        if item["data"] is None:
            continue
        style = item["style"]
        d     = item["data"]
        prec, rec, _ = precision_recall_curve(d["y_true"], d["y_prob"])
        label = f"{style['label']} (AUPRC = {d['auprc']:.3f})"
        ax.plot(rec, prec,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                label=label,
                zorder=style["zorder"])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0,   1.05)
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--")


# =====================================================================
# 儲存 performance summary CSV
# =====================================================================
def save_summary(models_data: list, output_dir: str) -> None:
    rows = []
    for item in models_data:
        if item["data"] is None:
            continue
        d = item["data"]
        rows.append({
            "Model":  item["style"]["label"],
            "AUROC":  round(d["auroc"], 4),
            "AUPRC":  round(d["auprc"], 4),
            "N_test": len(d["y_true"]),
        })
    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(output_dir, "model_performance_summary.csv")
        df.to_csv(out, index=False)
        print(f"\n📋 Performance summary saved: {out}")
        print(df.to_string(index=False))


# =====================================================================
# 主程式
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot ROC and PR curves for RF / XGBoost / LSTM / Transformer"
    )
    p.add_argument("--rf_dir",          type=str, default=DEFAULT_RF_DIR)
    p.add_argument("--xgb_dir",         type=str, default=DEFAULT_XGB_DIR)
    p.add_argument("--lstm_dir",        type=str, default=DEFAULT_LSTM_DIR)
    p.add_argument("--transformer_dir", type=str, default=DEFAULT_TRANSFORMER_DIR)
    p.add_argument("--output_dir",      type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dpi",             type=int, default=300,
                   help="Output image resolution (default 300 dpi)")
    p.add_argument("--font_family",     type=str, default="Arial",
                   help="Font family for figures (default Arial)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 字體設定
    plt.rcParams.update({
        "font.family":      args.font_family,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "figure.dpi":       100,
    })

    # 目錄對應
    dir_map = {
        "rf":          args.rf_dir,
        "xgb":         args.xgb_dir,
        "lstm":        args.lstm_dir,
        "transformer": args.transformer_dir,
    }

    print("=" * 60)
    print("Loading test predictions …")
    print("=" * 60)

    models_data = []
    all_y_true  = []
    for style in MODEL_STYLES:
        key  = style["key"]
        data = load_predictions(dir_map[key], key)
        models_data.append({"style": style, "data": data})
        if data is not None:
            all_y_true.extend(data["y_true"].tolist())

    # 計算陽性率（No-Skill PR baseline）
    prevalence = np.mean(all_y_true) if all_y_true else 0.4
    n_available = sum(1 for m in models_data if m["data"] is not None)

    if n_available == 0:
        print("\n❌ 找不到任何 test_predictions.csv，請先執行各模型的訓練腳本。")
        return

    print(f"\nPositive class prevalence (for PR baseline): {prevalence:.3f}")
    print(f"Available models: {n_available}/4\n")

    # ── 1. 個別 ROC 圖 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_roc(ax, models_data, title="ROC Curves — Four Model Comparison")
    fig.tight_layout()
    roc_path = os.path.join(args.output_dir, "roc_comparison.png")
    fig.savefig(roc_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ ROC curve saved:          {roc_path}")

    # ── 2. 個別 PR 圖 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_pr(ax, models_data, prevalence,
            title="Precision-Recall Curves — Four Model Comparison")
    fig.tight_layout()
    pr_path = os.path.join(args.output_dir, "pr_comparison.png")
    fig.savefig(pr_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ PR curve saved:           {pr_path}")

    # ── 3. 合圖（ROC + PR 並排，論文用）────────────────────────────
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 6))

    plot_roc(ax_roc, models_data, title="(A) ROC Curves")
    plot_pr(ax_pr,  models_data, prevalence, title="(B) Precision-Recall Curves")

    fig.suptitle(
        "Model Performance Comparison — Pre-Extubation Failure Prediction",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    combined_path = os.path.join(args.output_dir, "roc_pr_comparison.png")
    fig.savefig(combined_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Combined figure saved:    {combined_path}")

    # ── 4. Performance summary CSV ──────────────────────────────────
    save_summary(models_data, args.output_dir)

    print(f"\n📁 All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
