#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_model_comparison_stats.py

為四個模型（Transformer / LSTM / XGBoost / Random Forest）產生：
  1. Bootstrap 95% CI for AUROC and AUPRC（每個模型）
  2. Pairwise DeLong's test（6 對比較，Bonferroni 校正）
  3. 輸出 Table A5 — 適合直接放入論文 Appendix

【輸出】
  results/comparison/
    table_a5_model_comparison.csv    完整統計表（論文 Table A5）
    table_a5_model_comparison.txt    純文字版（方便複製）

【執行】
  python "%EXTUBATION_PROJECT_ROOT%/model training/compute_model_comparison_stats.py"
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


# =====================================================================
# Paths
# =====================================================================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
BASE        = rf"{EXTUBATION_ROOT}\results"
OUTPUT_DIR  = os.path.join(BASE, "comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PRED_PATHS = {
    "Transformer":   os.path.join(BASE, "transformer",  "test_predictions.csv"),
    "LSTM":          os.path.join(BASE, "lstm",          "test_predictions.csv"),
    "XGBoost":       os.path.join(BASE, "xgb",          "test_predictions.csv"),
    "Random Forest": os.path.join(BASE, "rf",           "test_predictions.csv"),
}

N_BOOTSTRAP = 2000
SEED        = 42

# =====================================================================
# DeLong's Test（精確實作，適用於成對 AUROC 比較）
# 參考：DeLong et al. (1988) Biometrics 44(3):837-845
# =====================================================================

def compute_midrank(x):
    """計算 x 的 midrank（用於 DeLong placement value 計算）"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j + 1] = 0.5 * (i + j + 2)   # 1-indexed midrank
        i = j + 1
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def compute_placement_values(y_true, y_prob):
    """
    計算 DeLong placement values (V10, V01)。
    V10[i]: 第 i 個陽性樣本相對所有陰性樣本的 placement value
    V01[j]: 第 j 個陰性樣本相對所有陽性樣本的 placement value
    """
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    m   = len(pos)
    n   = len(neg)

    # Combined midrank
    combined     = np.concatenate([pos, neg])
    midranks     = compute_midrank(combined)
    pos_midranks = midranks[:m]
    neg_midranks = midranks[m:]

    V10 = (pos_midranks - compute_midrank(pos)) / n
    V01 = (neg_midranks - compute_midrank(neg)) / m
    return V10, V01


def delong_test(y_true, y_prob_a, y_prob_b):
    """
    DeLong's test for comparing two correlated AUROCs.
    y_true: 共用的 ground truth（0/1）
    y_prob_a, y_prob_b: 兩個模型的預測機率

    Returns: z_stat, p_value（雙尾）
    """
    y_true  = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    m = int(y_true.sum())        # 陽性數
    n = int((1 - y_true).sum())  # 陰性數

    V10_a, V01_a = compute_placement_values(y_true, y_prob_a)
    V10_b, V01_b = compute_placement_values(y_true, y_prob_b)

    auc_a = y_prob_a[y_true == 1].mean() - V10_a.mean() + 0.5   # approximation via midrank
    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    # Covariance matrix of (AUC_a, AUC_b)
    S10 = np.cov(np.vstack([V10_a, V10_b]))   # (2,2)
    S01 = np.cov(np.vstack([V01_a, V01_b]))   # (2,2)

    S = S10 / m + S01 / n   # variance-covariance matrix

    diff    = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return 0.0, 1.0

    z     = diff / np.sqrt(var_diff)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))   # 雙尾
    return float(z), float(p_val)


# =====================================================================
# Bootstrap CI
# =====================================================================

def bootstrap_metric(y_true, y_prob, metric_fn, n_boot=2000, seed=42):
    """Bootstrap 95% CI for a scalar metric function."""
    rng    = np.random.default_rng(seed)
    n      = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        yp  = y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        values.append(metric_fn(yt, yp))
    values = np.array(values)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# =====================================================================
# Main
# =====================================================================

def main():
    # ── 1. Load predictions ──────────────────────────────────────────
    print("Loading predictions...")
    raw_preds = {}
    for name, path in MODEL_PRED_PATHS.items():
        if not os.path.exists(path):
            print(f"  [WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        raw_preds[name] = df
        print(f"  {name}: n={len(df)}  has_stay_id={'stay_id' in df.columns}")

    # ── 依 stay_id 對齊（避免不同模型儲存順序不同導致比較錯誤）──
    has_id = all("stay_id" in raw_preds[m].columns for m in raw_preds)
    if has_id:
        print("\n  stay_id 對齊模式：以第一個模型的 stay_id 順序為基準")
        ref_ids = raw_preds[list(raw_preds.keys())[0]]["stay_id"].values
        for name in raw_preds:
            df = raw_preds[name].set_index("stay_id").reindex(ref_ids).reset_index()
            if df["y_true"].isna().any():
                print(f"  [WARN] {name} 有 stay_id 對齊失敗，請重新產生 test_predictions.csv")
            raw_preds[name] = df
        print("  對齊完成")
    else:
        print("\n  [WARN] 部分模型無 stay_id 欄位，無法對齊 → 請重新執行各模型腳本加上 --save_predictions 1")

    preds = {}
    for name, df in raw_preds.items():
        preds[name] = {
            "y_true": df["y_true"].values.astype(int),
            "y_prob": df["y_prob"].values.astype(float),
        }

    model_names = list(preds.keys())

    # ── 2. Per-model AUROC + AUPRC + Bootstrap CI ────────────────────
    print(f"\nComputing Bootstrap CI (n_boot={N_BOOTSTRAP})...")
    rows = []
    for name in model_names:
        yt = preds[name]["y_true"]
        yp = preds[name]["y_prob"]

        auroc      = roc_auc_score(yt, yp)
        auprc      = average_precision_score(yt, yp)
        auc_lo, auc_hi   = bootstrap_metric(yt, yp, roc_auc_score,   N_BOOTSTRAP, SEED)
        prc_lo, prc_hi   = bootstrap_metric(yt, yp, average_precision_score, N_BOOTSTRAP, SEED)

        rows.append({
            "Model":       name,
            "AUROC":       round(auroc, 4),
            "AUROC_CI_lo": round(auc_lo, 4),
            "AUROC_CI_hi": round(auc_hi, 4),
            "AUPRC":       round(auprc, 4),
            "AUPRC_CI_lo": round(prc_lo, 4),
            "AUPRC_CI_hi": round(prc_hi, 4),
        })
        print(f"  {name:15s}  AUROC={auroc:.4f} ({auc_lo:.4f}–{auc_hi:.4f})  "
              f"AUPRC={auprc:.4f} ({prc_lo:.4f}–{prc_hi:.4f})")

    df_metrics = pd.DataFrame(rows)

    # ── 3. Pairwise DeLong's test ─────────────────────────────────────
    print("\nComputing pairwise DeLong's test...")
    pairs         = list(combinations(model_names, 2))
    n_comparisons = len(pairs)   # 6 pairs → Bonferroni factor = 6

    delong_rows = []
    for m_a, m_b in pairs:
        yt   = preds[m_a]["y_true"]           # 共用 ground truth
        yp_a = preds[m_a]["y_prob"]
        yp_b = preds[m_b]["y_prob"]

        auc_a = roc_auc_score(yt, yp_a)
        auc_b = roc_auc_score(yt, yp_b)
        z, p  = delong_test(yt, yp_a, yp_b)
        p_adj = min(p * n_comparisons, 1.0)   # Bonferroni correction

        sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else
              ("*"   if p_adj < 0.05  else "ns"))

        delong_rows.append({
            "Model A":         m_a,
            "Model B":         m_b,
            "AUROC_A":         round(auc_a, 4),
            "AUROC_B":         round(auc_b, 4),
            "ΔAUROC (A−B)":   round(auc_a - auc_b, 4),
            "z":               round(z, 3),
            "p_value":         round(p, 4),
            "p_adj_Bonferroni":round(p_adj, 4),
            "Significant":     sig,
        })
        print(f"  {m_a:15s} vs {m_b:15s}  z={z:+.3f}  p={p:.4f}  p_adj={p_adj:.4f}  {sig}")

    df_delong = pd.DataFrame(delong_rows)

    # ── 4. 組合 Table A5 ──────────────────────────────────────────────
    # Part 1: 各模型效能 + Bootstrap CI
    # Part 2: Pairwise DeLong
    metrics_csv = os.path.join(OUTPUT_DIR, "table_a5_bootstrap_metrics.csv")
    delong_csv  = os.path.join(OUTPUT_DIR, "table_a5_delong_pairwise.csv")
    df_metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    df_delong.to_csv(delong_csv,   index=False, encoding="utf-8-sig")

    # ── 5. 輸出純文字版（方便貼入論文）──────────────────────────────
    txt_path = os.path.join(OUTPUT_DIR, "table_a5_model_comparison.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Table A5. Model Performance Comparison with Statistical Significance\n")
        f.write("=" * 75 + "\n\n")

        f.write("Part A: AUROC and AUPRC with Bootstrap 95% CI\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Model':<18} {'AUROC':>8} {'95% CI':>20} {'AUPRC':>8} {'95% CI':>20}\n")
        f.write("-" * 75 + "\n")
        for _, r in df_metrics.iterrows():
            ci_auc = f"({r['AUROC_CI_lo']:.4f}–{r['AUROC_CI_hi']:.4f})"
            ci_prc = f"({r['AUPRC_CI_lo']:.4f}–{r['AUPRC_CI_hi']:.4f})"
            f.write(f"{r['Model']:<18} {r['AUROC']:>8.4f} {ci_auc:>20} {r['AUPRC']:>8.4f} {ci_prc:>20}\n")

        f.write(f"\n\nPart B: Pairwise DeLong's Test (Bonferroni-corrected, {n_comparisons} comparisons)\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Comparison':<35} {'ΔAUROC':>8} {'z':>7} {'p':>8} {'p_adj':>8} {'Sig':>5}\n")
        f.write("-" * 75 + "\n")
        for _, r in df_delong.iterrows():
            comparison = f"{r['Model A']} vs {r['Model B']}"
            f.write(f"{comparison:<35} {r['ΔAUROC (A−B)']:>+8.4f} "
                    f"{r['z']:>7.3f} {r['p_value']:>8.4f} "
                    f"{r['p_adj_Bonferroni']:>8.4f} {r['Significant']:>5}\n")

        f.write("\nNote: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant\n")
        f.write(f"Bootstrap CI: {N_BOOTSTRAP} resamples (seed={SEED})\n")
        f.write("DeLong's test: two-sided; Bonferroni correction applied for multiple comparisons\n")

    print(f"\n{'='*60}")
    print(f"Saved:")
    print(f"  {metrics_csv}")
    print(f"  {delong_csv}")
    print(f"  {txt_path}")

    # ── 6. 印出 Table A5 文字版 ───────────────────────────────────────
    print()
    with open(txt_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
