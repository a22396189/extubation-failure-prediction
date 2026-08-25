"""
Table 4「FiO2」（snapshot, time_bin == -8）與 Table 5「FiO2, trend per 4h」
註腳用描述性統計。

背景：
    論文 4.1 Descriptive Statistics 中，Table 4（拔管前 4-8h snapshot）與
    Table 5（48h 觀測窗摘要）的 FiO2 相關列，皆出現「兩組 median (Q1, Q3)
    四捨五入後顯示相同，但 Wilcoxon rank-sum p < 0.001」的現象。正文
    p.45-46 已有文字說明，但表格本身的註腳看不到，讀者容易誤以為矛盾。
    本檔案補算兩個具體數字，供 Table 4 / Table 5 註腳引用：
        1. 分布是否偏向某一側（上尾／下尾）的比例
        2. rank-biserial correlation 效應量

資料來源與算法：
    - 皆使用「補值前」的 extubation_features_enhanced_gap4_52to4.csv
      （對應 extubation_stats_analysis_final.R 第 12 行：
       「來源：extubation_features_enhanced（補值前，真實 NaN）」）。
      Table 4 / Table 5 的 p 值原本就是用這份未補值資料算出來的，這裡沿用
      同一份資料，才能讓引用的比例／效應量與原始檢定的資料基礎一致。
    - Table 4：直接取 time_bin == -8（拔管決策前 4-8h）當筆 FiO2，
      對應 R 腳本 df_snapshot／table2。
    - Table 5：逐病人以 time_bin 對 FiO2 做線性迴歸，斜率 ×4（每 4 小時
      的變化量），至少需 3 筆非缺失觀測，否則為 NaN，對應 R 腳本
      calc_slope(..., min_obs = 3)／df_window／table3。
    - 兩者皆用 Mann-Whitney U（等價於 R 的 wilcox.test）取得 p 值與
      rank-biserial correlation = 1 - 2U / (n1 * n2)。
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


ENHANCED_CSV = (
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
    rf"{EXTUBATION_ROOT}"
    r"\data\outputs\gap4_52to4\extubation_features_enhanced_gap4_52to4.csv"
)


def calc_slope(time_bins: np.ndarray, values: np.ndarray, min_obs: int = 3) -> float:
    """對應 R 腳本的 calc_slope()：線性迴歸斜率 ×4 = 每 4h 變化量。

    刻意不用 np.polyfit()／np.linalg.lstsq()：這兩者內部走 SVD，對於
    FiO2 在觀測窗內完全沒變動（常見情形——呼吸器設定沒調整）的病人，
    數學上斜率應精確為 0，但 SVD 會殘留與 BLAS/LAPACK 實作相關的浮點
    誤差（例如 3e-17），導致不同機器上「slope == 0」的判斷結果不一致
    （這正是你我兩次執行結果不同的原因）。改用簡單線性迴歸的封閉解公式
    （covariance / variance），當 y 為常數時 y - mean(y) 對每個元素都精確
    為 0，slope 會精確得到 0.0，在不同平台上可重現。
    """
    valid = ~np.isnan(values)
    if valid.sum() < min_obs:
        return np.nan
    x = time_bins[valid]
    y = values[valid]
    dx = x - x.mean()
    dy = y - y.mean()
    slope = np.sum(dx * dy) / np.sum(dx * dx)
    return slope * 4


def mwu_effect(s: pd.Series, f: pd.Series):
    s_valid, f_valid = s.dropna(), f.dropna()
    U, p = st.mannwhitneyu(s_valid, f_valid, alternative="two-sided")
    rank_biserial = 1 - (2 * U) / (len(s_valid) * len(f_valid))
    return s_valid, f_valid, U, p, rank_biserial


def report_quantiles(name: str, s: pd.Series, f: pd.Series) -> None:
    for label, x in [("Success", s), ("Failure", f)]:
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        print(f"  {name} {label:7s}: median={med:.4f}  (Q1={q1:.4f}, Q3={q3:.4f})  n={len(x)}")


def main() -> None:
    raw_df = pd.read_csv(ENHANCED_CSV)

    # ------------------------------------------------------------
    # Table 4: FiO2 snapshot @ time_bin == -8
    # ------------------------------------------------------------
    print("=" * 70)
    print("Table 4 — FiO2 (snapshot, time_bin == -8)")
    print("=" * 70)

    snap = raw_df.loc[raw_df.time_bin == -8, ["stay_id", "Extubation_failure", "FiO2"]]
    s4 = snap.loc[snap.Extubation_failure == 0, "FiO2"]
    f4 = snap.loc[snap.Extubation_failure == 1, "FiO2"]

    report_quantiles("FiO2", s4.dropna(), f4.dropna())

    s4_valid, f4_valid, U4, p4, rb4 = mwu_effect(s4, f4)

    # 上尾集中：以兩組合併中位數為界，比較「高於合併中位數」的比例；
    # 另外列出常見高 FiO2 門檻（>=0.5）方便對照臨床意義。
    pooled_median4 = pd.concat([s4_valid, f4_valid]).median()
    print(f"\n  合併中位數 (pooled median) = {pooled_median4:.4f}")
    print(f"  FiO2 > 合併中位數 比例 (Success, Failure): "
          f"{(s4_valid > pooled_median4).mean():.4f}, {(f4_valid > pooled_median4).mean():.4f}")
    print(f"  FiO2 >= 0.5      比例 (Success, Failure): "
          f"{(s4_valid >= 0.5).mean():.4f}, {(f4_valid >= 0.5).mean():.4f}")
    print(f"\n  Mann-Whitney U = {U4:.1f}, p = {p4:.3e}")
    print(f"  rank-biserial correlation = {rb4:.4f}")

    # ------------------------------------------------------------
    # Table 5: FiO2 trend per 4h (slope over 48h window)
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("Table 5 — FiO2, trend per 4h (48h window slope)")
    print("=" * 70)

    # 注意：不用 groupby().apply(lambda g: pd.Series({...})) 這種「回傳
    # dict/Series」的寫法 —— 這種寫法在不同 pandas 版本對「是否要把分組欄位
    # 也丟進 apply」的判斷不一致（即 FutureWarning: "DataFrameGroupBy.apply
    # operated on the grouping columns" 所指的行為），會在不同版本得到不同
    # 結果，不可靠。改用兩個各自明確、回傳純量的 groupby 運算，避免歧義。
    label = raw_df.groupby("stay_id")["Extubation_failure"].first()
    slope = raw_df.groupby("stay_id").apply(
        lambda g: calc_slope(
            g["time_bin"].to_numpy(dtype=float),
            g["FiO2"].to_numpy(dtype=float),
        ),
        include_groups=False,
    )
    trend_df = pd.DataFrame({"Extubation_failure": label, "FiO2_trend": slope}).reset_index()

    s5 = trend_df.loc[trend_df.Extubation_failure == 0, "FiO2_trend"]
    f5 = trend_df.loc[trend_df.Extubation_failure == 1, "FiO2_trend"]

    report_quantiles("trend", s5.dropna(), f5.dropna())

    s5_valid, f5_valid, U5, p5, rb5 = mwu_effect(s5, f5)

    print(f"\n  trend == 0 比例 (Success, Failure): {s5_valid.eq(0).mean():.4f}, {f5_valid.eq(0).mean():.4f}")
    print(f"  trend > 0  比例 (Success, Failure): {s5_valid.gt(0).mean():.4f}, {f5_valid.gt(0).mean():.4f}")
    print(f"  trend < 0  比例 (Success, Failure): {s5_valid.lt(0).mean():.4f}, {f5_valid.lt(0).mean():.4f}")
    print(f"\n  Mann-Whitney U = {U5:.1f}, p = {p5:.3e}")
    print(f"  rank-biserial correlation = {rb5:.4f}")

    # ------------------------------------------------------------
    # 合併摘要（供寫註腳用）
    # ------------------------------------------------------------
    print()
    print("=" * 70)
    print("合併摘要")
    print("=" * 70)
    print(f"Table 4 FiO2 (snapshot)   : r = {rb4:.2f}, "
          f"upper-tail(>pooled median) Failure {100*(f4_valid > pooled_median4).mean():.1f}% "
          f"vs Success {100*(s4_valid > pooled_median4).mean():.1f}%, p = {p4:.1e}")
    print(f"Table 5 FiO2 trend/4h     : r = {rb5:.2f}, "
          f"rising-trend Failure {100*f5_valid.gt(0).mean():.1f}% "
          f"vs Success {100*s5_valid.gt(0).mean():.1f}%, p = {p5:.1e}")


if __name__ == "__main__":
    main()
