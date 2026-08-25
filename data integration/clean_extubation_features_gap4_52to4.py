# =============================================================
# clean_extubation_features_gap4_52to4.py
#
# 【目的】
#   依據《MIMIC-IV 拔管模型前處理之建議合理範圍》
#   對合併後的特徵表進行極端值清洗（outlier → NaN），
#   並在清洗後重算 BMI 與 PaO2_FiO2_Ratio（避免使用已污染的原始值）。
#
# 【輸入】
#   - extubation_features_merged_gap4_52to4_reordered.csv
#
# 【清洗流程】
#   Step 0：inf / -inf → NaN；time_bin 轉 Int64
#   Step 1：FiO2 單位轉換（> 1.0 視為百分比 → 除以 100）
#   Step 2：依合理範圍逐欄清洗（超出範圍 → NaN）
#           ⚠️ BMI 與 PaO2_FiO2_Ratio 跳過，後面步驟重算
#   Step 3：BMI 重算（若超出範圍或缺失，以清洗後 height/weight 重算）
#   Step 4：PaO2_FiO2_Ratio 重算（以清洗後 PaO2 / FiO2 重算）
#   Step 5：輸出 CSV + 關鍵欄位統計摘要
#
# 【注意】
#   - 合理範圍來自《MIMIC-IV 拔管模型前處理之建議合理範圍》文件
#   - 清洗後仍缺失的值，留至 impute 步驟統一填補
#   - 非數值欄位（sex / BMI_source 等）不做範圍清洗
#
# 【輸出】
#   - extubation_features_cleaned_gap4_52to4.csv
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
input_path  = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4\extubation_features_merged_gap4_52to4_reordered.csv")
output_path = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4\extubation_features_cleaned_gap4_52to4.csv")

print(f"🚀 讀取資料: {input_path} ...")
df = pd.read_csv(input_path)
print(f"✅ 原始資料維度: {df.shape}")

# =============================================================
# Step 0：基礎整理
# =============================================================
print("\n" + "=" * 50)
print("Step 0：inf/-inf → NaN；time_bin 轉 Int64")
print("=" * 50)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

if "time_bin" in df.columns:
    df["time_bin"] = pd.to_numeric(df["time_bin"], errors="coerce").astype("Int64")

# =============================================================
# Step 1：FiO2 單位轉換（必須在範圍清洗前執行）
# FiO2 > 1.0 → 視為百分比格式（e.g., 40 → 0.40）
# =============================================================
print("\n" + "=" * 50)
print("Step 1：FiO2 單位轉換")
print("=" * 50)

if "FiO2" in df.columns:
    df["FiO2"] = pd.to_numeric(df["FiO2"], errors="coerce")
    mask_percent = df["FiO2"] > 1.0
    n_convert = int(mask_percent.sum())
    if n_convert > 0:
        df.loc[mask_percent, "FiO2"] = df.loc[mask_percent, "FiO2"] / 100.0
        print(f"✅ 偵測到 {n_convert:,} 筆 FiO2 > 1.0，已轉換為小數格式")
    else:
        print("✅ 所有 FiO2 已為小數格式，無需轉換")

# =============================================================
# Step 2：依合理範圍逐欄清洗（超出範圍 → NaN）
# BMI 與 PaO2_FiO2_Ratio 跳過，於後續步驟重算
# =============================================================
print("\n" + "=" * 50)
print("Step 2：極端值清洗（超出合理範圍 → NaN）")
print("=" * 50)

# 格式：'欄位名': (下限, 上限)
# 來源：《MIMIC-IV 拔管模型前處理之建議合理範圍》
limits = {
    # 生命徵象
    "heart_rate":  (25, 350),
    "resp_rate":   (3, 100),
    "spo2":        (40, 100),
    "sbp":         (0, 375),
    "dbp":         (0, 375),
    "mbp":         (10, 200),
    "temperature": (23, 43),
    "GCS":         (3, 15),

    # 呼吸器參數
    "FiO2":         (0.21, 1.0),
    "PEEP":         (0, 40),
    "MAP":          (0, 60),       # Mean Airway Pressure（非 Mean Arterial Pressure）
    "Tidal_Volume": (100, 1500),
    "MV_day":       (0, 365),

    # 動脈血氣
    "pH":    (6.80, 7.80),
    "PaO2":  (20, 700),
    "PaCO2": (10, 150),
    "BE":    (-30, 30),
    "PaO2_FiO2_Ratio": (30, 700),  # ⚠️ 列在此處但跳過，Step 4 重算

    # 實驗室檢驗
    "Cr":       (0.1, 25.0),
    "WBC":      (0.1, 100.0),
    "Hb":       (3.0, 25.0),
    "PLT":      (5.0, 1500.0),
    "AnionGap": (0, 50),
    "Lactate":  (0.1, 30.0),
    "Glucose":  (10, 2000),

    # 體液與人體測量
    "input_ml":    (0, 30000),
    "urineoutput": (0, 20000),
    "io_balance":  (-20000, 20000),
    "age":         (18, 100),
    "height":      (100, 250),
    "weight":      (20, 300),
    "BMI":         (10, 100),      # ⚠️ 列在此處但跳過，Step 3 重算
}

# 跳過範圍清洗的欄位（僅需列出「有定義在 limits 字典內、但不希望此處清洗」的欄位）
# ⚠️ 不在 limits 字典內的欄位（sex、BMI_source、CCI flags、ID 欄位等）
#    根本不會進入 for 迴圈，不需要加入此 set
skip_cols = {
    "BMI",              # 有定義在 limits，但跳過 → Step 3 以 height/weight 重算
    "PaO2_FiO2_Ratio",  # 有定義在 limits，但跳過 → Step 4 以 PaO2/FiO2 重算
}

total_cleaned = 0
for col, (min_val, max_val) in limits.items():
    if col not in df.columns or col in skip_cols:
        continue
    df[col] = pd.to_numeric(df[col], errors="coerce")
    outliers = (df[col] < min_val) | (df[col] > max_val)
    count = int(outliers.sum())
    if count > 0:
        df.loc[outliers, col] = np.nan
        total_cleaned += count
        print(f"   {col:20s}  清除 {count:>6,} 筆  （合理範圍 {min_val} ~ {max_val}）")

print(f"\n✅ Step 2 完成：共清除 {total_cleaned:,} 筆異常值")

# =============================================================
# Step 3：BMI 重算
# 若 BMI 超出範圍或缺失，以清洗後的 height / weight 重算
# =============================================================
print("\n" + "=" * 50)
print("Step 3：BMI 重算（使用清洗後 height / weight）")
print("=" * 50)

if all(c in df.columns for c in ["BMI", "weight", "height"]):
    min_bmi, max_bmi = limits["BMI"]
    for c in ["BMI", "weight", "height"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_bad_bmi  = (df["BMI"] < min_bmi) | (df["BMI"] > max_bmi) | df["BMI"].isna()
    mask_recalc   = mask_bad_bmi & df["weight"].notna() & df["height"].notna() & (df["height"] > 0)

    if int(mask_recalc.sum()) > 0:
        df.loc[mask_recalc, "BMI"] = (
            df.loc[mask_recalc, "weight"]
            / ((df.loc[mask_recalc, "height"] / 100.0) ** 2)
        )
        print(f"✅ 重算 BMI：{int(mask_recalc.sum()):,} 筆")

    # 重算後仍超出範圍 → NaN
    still_bad = (df["BMI"] < min_bmi) | (df["BMI"] > max_bmi)
    if int(still_bad.sum()) > 0:
        df.loc[still_bad, "BMI"] = np.nan
        print(f"⚠️  重算後仍有 {int(still_bad.sum()):,} 筆 BMI 無法修復 → NaN")

    print(f"   BMI 最終缺失：{df['BMI'].isna().sum():,} 筆")
else:
    print("⚠️ 缺少 BMI / height / weight 欄位，跳過 BMI 重算")

# =============================================================
# Step 4：PaO2_FiO2_Ratio 重算
# 若 Ratio 超出範圍、缺失或為 inf，以清洗後 PaO2 / FiO2 重算
# =============================================================
print("\n" + "=" * 50)
print("Step 4：PaO2_FiO2_Ratio 重算（使用清洗後 PaO2 / FiO2）")
print("=" * 50)

if all(c in df.columns for c in ["PaO2_FiO2_Ratio", "PaO2", "FiO2"]):
    min_pf, max_pf = limits["PaO2_FiO2_Ratio"]
    for c in ["PaO2_FiO2_Ratio", "PaO2", "FiO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_bad_pf = (
        df["PaO2_FiO2_Ratio"].isna()
        | (df["PaO2_FiO2_Ratio"] < min_pf)
        | (df["PaO2_FiO2_Ratio"] > max_pf)
        | np.isinf(df["PaO2_FiO2_Ratio"])
    )
    mask_recalc = mask_bad_pf & df["PaO2"].notna() & df["FiO2"].notna() & (df["FiO2"] > 0)

    if int(mask_recalc.sum()) > 0:
        df.loc[mask_recalc, "PaO2_FiO2_Ratio"] = (
            df.loc[mask_recalc, "PaO2"] / df.loc[mask_recalc, "FiO2"]
        )
        print(f"✅ 重算 PaO2_FiO2_Ratio：{int(mask_recalc.sum()):,} 筆")

    # 重算後再次清除 inf 與超出範圍
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    still_bad_pf = (
        (df["PaO2_FiO2_Ratio"] < min_pf)
        | (df["PaO2_FiO2_Ratio"] > max_pf)
    )
    if int(still_bad_pf.sum()) > 0:
        df.loc[still_bad_pf, "PaO2_FiO2_Ratio"] = np.nan
        print(f"⚠️  重算後仍有 {int(still_bad_pf.sum()):,} 筆 Ratio 無法修復 → NaN")

    print(f"   PaO2_FiO2_Ratio 最終缺失：{df['PaO2_FiO2_Ratio'].isna().sum():,} 筆")
else:
    print("⚠️ 缺少 PaO2_FiO2_Ratio / PaO2 / FiO2 欄位，跳過 Ratio 重算")

# =============================================================
# Step 5：輸出
# =============================================================
print("\n" + "=" * 50)
print("Step 5：輸出 CSV")
print("=" * 50)

df.to_csv(output_path, index=False)
print(f"🎉 資料清洗完成！已輸出至: {output_path}")
print(f"📊 最終資料維度: {df.shape}")

# 關鍵欄位清洗後統計摘要
check_cols = [c for c in ["FiO2", "BMI", "PaO2_FiO2_Ratio", "MAP", "Glucose", "PEEP", "pH"] if c in df.columns]
if check_cols:
    print("\n🔍 關鍵欄位統計摘要（清洗後）：")
    summary = df[check_cols].describe().loc[["count", "min", "max"]]
    print(summary.to_string())
