# =============================================================
# reorder_extubation_features_gap4_52to4.py
#
# 【目的】
#   將 extubation_features_merged_gap4_52to4.csv 進行以下整理：
#   (1) 修正欄位衝突：同時存在 "glucose"（vitalsign）與 "Glucose"（lab）
#       → 以 "Glucose" 為準，用 "glucose" 補缺值後刪除 "glucose"
#   (2) 依臨床語意重新排列欄位順序（動態 → 靜態 → 標籤）
#   (3) 任何未列在 column_order 的欄位（如 BMI_source）自動接在最後保留
#
# 【欄位設計說明】
#   - sbp / dbp 保留在資料中，模型訓練時可選擇使用 mbp 取代
#   - height / weight 保留，供後續 add_derived 計算 IBW 使用
#   - Tidal_Volume 保留（TV_per_kg 在 add_derived 步驟計算）
#   - Primary_Diagnosis / ICD_Version 不作為模型特徵，不納入 column_order
#   - Charlson_Score 為靜態特徵之一，已於 merge 步驟合併
#
# 【輸入】
#   - extubation_features_merged_gap4_52to4.csv
#
# 【輸出】
#   - extubation_features_merged_gap4_52to4_reordered.csv
#
# 【執行步驟】
#   Step 1：修正 glucose / Glucose 欄位衝突
#   Step 2：依指定順序排列欄位（未列出欄位接在最後）
#   Step 3：輸出 CSV
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
input_path  = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4\extubation_features_merged_gap4_52to4.csv")
output_path = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4\extubation_features_merged_gap4_52to4_reordered.csv")

print(f"🚀 讀取資料: {input_path} ...")
df = pd.read_csv(input_path)
print(f"✅ 原始資料 shape = {df.shape}")
print(f"   欄位清單: {df.columns.tolist()}")

# =============================================================
# Step 1：欄位衝突修正：glucose（vitalsign）vs Glucose（lab）
# =============================================================
print("\n" + "=" * 50)
print("Step 1：修正 glucose / Glucose 欄位衝突")
print("=" * 50)

has_lower = "glucose" in df.columns
has_upper = "Glucose" in df.columns

if has_lower or has_upper:
    # 確保為 numeric（避免字串混入）
    if has_upper:
        df["Glucose"] = pd.to_numeric(df["Glucose"], errors="coerce")
    if has_lower:
        df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    if has_upper and has_lower:
        # 以 lab 的 Glucose 為主，vitalsign 的 glucose 補缺值
        before_missing = df["Glucose"].isna().sum()
        df["Glucose"] = df["Glucose"].fillna(df["glucose"])
        after_missing  = df["Glucose"].isna().sum()
        filled = int(before_missing - after_missing)
        df.drop(columns=["glucose"], inplace=True)
        print(f"✅ 兩欄皆存在：用 'glucose' 補 'Glucose' 缺值 {filled:,} 筆，已刪除 'glucose'")
    elif has_lower and not has_upper:
        df.rename(columns={"glucose": "Glucose"}, inplace=True)
        print("✅ 僅有 'glucose'，已改名為 'Glucose'")
    else:
        print("✅ 僅有 'Glucose'，無衝突")
else:
    print("ℹ️ 資料中沒有 glucose / Glucose 欄位，略過")

print(f"✅ 處理後 shape = {df.shape}")

# =============================================================
# Step 2：欄位排序
# =============================================================
print("\n" + "=" * 50)
print("Step 2：依臨床語意排列欄位")
print("=" * 50)

column_order = [
    # ── 識別與時間 ────────────────────────────────────────────
    "subject_id", "stay_id", "time_bin",

    # ── 動態特徵（26 個，依模型設計順序）────────────────────────

    # 1. 生理徵象 (Vital Signs)
    "heart_rate", "resp_rate", "spo2",
    "sbp", "dbp", "mbp",          # sbp/dbp 保留；模型可選用 mbp
    "temperature",
    "GCS",

    # 2. 呼吸器參數 (Ventilator)
    "FiO2", "MAP", "PEEP",
    "Tidal_Volume",                # TV_per_kg 由 add_derived 計算後取代
    "MV_day",

    # 3. 動脈血氣 (Blood Gas)
    "pH", "PaO2", "PaCO2", "BE", "PaO2_FiO2_Ratio",

    # 4. 實驗室數據 (Lab)
    "Cr", "WBC", "Hb", "PLT", "AnionGap", "Lactate", "Glucose",

    # 5. 輸入輸出 (I/O)
    "input_ml", "urineoutput", "io_balance",

    # 6. 處置與藥物 (Interventions)
    "Vasopressor_use", "Hemodialysis_use",

    # ── 靜態特徵（4 個模型特徵 + 身高體重供衍生計算用）──────────
    "age", "sex", "BMI", "Charlson_Score",
    "height", "weight",            # 供 add_derived 計算 IBW 使用，非模型直接輸入

    # ── 預測目標 ──────────────────────────────────────────────
    "Extubation_failure",
]

# 未列入 column_order 的欄位（如 BMI_source、diag_clean 等）自動接在最後保留
missing_in_data = [c for c in column_order if c not in df.columns]
present_in_order = [c for c in column_order if c in df.columns]
extra_cols = [c for c in df.columns if c not in column_order]

if missing_in_data:
    print("⚠️ 以下欄位在資料中不存在（略過，不影響輸出）：")
    for m in missing_in_data:
        print(f"   ❌ {m}")
    if "Extubation_failure" in missing_in_data:
        if "Extubation_success" in df.columns:
            print("💡 提示：資料中有 'Extubation_success'，請確認 merge 步驟 label 欄位名稱一致。")

if extra_cols:
    print(f"\nℹ️ 以下欄位不在 column_order，將接在最後保留（共 {len(extra_cols)} 個）：")
    for e in extra_cols:
        print(f"   ➕ {e}")

final_cols = present_in_order + extra_cols
df_reordered = df[final_cols]

print(f"\n✅ 欄位排序完成：共 {len(final_cols)} 欄（指定 {len(present_in_order)} + 額外 {len(extra_cols)}）")

# =============================================================
# Step 3：輸出
# =============================================================
print("\n" + "=" * 50)
print("Step 3：輸出 CSV")
print("=" * 50)

df_reordered.to_csv(output_path, index=False)
print(f"🎉 重新排列完成！已輸出：{output_path}")
print(f"📊 輸出資料維度：{df_reordered.shape}")

print("\n🔍 前 5 列資料預覽：")
print(df_reordered.head())
