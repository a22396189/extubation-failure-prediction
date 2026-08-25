# =============================================================
# add_derived_features_gap4_52to4.py
#
# 【目的】
#   對清洗後的特徵表新增三個衍生變數，供模型使用：
#     (1) IBW（Ideal Body Weight，理想體重）：Devine 公式公制版
#     (2) TV_per_kg = Tidal_Volume / IBW（mL/kg）：肺保護通氣評估
#     (3) OI = (MAP × FiO2 × 100) / PaO2：Oxygenation Index 氧合指數
#         ⚠️ 此處 MAP = Mean Airway Pressure（非 Mean Arterial Pressure）
#
# 【CCI 處理說明】
#   Charlson_Score 已於 merge 步驟（merge_extubation_features_gap4_52to4.py）
#   合併至主表，本程式會先檢查是否已存在：
#   - 若已存在 → 跳過合併（避免重複欄位）
#   - 若不存在 → 從 extubation_features_CCI.csv 補充合併（fallback）
#
# 【輸入】
#   - extubation_features_cleaned_gap4_52to4.csv：清洗後主表
#   - extubation_features_CCI.csv（fallback 用）：Charlson Comorbidity Index
#   - extubation_features_bmi.csv（fallback 用）：若主表缺 height 欄位時補入
#
# 【衍生變數計算公式】
#   IBW（男）= 50   + 0.91 × (height_cm − 152.4)，height < 152.4 時取 50
#   IBW（女）= 45.5 + 0.91 × (height_cm − 152.4)，height < 152.4 時取 45.5
#   TV_per_kg = Tidal_Volume / IBW
#   OI        = (MAP × FiO2 × 100) / PaO2，FiO2 須為小數（0.21–1.0）
#
# 【輸出】
#   - extubation_features_enhanced_gap4_52to4.csv
#
# 【執行步驟】
#   Step 1：讀取資料
#   Step 2：CCI 合併（若 Charlson_Score 已存在則跳過）
#   Step 3：height 補入（若主表已有 height 則跳過）
#   Step 4：計算 IBW、TV_per_kg、OI
#   Step 5：輸出 CSV
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
base_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
gap_dir  = base_dir / "gap4_52to4"

input_path  = gap_dir / "extubation_features_cleaned_gap4_52to4.csv"
cci_path    = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_features_CCI.csv")
height_path = base_dir / "extubation_features_bmi.csv"     # fallback：若主表缺 height
output_path = gap_dir / "extubation_features_enhanced_gap4_52to4.csv"
gap_dir.mkdir(parents=True, exist_ok=True)

# =============================================================
# Step 1：讀取資料
# =============================================================
print("=" * 60)
print("Step 1：讀取資料")
print("=" * 60)

df = pd.read_csv(input_path)
print(f"✅ Cleaned 主表：{df.shape[0]:,} 列 × {df.shape[1]} 欄")

# key 欄位轉 Int64（避免 merge 時型別不一致）
for _k in ["stay_id", "subject_id"]:
    if _k in df.columns:
        df[_k] = pd.to_numeric(df[_k], errors="coerce").astype("Int64")

# =============================================================
# Step 2：CCI 合併（Charlson_Score）
# 若已存在（由 merge 步驟帶入）→ 跳過，避免欄位重複
# =============================================================
print("\n" + "=" * 60)
print("Step 2：CCI（Charlson_Score）確認")
print("=" * 60)

if "Charlson_Score" in df.columns:
    n_missing_cci = df["Charlson_Score"].isna().sum()
    print(f"✅ Charlson_Score 已存在於主表（由 merge 步驟帶入），跳過合併")
    print(f"   缺失筆數：{n_missing_cci:,}")
else:
    print("⚠️  Charlson_Score 不在主表，從 CCI 檔補充合併（fallback）...")
    if not cci_path.exists():
        raise FileNotFoundError(f"❌ 找不到 CCI 檔：{cci_path}")

    df_cci = pd.read_csv(cci_path)
    for _k in ["stay_id", "subject_id"]:
        if _k in df_cci.columns:
            df_cci[_k] = pd.to_numeric(df_cci[_k], errors="coerce").astype("Int64")

    if "stay_id" not in df_cci.columns:
        raise ValueError("❌ extubation_features_CCI.csv 缺少 stay_id，無法 merge")
    if "Charlson_Score" not in df_cci.columns:
        raise ValueError("❌ extubation_features_CCI.csv 缺少 Charlson_Score 欄位")

    cci_keys = [k for k in ["subject_id", "stay_id"] if k in df_cci.columns]
    df_cci   = df_cci.drop_duplicates(subset=cci_keys)

    # 決定要合併的 CCI 欄位（Charlson_Score + 可選各共病 flags）
    WANT_CCI_FLAGS = False   # 設為 True 可一併合併 MI/CHF/COPD 等 flags
    if WANT_CCI_FLAGS:
        drop_cols    = set(cci_keys + ["Extubation_failure"])
        flag_cols    = [c for c in df_cci.columns if c not in drop_cols and c != "Charlson_Score"]
        cci_use_cols = cci_keys + ["Charlson_Score"] + flag_cols
    else:
        cci_use_cols = cci_keys + ["Charlson_Score"]

    df = df.merge(df_cci[cci_use_cols], on=cci_keys, how="left")
    print(f"   ✅ CCI fallback 合併完成，主表欄位數：{df.shape[1]}")

# =============================================================
# Step 3：height 補入（若主表已有 height 則跳過）
# =============================================================
print("\n" + "=" * 60)
print("Step 3：height 確認")
print("=" * 60)

if "height" in df.columns:
    print(f"✅ height 已存在於主表，跳過補入")
    print(f"   height 缺失：{df['height'].isna().sum():,} 筆")
else:
    print("⚠️  height 不在主表，從 bmi 檔補入（fallback）...")
    if not height_path.exists():
        print(f"⚠️  找不到 height fallback 檔：{height_path}，height 填 NaN")
        df["height"] = np.nan
    else:
        df_height = pd.read_csv(height_path)
        for _k in ["stay_id", "subject_id"]:
            if _k in df_height.columns:
                df_height[_k] = pd.to_numeric(df_height[_k], errors="coerce").astype("Int64")

        if "height" not in df_height.columns:
            print("⚠️  height 欄位不存在於 bmi 檔，height 填 NaN")
            df["height"] = np.nan
        else:
            h_keys    = [k for k in ["subject_id", "stay_id"] if k in df_height.columns]
            df_height = df_height.drop_duplicates(subset=h_keys)
            df = df.merge(df_height[h_keys + ["height"]], on=h_keys, how="left")
            print(f"   ✅ height fallback 合併完成")

# 保險：計算用欄位統一轉 numeric
for c in ["height", "weight", "Tidal_Volume", "MAP", "FiO2", "PaO2"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# =============================================================
# Step 4：計算衍生變數
# =============================================================
print("\n" + "=" * 60)
print("Step 4：計算衍生變數（IBW、TV_per_kg、OI）")
print("=" * 60)

# --- Helper：性別判斷 ---
def _is_male(x):
    """相容多種性別表示格式：'Male'/'M'/1/'1'/True 等"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(int(x) == 1)
    s = str(x).strip().lower()
    if s in ["male", "m", "man", "1", "true"]:
        return True
    if s in ["female", "f", "woman", "0", "false"]:
        return False
    return np.nan

# --- IBW（Devine 公式公制版）---
def get_ibw(height_cm, sex_val):
    """
    Male:   IBW = 50   + 0.91 × (height_cm − 152.4)
    Female: IBW = 45.5 + 0.91 × (height_cm − 152.4)
    height < 152.4 時，diff 截為 0（Devine 公式既有限制）
    """
    if pd.isna(height_cm):
        return np.nan
    male_flag = _is_male(sex_val)
    if pd.isna(male_flag):
        return np.nan
    base = 50.0 if male_flag else 45.5
    diff = max(float(height_cm) - 152.4, 0.0)
    return base + 0.91 * diff

if "sex" in df.columns and "height" in df.columns:
    df["IBW"] = [get_ibw(h, s) for h, s in zip(df["height"], df["sex"])]
    n_ibw_missing = df["IBW"].isna().sum()
    print(f"✅ IBW 計算完成（缺失：{n_ibw_missing:,} 筆）")
else:
    df["IBW"] = np.nan
    print("⚠️  缺少 sex 或 height，IBW 全部填 NaN")

# --- TV_per_kg（mL/kg IBW）---
if "Tidal_Volume" in df.columns:
    denom = df["IBW"].replace(0, np.nan)
    df["TV_per_kg"] = (df["Tidal_Volume"] / denom).replace([np.inf, -np.inf], np.nan)
    n_tv_missing = df["TV_per_kg"].isna().sum()
    print(f"✅ TV_per_kg 計算完成（缺失：{n_tv_missing:,} 筆）")
else:
    df["TV_per_kg"] = np.nan
    print("⚠️  缺少 Tidal_Volume，TV_per_kg 全部填 NaN")

# --- Oxygenation Index（OI）---
# OI = (Mean Airway Pressure × FiO2 × 100) / PaO2
# FiO2 應為小數（clean 步驟已轉換）；PaO2 單位 mmHg
if all(c in df.columns for c in ["MAP", "FiO2", "PaO2"]):
    pao2_safe  = df["PaO2"].replace(0, np.nan)
    df["OI"]   = (df["MAP"] * df["FiO2"] * 100.0 / pao2_safe).replace([np.inf, -np.inf], np.nan)
    n_oi_missing = df["OI"].isna().sum()
    print(f"✅ OI 計算完成（缺失：{n_oi_missing:,} 筆）")
else:
    df["OI"] = np.nan
    print("⚠️  缺少 MAP / FiO2 / PaO2，OI 全部填 NaN")

# =============================================================
# Step 5：輸出
# =============================================================
print("\n" + "=" * 60)
print("Step 5：輸出 CSV")
print("=" * 60)

new_cols = [c for c in ["Charlson_Score", "IBW", "TV_per_kg", "OI"] if c in df.columns]
print(f"✅ 新增 / 確認欄位：{', '.join(new_cols)}")
print(f"📊 輸出維度：{df.shape}")

df.to_csv(output_path, index=False)
print(f"💾 已儲存至：{output_path}")

# 預覽
show_cols = [c for c in [
    "stay_id", "time_bin", "sex", "height",
    "Tidal_Volume", "IBW", "TV_per_kg",
    "FiO2", "PaO2", "MAP", "OI",
    "Charlson_Score"
] if c in df.columns]
print("\n🔍 Preview（前 10 筆）：")
print(df[show_cols].head(10).to_string(index=False))
