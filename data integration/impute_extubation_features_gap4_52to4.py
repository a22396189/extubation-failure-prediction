# =============================================================
# impute_extubation_features_gap4_52to4.py
#
# 【目的】
#   在資料切分（train / val / test）之後執行缺失值填補，
#   確保 population mean 僅從 train set 計算，
#   避免 val / test 的統計資訊洩漏至填補過程（imputation leakage）。
#
# 【輸入】
#   - extubation_features_enhanced_gap4_52to4.csv
#     （add_derived_features_gap4_52to4.py 的輸出）
#
# 【填補策略】
#   ┌─────────────────────────────────┬──────────────────────────────────┐
#   │ 欄位類型                        │ 填補方式                          │
#   ├─────────────────────────────────┼──────────────────────────────────┤
#   │ 動態連續特徵                    │ LOCF（within stay）→ train mean  │
#   │ Vasopressor_use / Hemodialysis  │ 填 0（缺失 = 未使用）             │
#   │ io_balance                      │ 填 0（缺失 = 無記錄，平衡為 0）  │
#   │ 靜態連續（age / BMI / CCI Score）│ ffill+bfill（within stay）→ train mean│
#   │ 靜態二元（sex）                 │ 填 train mode                     │
#   │ CCI 成分旗標（MI / CHF / ...）  │ 填 0（缺失假設無此共病）          │
#   └─────────────────────────────────┴──────────────────────────────────┘
#
#   ⚠️ LOCF 邏輯：
#     - 依 stay_id 分組，time_bin 由小到大（-52 → -8）排序
#     - 各 bin 的值由上一個有值的 bin 向後傳遞
#     - time_bin=-52（tp0）無前值 → LOCF 無效 → train mean 填補
#
# 【輸出】
#   - extubation_features_imputed_gap4_52to4.csv
#     含 'split' 欄位（train / val / test），讓模型訓練腳本可直接使用
#
# 【執行步驟】
#   Step 1：讀取資料，stratified split by stay_id（70/15/15, seed=42）
#   Step 2：從 train set 計算 population mean / mode
#   Step 3：依填補策略逐類欄位填補（train / val / test 各自套用 train 的統計值）
#   Step 4：驗證模型特徵無殘餘 NaN
#   Step 5：合併輸出（附加 'split' 欄位）
#
# 【注意】
#   - seed 與 split ratio 必須與 transformer_pre_extubation_risk_trajectory.py 一致
#   - StandardScaler 不在此處執行，留在模型訓練腳本中 fit（train only）
# =============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


# =============================================================
# 路徑設定（與 add_derived_features_gap4_52to4.py 慣例一致）
# =============================================================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
base_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs")
gap_dir  = base_dir / "gap4_52to4"
gap_dir.mkdir(parents=True, exist_ok=True)

input_path  = gap_dir / "extubation_features_enhanced_gap4_52to4.csv"
output_path = gap_dir / "extubation_features_imputed_gap4_52to4.csv"

# =============================================================
# 特徵定義（與 transformer_pre_extubation_risk_trajectory.py 保持一致）
# =============================================================
TARGET = "Extubation_failure"

# 26 個動態特徵（模型輸入）
DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]

# 4 個靜態特徵（模型輸入）
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]

# 時間序列 bins（與模型一致）
SEQ_TIME_BINS = list(range(-52, -4, 4))  # [-52, -48, ..., -8]

# 動態：填 0（缺失即無使用/無紀錄，有真實臨床意義）
ZERO_FILL_DYNAMIC = ["Vasopressor_use", "Hemodialysis_use", "io_balance"]

# 動態：LOCF → train mean（連續生理量測，缺失代表未量測）
LOCF_DYNAMIC = [c for c in DYNAMIC_COLS if c not in ZERO_FILL_DYNAMIC]

# CCI 成分旗標（填 0：缺失假設無此共病）
CCI_FLAGS = [
    "MI", "CHF", "PVD", "CVD", "Dementia", "COPD", "Rheumatic",
    "Peptic_Ulcer", "Mild_Liver", "Diabetes_Simple", "Diabetes_Complex",
    "Paraplegia", "Renal_Disease", "Cancer", "Severe_Liver",
    "Metastatic_Solid_Tumor", "AIDS"
]

# 靜態連續（ffill+bfill within stay → train mean）
STATIC_CONTINUOUS = ["age", "BMI", "Charlson_Score"]

# 靜態二元（填 train mode）
STATIC_BINARY = ["sex"]

# =============================================================
# 輔助函數
# =============================================================
def split_by_stay_id(df, train_ratio=0.7, seed=42):
    """
    Stratified split by stay_id（與 transformer 腳本邏輯完全一致）。
    ⚠️ seed 與 ratio 必須與模型訓練腳本相同，確保 split 結果一致。
    """
    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    y = stay_labels[TARGET].values

    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=y
    )
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels[TARGET].values
    )
    return train_ids, val_ids, test_ids


def apply_locf_then_mean(df, cols, pop_mean):
    """
    1. 依 (stay_id, time_bin) 排序（確保 -52 → -48 → ... → -8）
    2. 在每個 stay_id 群組內做 LOCF（ffill）
    3. 仍缺失的（包含 tp0）用 train pop_mean 填補
    """
    df = df.sort_values(["stay_id", "time_bin"])
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("stay_id")[col].transform(lambda x: x.ffill())
        df[col] = df[col].fillna(pop_mean.get(col, 0.0))
    return df


def apply_static_impute(df, static_continuous, static_binary, pop_mean, sex_mode):
    """
    靜態特徵：同一 stay 內 ffill + bfill（傳遞已有的值到其他 bins），
    再用 train 統計值填補完全缺失的病人。
    """
    df = df.sort_values(["stay_id", "time_bin"])

    for col in static_continuous:
        if col not in df.columns:
            continue
        df[col] = df.groupby("stay_id")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(pop_mean.get(col, 0.0))

    for col in static_binary:
        if col not in df.columns:
            continue
        df[col] = df.groupby("stay_id")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(sex_mode)

    return df


# =============================================================
# Step 1：讀取資料 + Stratified Split
# =============================================================
print("=" * 60)
print("Step 1：讀取資料與資料切分")
print("=" * 60)

df = pd.read_csv(input_path)
print(f"✅ 讀取完成：{len(df):,} 列，{df['stay_id'].nunique():,} 位病人")
print(f"   欄位數：{len(df.columns)}")
print(f"   time_bin 範圍：{sorted(df['time_bin'].unique())}")

# sex 欄位字串轉數值（與 transformer 腳本一致）
if "sex" in df.columns and df["sex"].dtype == object:
    df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    print(f"   sex 欄位：字串 → 0/1（Male=1, Female=0）")

# 確認 time_bin 篩選（只保留模型用到的 12 bins）
df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
print(f"   篩選 SEQ_TIME_BINS 後：{len(df):,} 列，{df['stay_id'].nunique():,} 位病人")

# ------------------------------------------------------------------
# ⚠️ Missingness Mask 計算（必須在任何填補之前執行）
# ------------------------------------------------------------------
# mask_{col} = 1 表示該特徵在此 bin「有原始量測值」（not NaN）
#            = 0 表示該特徵在此 bin「原本缺失」（將被填補）
# bin_has_data = 1 表示該 time_bin 至少有一個 dynamic feature 有量測值
#              = 0 表示整個 bin 完全無資料（全部為填補值）
# 這些欄位將隨 imputed CSV 一起儲存，供 Transformer 直接讀取，
# 確保模型能夠辨識哪些位置是「真實量測」vs「填補估計」。
# ------------------------------------------------------------------
print("\n   計算 missingness mask（填補前標記原始缺失位置）...")
mask_cols_added = []
for col in DYNAMIC_COLS:
    mask_col = f"mask_{col}"
    if col in df.columns:
        df[mask_col] = (~df[col].isna()).astype(np.int8)  # 1=有量測, 0=缺失
    else:
        df[mask_col] = np.int8(0)                          # 欄位不存在 → 視為全缺失
    mask_cols_added.append(mask_col)

present_mask_cols = [f"mask_{c}" for c in DYNAMIC_COLS if c in df.columns]
df["bin_has_data"] = (df[present_mask_cols].sum(axis=1) > 0).astype(np.int8)

print(f"   ✅ mask 欄位建立完成：{len(mask_cols_added)} 個 dynamic feature mask + bin_has_data")
print(f"   全 bin 缺失的列數（bin_has_data=0）：{(df['bin_has_data']==0).sum():,}")

train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.7, seed=42)
train_df = df[df["stay_id"].isin(train_ids)].copy()
val_df   = df[df["stay_id"].isin(val_ids)].copy()
test_df  = df[df["stay_id"].isin(test_ids)].copy()

print(f"\n   Train: {train_df['stay_id'].nunique():,} stays "
      f"(failure rate = {train_df.groupby('stay_id')[TARGET].first().mean():.3f})")
print(f"   Val  : {val_df['stay_id'].nunique():,} stays "
      f"(failure rate = {val_df.groupby('stay_id')[TARGET].first().mean():.3f})")
print(f"   Test : {test_df['stay_id'].nunique():,} stays "
      f"(failure rate = {test_df.groupby('stay_id')[TARGET].first().mean():.3f})")

# =============================================================
# Step 2：從 Train Set 計算 Population Mean / Mode
# =============================================================
print("\n" + "=" * 60)
print("Step 2：從 Train Set 計算填補統計值")
print("=" * 60)

# 動態連續：train mean
locf_cols_present = [c for c in LOCF_DYNAMIC if c in train_df.columns]
pop_mean_dynamic  = train_df[locf_cols_present].mean().to_dict()

# 靜態連續：每個 stay 取第一個有值的 row，再計算 train mean
train_stay_static = train_df.groupby("stay_id")[STATIC_CONTINUOUS].first()
pop_mean_static = {
    col: float(train_stay_static[col].mean())
    for col in STATIC_CONTINUOUS
    if col in train_stay_static.columns
}

# 靜態二元（sex）：train mode
sex_mode = int(train_df["sex"].mode()[0]) if "sex" in train_df.columns else 0

print(f"✅ 動態連續特徵 population mean（{len(pop_mean_dynamic)} 欄）：")
for col, val in pop_mean_dynamic.items():
    print(f"   {col:25s}: {val:.4f}")

print(f"\n✅ 靜態連續特徵 population mean：")
for col, val in pop_mean_static.items():
    print(f"   {col:25s}: {val:.4f}")

print(f"\n✅ sex mode（train）: {sex_mode}")

# 合併所有 mean 供後續使用
all_pop_mean = {**pop_mean_dynamic, **pop_mean_static}

# =============================================================
# Step 3：填補（train / val / test 各自套用 train 統計值）
# =============================================================
print("\n" + "=" * 60)
print("Step 3：填補缺失值")
print("=" * 60)


def impute_split(split_df, split_name):
    print(f"\n  [{split_name}] 填補中...")
    df_imp = split_df.copy()

    # ── (A) 動態連續：LOCF → train mean ──────────────────────────────
    before = sum(df_imp[c].isna().sum() for c in locf_cols_present if c in df_imp.columns)
    df_imp = apply_locf_then_mean(df_imp, locf_cols_present, all_pop_mean)
    after  = sum(df_imp[c].isna().sum() for c in locf_cols_present if c in df_imp.columns)
    print(f"    動態連續 LOCF + train mean：NaN {before:,} → {after:,}")

    # ── (B) 動態二元 / I/O：填 0 ─────────────────────────────────────
    for col in ZERO_FILL_DYNAMIC:
        if col in df_imp.columns:
            n = df_imp[col].isna().sum()
            df_imp[col] = df_imp[col].fillna(0)
            if n > 0:
                print(f"    {col}: {n:,} NaN → 0")

    # ── (C) 靜態連續 + 靜態二元：ffill+bfill → train mean/mode ──────
    df_imp = apply_static_impute(df_imp, STATIC_CONTINUOUS, STATIC_BINARY,
                                 all_pop_mean, sex_mode)
    print(f"    靜態特徵（age / BMI / Charlson_Score / sex）：ffill+bfill+train_stat 完成")

    # ── (D) CCI 旗標：填 0 ───────────────────────────────────────────
    cci_present = [c for c in CCI_FLAGS if c in df_imp.columns]
    for col in cci_present:
        df_imp[col] = df_imp[col].fillna(0).astype(int)
    if cci_present:
        print(f"    CCI flags（{len(cci_present)} 欄）：NaN → 0")

    # ── 加入 split 標記 ───────────────────────────────────────────────
    df_imp["split"] = split_name
    return df_imp


train_imp = impute_split(train_df, "train")
val_imp   = impute_split(val_df,   "val")
test_imp  = impute_split(test_df,  "test")

# =============================================================
# Step 4：驗證模型特徵無殘餘 NaN
# =============================================================
print("\n" + "=" * 60)
print("Step 4：驗證模型特徵殘餘缺失值")
print("=" * 60)

df_all = pd.concat([train_imp, val_imp, test_imp], ignore_index=True)
all_model_cols = DYNAMIC_COLS + STATIC_COLS

has_nan = False
for col in all_model_cols:
    if col not in df_all.columns:
        print(f"  ⚠️  欄位不存在於 CSV：{col}")
        continue
    n_nan = df_all[col].isna().sum()
    if n_nan > 0:
        print(f"  ⚠️  {col:25s}: 仍有 {n_nan:,} NaN（請檢查）")
        has_nan = True

if not has_nan:
    print("  ✅ 所有模型特徵（26 動態 + 4 靜態）均無殘餘 NaN")

# 非模型欄位的缺失情況（僅報告，不強制填補）
non_model_cols = [
    c for c in df_all.columns
    if c not in all_model_cols + ["stay_id", "subject_id", "time_bin", TARGET, "split"]
]
nan_report = {c: df_all[c].isna().sum() for c in non_model_cols if df_all[c].isna().sum() > 0}
if nan_report:
    print(f"\n  非模型欄位缺失摘要（{len(nan_report)} 欄有 NaN，保留不填）：")
    for col, n in nan_report.items():
        print(f"    {col:30s}: {n:,} NaN")

# =============================================================
# Step 5：輸出 CSV
# =============================================================
print("\n" + "=" * 60)
print("Step 5：輸出")
print("=" * 60)

df_all["time_bin"] = df_all["time_bin"].astype(int)
df_all.to_csv(output_path, index=False)

print(f"✅ 輸出完成：{len(df_all):,} 列，{df_all['stay_id'].nunique():,} 位病人")
print(f"   Train : {(df_all['split']=='train').sum():,} 列")
print(f"   Val   : {(df_all['split']=='val').sum():,} 列")
print(f"   Test  : {(df_all['split']=='test').sum():,} 列")
print(f"📁 儲存位置：{output_path}")

print("\n📊 各 split 的 Extubation_failure 比率（應接近原始 40.6%）：")
for sp in ["train", "val", "test"]:
    sub = df_all[df_all["split"] == sp]
    rate = sub.groupby("stay_id")[TARGET].first().mean()
    n    = sub["stay_id"].nunique()
    print(f"   {sp:5s}: {rate:.3f}  ({n:,} stays)")
