# =============================================================
# build_extubation_features_bmi.py
#
# 【目的】
#   計算每位 ICU 病人的 BMI，作為靜態特徵之一。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / subject_id
#   - first_day_height.parquet (MIMIC-IV derived)：ICU 第一天身高（cm）
#   - first_day_weight.parquet (MIMIC-IV derived)：ICU 入院體重 weight_admit（kg）
#   - omr.csv (MIMIC-IV hosp)：門診記錄，含 result_name / result_value，
#     當 result_name == "BMI (kg/m2)" 時可作為備援 BMI 來源
#
# 【BMI 來源優先順序】
#   (1) 主要來源：first_day_height + first_day_weight → BMI = weight / height(m)²
#       身高篩選：50–250 cm；體重篩選：> 0 kg
#   (2) 備援來源：omr.csv（result_name == "BMI (kg/m2)"）
#       僅在 height 或 weight 缺失、無法計算時才啟用，
#       取距拔管日期最近的 chartdate 對應紀錄補值
#
# 【注意】
#   極端值清洗（BMI 合理範圍）留至
#   clean_extubation_features_gap4_52to4.py 統一處理。
#
# 【輸出】
#   - extubation_features_bmi.csv：
#       含 height、weight、BMI、BMI_source（calculated/omr/missing）、
#       subject_id、stay_id、Extubation_failure
#
# 【執行步驟】
#   Step 0：若既有 CSV 存在，報告各欄缺失值狀況
#   Step 1：讀取身高、體重、拔管名單
#   Step 2：合併身高體重，計算主要 BMI
#   Step 3：從 omr.csv 補值（僅針對 BMI 仍為 NaN 的 subject_id）
#   Step 4：缺失值統計摘要
#   Step 5：輸出 CSV
# =============================================================

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )
MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR")
if not MIMIC_DATA_DIR:
    raise RuntimeError(
        "Environment variable MIMIC_DATA_DIR is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local MIMIC-IV raw-data / DuckDB project root."
    )


# === 路徑設定 ===
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
duckdb_path   = rf"{MIMIC_DATA_DIR}\mimic.duckdb"
height_path   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\first_day_height.parquet"
weight_path   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\first_day_weight.parquet"
omr_path      = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\omr.csv"
extub_path    = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"

output_dir  = Path(rf"{EXTUBATION_ROOT}\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_bmi.csv"

# =============================================================
# 0️⃣ 若已存在舊版 extubation_features_bmi.csv，先報告缺失值狀況
# =============================================================
if output_path.exists():
    print("=" * 60)
    print("📋 [既有檔案缺失值報告] extubation_features_bmi.csv")
    print("=" * 60)
    existing = pd.read_csv(output_path)
    print(f"   總筆數: {len(existing):,}")
    for col in existing.columns:
        n_missing = existing[col].isna().sum()
        pct = n_missing / len(existing) * 100
        print(f"   {col:30s}  缺失 {n_missing:>5,} 筆 ({pct:.1f}%)")
    print("=" * 60 + "\n")
else:
    print("ℹ️  尚無既有 extubation_features_bmi.csv，跳過缺失值報告\n")

# =============================================================
# 1️⃣ 讀取資料
# =============================================================
con = duckdb.connect(duckdb_path)
print("🚀 讀取 first_day_height、first_day_weight、extubation_outcome ...")

# 篩選合理身高範圍 (50–250 cm)
height_df = con.execute(f"""
    SELECT subject_id, stay_id, height
    FROM read_parquet('{height_path}')
    WHERE height BETWEEN 50 AND 250
""").df()

# 篩選合理體重 (> 0 kg)
weight_df = con.execute(f"""
    SELECT subject_id, stay_id, weight_admit AS weight
    FROM read_parquet('{weight_path}')
    WHERE weight_admit > 0
""").df()

extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])

print(f"✅ 身高資料: {len(height_df):,} 筆")
print(f"✅ 體重資料: {len(weight_df):,} 筆")
print(f"✅ 拔管名單: {len(extub):,} 位病人")

# =============================================================
# 2️⃣ 合併身高體重，計算 BMI（主要來源）
# =============================================================
print("\n🧮 合併身高體重並計算 BMI ...")

bmi_df = (
    extub[["subject_id", "stay_id", "extubation_time", "Extubation_failure"]]
    .merge(height_df, on=["subject_id", "stay_id"], how="left")
    .merge(weight_df, on=["subject_id", "stay_id"], how="left")
)

bmi_df["height_m"] = bmi_df["height"] / 100
bmi_df["BMI"] = np.where(
    bmi_df["height_m"].notna() & bmi_df["weight"].notna(),
    bmi_df["weight"] / (bmi_df["height_m"] ** 2),
    np.nan
)

n_missing_after_primary = bmi_df["BMI"].isna().sum()
print(f"📉 主要來源計算後 BMI 缺失: {n_missing_after_primary:,} 筆")

# =============================================================
# 3️⃣ 備援來源：OMR BMI（僅補 BMI 仍為 NaN 的 subject_id）
# =============================================================
if n_missing_after_primary > 0:
    print(f"\n🔄 從 omr.csv 抓取備援 BMI（補 {n_missing_after_primary:,} 筆缺失）...")

    # 讀取 OMR，只取 BMI (kg/m2) 紀錄
    omr_raw = con.execute(f"""
        SELECT subject_id,
               CAST(chartdate AS DATE) AS chartdate,
               CAST(result_value AS DOUBLE) AS bmi_omr
        FROM read_csv_auto('{omr_path}', SAMPLE_SIZE=-1)
        WHERE result_name = 'BMI (kg/m2)'
          AND TRY_CAST(result_value AS DOUBLE) IS NOT NULL
    """).df()

    print(f"✅ OMR BMI 紀錄: {len(omr_raw):,} 筆（共 {omr_raw['subject_id'].nunique():,} 位病人）")

    # 只針對「BMI 缺失的 subject_id」進行補值，減少運算量
    missing_subjects = bmi_df.loc[bmi_df["BMI"].isna(), ["subject_id", "extubation_time"]].copy()
    omr_candidates = omr_raw[omr_raw["subject_id"].isin(missing_subjects["subject_id"])].copy()

    if len(omr_candidates) > 0:
        # 合併拔管時間，計算每筆 OMR 紀錄與拔管時間的日期差（取絕對值）
        omr_candidates = omr_candidates.merge(
            missing_subjects.rename(columns={"extubation_time": "extubation_date"}),
            on="subject_id",
            how="left"
        )
        omr_candidates["extubation_date"] = pd.to_datetime(
            omr_candidates["extubation_date"]
        ).dt.date
        omr_candidates["chartdate"] = pd.to_datetime(omr_candidates["chartdate"]).dt.date
        omr_candidates["date_diff"] = (
            pd.to_datetime(omr_candidates["chartdate"])
            - pd.to_datetime(omr_candidates["extubation_date"])
        ).abs()

        # 每位 subject_id 取距離拔管時間最近的 OMR BMI 紀錄
        omr_best = (
            omr_candidates
            .sort_values("date_diff")
            .groupby("subject_id", as_index=False)
            .first()[["subject_id", "bmi_omr", "chartdate", "date_diff"]]
        )

        print(f"✅ OMR 可補值的 subject_id: {len(omr_best):,} 位")
        print(f"   平均距拔管日差距: {omr_best['date_diff'].dt.days.mean():.1f} 天")

        # 將 OMR BMI 補入主表（僅填 NaN）
        bmi_df = bmi_df.merge(omr_best[["subject_id", "bmi_omr"]], on="subject_id", how="left")
        bmi_df["BMI_source"] = np.where(
            bmi_df["BMI"].notna(), "calculated",
            np.where(bmi_df["bmi_omr"].notna(), "omr", "missing")
        )
        bmi_df["BMI"] = bmi_df["BMI"].fillna(bmi_df["bmi_omr"])
        bmi_df = bmi_df.drop(columns=["bmi_omr"])
    else:
        print("⚠️  OMR 中無符合條件的 BMI 備援紀錄")
        bmi_df["BMI_source"] = np.where(bmi_df["BMI"].notna(), "calculated", "missing")
else:
    bmi_df["BMI_source"] = "calculated"

# =============================================================
# 4️⃣ 缺失值統計摘要
# =============================================================
n_calculated = (bmi_df["BMI_source"] == "calculated").sum()
n_omr        = (bmi_df["BMI_source"] == "omr").sum()       if "BMI_source" in bmi_df.columns else 0
n_still_missing = bmi_df["BMI"].isna().sum()

print("\n" + "=" * 50)
print("📊 BMI 來源統計：")
print(f"   由 height/weight 計算: {n_calculated:,} 筆")
print(f"   由 OMR 補值:           {n_omr:,} 筆")
print(f"   仍缺失 (NaN):          {n_still_missing:,} 筆")
print("=" * 50)

# =============================================================
# 5️⃣ 輸出結果
# =============================================================
output_cols = ["subject_id", "stay_id", "height", "weight", "BMI", "BMI_source", "Extubation_failure"]
bmi_df = bmi_df.drop(columns=["height_m", "extubation_time"], errors="ignore")
bmi_df[output_cols].to_csv(output_path, index=False)

print(f"\n✅ BMI feature 已建立，共 {len(bmi_df):,} 筆")
print(f"📁 儲存位置: {output_path}")

print("\n🔍 前 10 筆資料預覽：")
print(bmi_df[output_cols].head(10))

print("\n📊 BMI 統計摘要：")
print(bmi_df["BMI"].describe().round(2))
