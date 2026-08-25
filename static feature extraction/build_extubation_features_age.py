# =============================================================
# build_extubation_features_age.py
#
# 【目的】
#   計算每位 ICU 病人在拔管事件當下的年齡，作為靜態特徵之一。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / subject_id
#   - icustays.parquet (MIMIC-IV icu)：stay_id → hadm_id 精確對應橋樑
#   - admissions.csv (MIMIC-IV hosp)：hadm_id → admittime（該次住院入院時間）
#   - patients.csv (MIMIC-IV hosp)：subject_id → anchor_age、anchor_year
#     （MIMIC-IV 以 anchor 系統遮蔽實際年份）
#
# 【年齡計算公式】
#   age = anchor_age + (admittime − anchor_year起點) / 一年秒數（31,556,908.8 秒）
#   依照 MIMIC-IV 官方建議算法。
#
# 【對應鏈】
#   stay_id → hadm_id（icustays，精確 1對1）
#           → admittime（admissions）
#           → age（patients anchor 換算）
#   不使用「subject_id 合併後找時間最近入院」的間接做法，
#   避免跨住院模糊配對的潛在錯誤。
#
# 【輸出】
#   - extubation_features_age.csv：含 age、subject_id、stay_id、Extubation_failure
#
# 【執行步驟】
#   Step 1：讀取 patients、admissions、icustays
#   Step 2：讀取拔管名單（extubation_outcome）
#   Step 3：stay_id → hadm_id → admittime（精確查表）
#   Step 4：計算年齡
#   Step 5：驗證無重複、輸出 CSV
# =============================================================
import duckdb
import pandas as pd
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
duckdb_path     = rf"{MIMIC_DATA_DIR}\mimic.duckdb"
patients_path   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\patients.csv"
admissions_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\admissions.csv"
icustays_path   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\icu\icustays.parquet"
extub_path      = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"

output_dir  = Path(rf"{EXTUBATION_ROOT}\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_age.csv"

# === 1️⃣ 連線與讀取資料 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 patients、admissions、icustays ...")

# 使用 DuckDB 快速讀取
patients   = con.execute(f"SELECT subject_id, anchor_age, anchor_year FROM read_csv_auto('{patients_path}', SAMPLE_SIZE=-1)").df()
admissions = con.execute(f"SELECT hadm_id, admittime FROM read_csv_auto('{admissions_path}', SAMPLE_SIZE=-1)").df()
icu_stays  = con.execute(f"SELECT stay_id, hadm_id FROM read_parquet('{icustays_path}')").df()

print(f"✅ patients: {len(patients):,} 筆")
print(f"✅ admissions: {len(admissions):,} 筆")
print(f"✅ icustays: {len(icu_stays):,} 筆")

# === 2️⃣ 讀取拔管名單 ===
print("\n📂 讀取 extubation_outcome ...")
extub = pd.read_csv(extub_path)
print(f"✅ 拔管名單: {len(extub):,} 筆")

# === 3️⃣ 精確對應：stay_id → hadm_id → admittime ===
print("\n🔗 stay_id → hadm_id（icustays 精確查表）...")

# Step A: stay_id → hadm_id（1對1，由 MIMIC-IV 資料結構保證）
extub = extub.merge(icu_stays, on="stay_id", how="left")

missing_hadm = extub["hadm_id"].isna().sum()
if missing_hadm > 0:
    print(f"⚠️  警告：{missing_hadm} 筆 stay_id 在 icustays 中無對應 hadm_id，請確認資料完整性")

# Step B: hadm_id → admittime（取該次住院的入院時間）
admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
extub = extub.merge(admissions, on="hadm_id", how="left")

missing_admittime = extub["admittime"].isna().sum()
if missing_admittime > 0:
    print(f"⚠️  警告：{missing_admittime} 筆 hadm_id 在 admissions 中無對應 admittime")

# Step C: subject_id → anchor_age / anchor_year
extub = extub.merge(patients, on="subject_id", how="left")

# === 4️⃣ 計算年齡 ===
print("\n🧮 計算拔管當次住院的入院年齡 ...")

# 依照 MIMIC-IV 官方算法：
#   age = anchor_age + (admittime - anchor_year 起點) / 一年秒數
# 一年平均秒數（熱帶年）= 31,556,908.8 秒
SECONDS_PER_YEAR = 31_556_908.8

extub["anchor_year_start"] = pd.to_datetime(
    extub["anchor_year"].astype(str) + "-01-01"
)
extub["age"] = extub["anchor_age"] + (
    (extub["admittime"] - extub["anchor_year_start"]).dt.total_seconds()
    / SECONDS_PER_YEAR
)

# === 5️⃣ 驗證：每個 stay_id 應只有 1 筆（確認無膨脹）===
if extub["stay_id"].duplicated().any():
    print("⚠️  警告：stay_id 出現重複，執行 drop_duplicates 保留第一筆")
    extub = extub.drop_duplicates(subset=["stay_id"], keep="first")

# === 6️⃣ 輸出結果 ===
output_cols = ["subject_id", "stay_id", "age", "Extubation_failure"]

extub[output_cols].to_csv(output_path, index=False)

print(f"\n✅ Age feature generated successfully: {len(extub):,} rows")
print(f"📁 Saved to: {output_path}")

print("\n🔍 前 10 筆資料預覽：")
print(extub[output_cols].head(10))

print("\n📊 年齡統計摘要：")
print(extub["age"].describe().round(2))
