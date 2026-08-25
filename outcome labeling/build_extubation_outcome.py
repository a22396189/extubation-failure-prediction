# =============================================================
# build_extubation_outcome.py
#
# 【目的】
#   為每位 ICU 病人（每個 stay_id）建立拔管失敗（Extubation Failure）標籤。
#
# 【輸入】
#   - mv_day_unique_subject_filtered.csv：已排除氣切案例的最終 Cohort，
#     含每位病人第一次拔管的 starttime / endtime（即 extubation_time）
#   - DuckDB ventilation 表：完整通氣事件紀錄，用於判斷拔管後 48h 內再插管
#   - admissions.csv (MIMIC-IV)：取得 deathtime，用於判斷拔管後 48h 內死亡
#
# 【拔管失敗定義】
#   拔管後 48 小時內滿足以下任一條件：
#     (1) 再插管（InvasiveVent 或 NonInvasiveVent 重新開始）
#     (2) 院內死亡
#
# 【輸出】
#   - extubation_outcome.csv：含 Extubation_failure 標籤（1=失敗, 0=成功）
#
# 【執行步驟】
#   Step 1：載入 Cohort、通氣紀錄、死亡紀錄
#   Step 2：合併死亡資訊；排除拔管時間 == 死亡時間之資料異常個案
#   Step 3：DuckDB SQL 判斷拔管後 48h 內是否再插管
#   Step 4：整合標籤（reintubated_within_48h / died_within_48h / Extubation_failure）
#   Step 5：輸出最終 CSV
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


# =============================================================
# 路徑設定
# =============================================================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
duckdb_path = rf"{MIMIC_DATA_DIR}\mimic.duckdb"

# 輸入檔案
cohort_csv = rf"{EXTUBATION_ROOT}\data\outputs\mv_day_unique_subject_filtered.csv"
admissions_csv = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\admissions.csv"

# 輸出設定
output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_outcome.csv"

# =============================================================
# DuckDB 連線
# =============================================================
con = duckdb.connect(duckdb_path)

# =============================================================
# 1️⃣ 載入資料
# =============================================================
print("🚀 載入資料中...")

# 讀取 Cohort（加入 starttime）
cohort = pd.read_csv(
    cohort_csv,
    parse_dates=["starttime", "endtime"]
)

cohort = cohort.rename(columns={"endtime": "extubation_time"})

# 讀取住院紀錄（死亡時間）
adm = pd.read_csv(
    admissions_csv,
    usecols=["subject_id", "deathtime"],
    parse_dates=["deathtime"]
)

print(f"📌 初始 Cohort 人數: {len(cohort)}")

# =============================================================
# 2️⃣ 合併死亡資訊（避免資料膨脹）
# =============================================================
# 1. 只保留有 deathtime 的紀錄
death_map = adm.dropna(subset=["deathtime"]).copy()

# 2. 每位病人只保留最早的死亡時間
death_map = (
    death_map
    .sort_values("deathtime")
    .drop_duplicates(subset=["subject_id"], keep="first")
)

print(f"💀 共有 {len(death_map)} 位病人有死亡紀錄")

# 3. 合併回 cohort（不會造成 row 膨脹）
cohort = cohort.merge(death_map, on="subject_id", how="left")

print(f"✅ 合併死亡資訊後人數 (應保持不變): {len(cohort)}")

# -------------------------------------------------------------
# ⚠️ 排除 terminal extubation：拔管時間 = 死亡時間
# 此類紀錄代表撤除維生系統後拔管，非計畫性拔管（planned extubation），
# 不屬於本研究對象，於標籤計算前先行移除。
# -------------------------------------------------------------
before_death_filter = len(cohort)

# 計算各類別數量（供參考）
n_ext_eq_death  = ((cohort["extubation_time"] == cohort["deathtime"]) &
                   cohort["deathtime"].notna()).sum()
n_ext_gt_death  = ((cohort["extubation_time"] >  cohort["deathtime"]) &
                   cohort["deathtime"].notna()).sum()

# 僅排除拔管時間 == 死亡時間（terminal extubation，非計畫性拔管）
# 拔管時間 > 死亡時間之個案視為記錄誤差，予以保留
cohort = cohort[
    (cohort["deathtime"].isna()) |
    (cohort["extubation_time"] != cohort["deathtime"])
].copy()

removed_death_anomaly = before_death_filter - len(cohort)
print(f"ℹ️  拔管時間 == 死亡時間（排除）: {n_ext_eq_death} 筆")
print(f"ℹ️  拔管時間 >  死亡時間（保留）: {n_ext_gt_death} 筆")
print(f"🗑️  共排除: {removed_death_anomaly} 筆")
print(f"✅ 排除後 Cohort 人數: {len(cohort)}")

# =============================================================
# 3️⃣ 使用 DuckDB SQL 判斷 48 小時內是否再插管
# =============================================================
con.register("cohort", cohort)

failure_check_df = con.execute("""
SELECT DISTINCT
       c.subject_id,
       c.stay_id,
       c.extubation_time,
       MAX(
           CASE
               WHEN v.ventilation_status IN ('InvasiveVent', 'NonInvasiveVent')
                    AND v.starttime > c.extubation_time
                    AND v.starttime <= c.extubation_time + INTERVAL 48 HOUR
               THEN 1
               ELSE 0
           END
       ) AS reintubated_within_48h
FROM cohort c
LEFT JOIN ventilation v
  ON c.stay_id = v.stay_id
GROUP BY c.subject_id, c.stay_id, c.extubation_time
""").df()

# =============================================================
# 4️⃣ 整合最終標籤
# =============================================================
final_df = cohort.merge(
    failure_check_df[["stay_id", "reintubated_within_48h"]],
    on="stay_id",
    how="left"
)

# 若無再插管紀錄，補 0
final_df["reintubated_within_48h"] = (
    final_df["reintubated_within_48h"]
    .fillna(0)
    .astype(int)
)

# 判斷是否於 48 小時內死亡
final_df["died_within_48h"] = (
    (final_df["deathtime"] - final_df["extubation_time"])
    .dt.total_seconds()
    .le(48 * 3600)
).fillna(False).astype(int)

# 定義 Extubation Failure
final_df["Extubation_failure"] = (
    (final_df["reintubated_within_48h"] == 1) |
    (final_df["died_within_48h"] == 1)
).astype(int)

# =============================================================
# 5️⃣ 檢查、清理與輸出
# =============================================================
# 死亡異常過濾已於 Step 2 執行（只排除 extubation_time == deathtime）

output_cols = [
    "subject_id",
    "stay_id",
    "starttime",              # ← 來自 mv_day_unique_subject_filtered.csv
    "extubation_time",
    "deathtime",
    "reintubated_within_48h",
    "died_within_48h",
    "Extubation_failure"
]

final_df[output_cols].to_csv(output_path, index=False)

print(f"\n✅ Extubation outcome 已建立，共 {len(final_df)} 位病人")
print(f"📁 儲存位置: {output_path}")

print("\n📊 標籤分佈 (1=失敗, 0=成功):")
print(final_df["Extubation_failure"].value_counts(normalize=True))

print(f"🔴 失敗人數 (Label=1): {final_df['Extubation_failure'].sum()}")
print(f"🟢 成功人數 (Label=0): {len(final_df) - final_df['Extubation_failure'].sum()}")
