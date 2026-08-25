# ==========================================================
# build_ventilation_mv3d_continuous.py
# 只保留「連續 invasive MV ≥ 72 小時」的病人與 episode
# ==========================================================

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


# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
con = duckdb.connect(rf"{MIMIC_DATA_DIR}\mimic.duckdb")

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ventilation_mv3d_continuous.csv"

print("🚀 讀取 ventilation 資料 ...")

# ventilation 需包含：
# stay_id, ventilation_status, starttime, endtime
vent = con.execute("""
    SELECT stay_id, ventilation_status, starttime, endtime
    FROM ventilation
    WHERE ventilation_status = 'InvasiveVent'
""").df()

# 轉時間格式
vent["starttime"] = pd.to_datetime(vent["starttime"])
vent["endtime"]   = pd.to_datetime(vent["endtime"])

print(f"📌 共 {len(vent):,} 筆 invasive MV 紀錄")

# ==========================================================
# 1️⃣ 對每個病人依 starttime 排序
# ==========================================================
vent = vent.sort_values(["stay_id", "starttime"])

# ==========================================================
# 2️⃣ 找出「連續」MV episode
#    若上一段 endtime 與下一段 starttime 間隔 <= 1 小時，則視為同一連續 episode
# ==========================================================
continuous_records = []

current = None
for row in vent.itertuples():
    if current is None:
        current = [row.stay_id, row.starttime, row.endtime]
        continue

    same_stay = (row.stay_id == current[0])
    continuous = (row.starttime - current[2]).total_seconds() <= 3600

    if same_stay and continuous:
        # 延長 episode
        current[2] = max(current[2], row.endtime)
    else:
        # 儲存上一段 episode
        continuous_records.append(current)
        current = [row.stay_id, row.starttime, row.endtime]

# 最後一段記得加入
if current:
    continuous_records.append(current)

cont_df = pd.DataFrame(continuous_records, columns=["stay_id", "starttime", "endtime"])
cont_df["MV_days"] = (cont_df["endtime"] - cont_df["starttime"]).dt.total_seconds() / 86400

print(f"📘 找到 {len(cont_df):,} 個連續 invasive MV episodes")

# ==========================================================
# 3️⃣ 只保留 ≥ 72 小時的連續 MV episode
# ==========================================================
mv3d = cont_df[cont_df["MV_days"] >= 3].copy()
print(f"🔥 連續 ≥ 72 小時 MV episodes: {len(mv3d):,}")

# ==========================================================
# 4️⃣ 輸出
# ==========================================================
mv3d.to_csv(output_path, index=False)
print(f"✅ 已輸出: {output_path}")
