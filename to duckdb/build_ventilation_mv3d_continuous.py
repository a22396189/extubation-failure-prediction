# ==========================================================
# build_ventilation_mv3d_continuous.py
# 只保留「連續 invasive MV ≥ 72 小時」的病人與 episode
# ==========================================================

import duckdb
import pandas as pd
from pathlib import Path

# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
con = duckdb.connect(r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb")

output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
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
