# =============================================================
# compute_mv_day_from_continuous.py (FINAL VERSION)
# 從 ventilation_mv3d_continuous.csv 中提取每個 stay_id
# 「最長的一段」連續 invasive MV episode (每個 stay_id 只會有一筆資料)
# =============================================================

import pandas as pd
from pathlib import Path

# === 路徑設定 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
input_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\ventilation_mv3d_continuous.csv"
output_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\mv_day_from_continuous.csv"

print("🚀 讀取 ventilation_mv3d_continuous.csv ...")
df = pd.read_csv(input_path, parse_dates=["starttime", "endtime"])

# =============================================================
# 1️⃣ 直接使用已計算好的 MV_days（這是「連續 episode 的時間長度」）
# =============================================================

if "MV_days" not in df.columns:
    df["MV_days"] = (df["endtime"] - df["starttime"]).dt.total_seconds() / 86400

print("📌 原始連續 MV episode 數量：", len(df))

# =============================================================
# 2️⃣ 對每個 stay_id 找出「最長」的連續 MV episode
# =============================================================
# idxmax() 找到該 stay_id 內 MV_days 最大的列的 index

longest_mv = df.loc[df.groupby("stay_id")["MV_days"].idxmax()].reset_index(drop=True)

print("🔥 最長連續 MV episode 數量（每 stay_id 1 筆）：", len(longest_mv))

# =============================================================
# 3️⃣ 驗證所有 MV_days 是否 ≥ 3 天
# =============================================================
valid_check = (longest_mv["MV_days"] >= 3).all()

print(f"🧪 MV_days 全部 ≥ 3 天？ {valid_check}")

# =============================================================
# 4️⃣ 輸出結果
# =============================================================
longest_mv.to_csv(output_path, index=False)
print(f"📁 已輸出最長連續 MV episode: {output_path}")

print("\n🔍 前 10 筆預覽：")
print(longest_mv.head(10))
