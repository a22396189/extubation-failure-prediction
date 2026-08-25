# =============================================================
# compute_mv_day_from_continuous.py (FINAL VERSION)
# 從 ventilation_mv3d_continuous.csv 中提取每個 stay_id
# 「最長的一段」連續 invasive MV episode (每個 stay_id 只會有一筆資料)
# =============================================================

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


# === 路徑設定 ===
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
input_path = rf"{EXTUBATION_ROOT}\data\outputs\ventilation_mv3d_continuous.csv"
output_path = rf"{EXTUBATION_ROOT}\data\outputs\mv_day_from_continuous.csv"

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
