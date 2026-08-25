# =============================================================
# filter_tracheostomy.py
# 排除「拔管時間 等於 氣切時間」的案例
# =============================================================
import duckdb
import pandas as pd
from pathlib import Path

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
db_path      = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"
subject_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\mv_day_unique_subject.csv"
output_path  = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\mv_day_unique_subject_filtered.csv"

# =============================================================
# 1️⃣ 找出每個 stay_id 的「最早氣切時間」
# =============================================================
print("🚀 從 DuckDB ventilation 表讀取氣切紀錄...")
con = duckdb.connect(db_path)
trach_records = con.execute("""
    SELECT stay_id, starttime
    FROM ventilation
    WHERE ventilation_status = 'Tracheostomy'
""").df()
trach_records["starttime"] = pd.to_datetime(trach_records["starttime"])

# 找出每個 stay_id 的「第一筆」氣切開始時間
# groupby stay_id -> 取 starttime 的最小值
first_trach_time = (
    trach_records.groupby("stay_id", as_index=False)
    .agg(trach_start_time=("starttime", "min"))
)
con.close()

print(f"⚠️ 共有 {len(first_trach_time)} 個 stay_id 擁有氣切紀錄。")
print(first_trach_time.head())

# =============================================================
# 2️⃣ 與你的主資料表合併並比對
# =============================================================
print("\n🔄 讀取 mv_day_unique_subject.csv 進行時間比對...")
df_subject = pd.read_csv(subject_path, parse_dates=["endtime"])
original_count = len(df_subject)

# 將氣切時間 merge 進來 (使用 left join，因為很多病人根本沒有氣切)
merged_df = pd.merge(df_subject, first_trach_time, on="stay_id", how="left")

# =============================================================
# 3️⃣ 執行篩選條件
# =============================================================
# 條件 A: 沒有氣切紀錄 (trach_start_time 是 NaT/Null) -> 保留
# 條件 B: 有氣切紀錄，但是 拔管時間 (endtime) 早於 氣切開始時間 -> 保留

condition_keep = (
    (merged_df["trach_start_time"].isna()) | 
    (merged_df["endtime"] < merged_df["trach_start_time"])
)

df_clean = merged_df[condition_keep].copy()

# 移除剛剛暫時借用的 trach_start_time 欄位，保持版面乾淨
df_clean = df_clean.drop(columns=["trach_start_time"])

removed_count = original_count - len(df_clean)

# =============================================================
# 4️⃣ 結果輸出
# =============================================================
print(f"📊 原始資料筆數: {original_count}")
print(f"✂️ 排除筆數 (拔管發生在氣切之後): {removed_count}")
print(f"✅ 最終資料筆數: {len(df_clean)}")

df_clean.to_csv(output_path, index=False)
print(f"\n📁 檔案已儲存至: {output_path}")

# 🔍 檢查一下被排除的案例 (驗證邏輯是否正確)
removed_cases = merged_df[~condition_keep]
if not removed_cases.empty:
    print("\n🔍 [檢查] 以下是被排除的案例範例 (endtime >= trach_start_time):")
    print(removed_cases[["stay_id", "endtime", "trach_start_time"]].head())