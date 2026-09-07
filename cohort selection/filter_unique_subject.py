import pandas as pd
import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )

# =============================================================
# filter_unique_subject.py 將 stay_id 對應到 subject_id，若同一病人多次住院，保留 endtime 最早的那一筆
# 設定路徑
# =============================================================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
mv_file_path = rf"{EXTUBATION_ROOT}\data\outputs\mv_day_from_continuous.csv" # subject_id 最長的一段連續 invasive MV episode
map_file_path = rf"{EXTUBATION_ROOT}\data\outputs\stay_subject_map.csv" 
output_path = rf"{EXTUBATION_ROOT}\data\outputs\mv_day_unique_subject.csv"

# =============================================================
# 1️⃣ 讀取資料
# =============================================================
print("🚀 讀取檔案中...")
df_mv = pd.read_csv(mv_file_path)
df_map = pd.read_csv(map_file_path)

# 確保時間欄位是 datetime 格式 (這步很重要，否則排序會錯)
if "endtime" in df_mv.columns:
    df_mv["endtime"] = pd.to_datetime(df_mv["endtime"])

print(f"📌 原始 MV 資料筆數 (Stay Level): {len(df_mv)}")

# =============================================================
# 2️⃣ 合併 Subject_ID
# =============================================================
# 使用 merge 將 subject_id 對應進來
# how='inner' 確保只有同時存在於 mapping 表的資料才保留 (比較安全)
merged_df = pd.merge(df_mv, df_map, on="stay_id", how="inner")

print(f"🔗 合併後資料筆數: {len(merged_df)}")

# =============================================================
# 3️⃣ 篩選邏輯：同 Subject 取 Endtime 最早
# =============================================================

# 步驟 A: 排序
# 先依 subject_id 分組，再依 endtime 由小到大排 (最早的時間在最上面)
merged_df = merged_df.sort_values(by=["subject_id", "endtime"], ascending=[True, True])

# 步驟 B: 去重
# subset=['subject_id']: 只要 subject_id 重複就視為重複
# keep='first': 保留排序後的第一筆 (也就是 endtime 最早的那筆)
final_df = merged_df.drop_duplicates(subset=["subject_id"], keep="first")

# =============================================================
# 4️⃣ 檢查與輸出
# =============================================================
removed_count = len(merged_df) - len(final_df)
print(f"\n✂️ 移除重複 Subject 的筆數: {removed_count}")
print(f"✅ 最終剩餘筆數 (Unique Subject Level): {len(final_df)}")

# 儲存結果
final_df.to_csv(output_path, index=False)
print(f"\n📁 檔案已儲存至: {output_path}")

# 驗證是否有重複 subject_id
is_unique = final_df["subject_id"].is_unique
print(f"🧪 驗證 subject_id 是否唯一? {is_unique}")

# 預覽
print("\n🔍 前 5 筆預覽：")
print(final_df[["subject_id", "stay_id", "endtime", "MV_days"]].head())