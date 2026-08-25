# =============================================================
# build_extubation_features_sex.py
#
# 【目的】
#   提取每位 ICU 病人的性別，作為靜態特徵之一。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / subject_id
#   - patients.csv (MIMIC-IV hosp)：subject_id → gender（M/F）
#
# 【處理邏輯】
#   以 subject_id 直接合併 patients 表，gender 欄位值統一轉換：
#     M → Male、F → Female
#   欄位重新命名為 sex（與其他特徵檔命名慣例一致）。
#
# 【輸出】
#   - extubation_features_sex.csv：
#       含 sex（Male/Female）、subject_id、stay_id、Extubation_failure
#
# 【執行步驟】
#   Step 1：讀取 patients.csv，取 subject_id 與 gender
#   Step 2：讀取拔管名單（extubation_outcome）
#   Step 3：以 subject_id 合併性別資訊，標準化值為 Male/Female
#   Step 4：輸出 CSV
# =============================================================
import duckdb
import pandas as pd
from pathlib import Path

# === 路徑設定 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
duckdb_path = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"
patients_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\hosp\patients.csv"
extub_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_outcome.csv"

output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_sex.csv"

# === 1️⃣ 讀取資料 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 patients.csv ...")

# 讀取 subject_id 與 gender
patients = con.execute(f"SELECT subject_id, gender FROM read_csv_auto('{patients_path}', SAMPLE_SIZE=-1)").df()
print(f"✅ 共 {len(patients):,} 筆病人基本資料")

# === 2️⃣ 讀取拔管 outcome ===
extub = pd.read_csv(extub_path)
print(f"✅ 共 {len(extub):,} 筆拔管紀錄")

# === 3️⃣ 合併性別資訊 ===
merged = extub.merge(patients, on="subject_id", how="left")

# 標準化性別值 (轉為 Male / Female)
merged["gender"] = merged["gender"].replace({"M": "Male", "F": "Female"})

# === 4️⃣ 產生輸出欄位 ===
# Extubation_failure (1=失敗)
output_df = merged[["subject_id", "stay_id", "gender", "Extubation_failure"]].copy()
output_df.rename(columns={"gender": "sex"}, inplace=True)

# === 5️⃣ 輸出結果 ===
output_df.to_csv(output_path, index=False)
print(f"✅ Sex feature generated successfully: {len(output_df)} rows")
print(f"📁 Saved to: {output_path}")

print("\n🔍 前 10 筆資料：")
print(output_df.head(10))