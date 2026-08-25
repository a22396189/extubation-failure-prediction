"""
build_stay_subject_map.py

【目的】
    從 icustays.csv 建立 stay_id <-> subject_id 對照表，供其他腳本
    以 stay_id 反查所屬病人 subject_id。

【輸出】
    - data/outputs/stay_subject_map.csv
"""

import duckdb
import pandas as pd
from pathlib import Path

# === 路徑設定 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
icu_csv = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\icu\icustays.csv"
output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "stay_subject_map.csv"

# === 方法 1：使用 DuckDB 直接讀壓縮檔 ===
con = duckdb.connect()

df = con.execute(f"""
SELECT DISTINCT stay_id, subject_id
FROM read_csv_auto('{icu_csv}', SAMPLE_SIZE=-1)
WHERE stay_id IS NOT NULL AND subject_id IS NOT NULL
ORDER BY stay_id;
""").df()

# === 儲存成 CSV ===
df.to_csv(output_path, index=False)
print(f"✅ stay_subject_map.csv 已建立，共 {len(df)} 筆對照")
print(f"📁 儲存位置: {output_path}")
print(df.head())
