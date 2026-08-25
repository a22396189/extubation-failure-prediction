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
icu_csv = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\icu\icustays.csv"
output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs")
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
