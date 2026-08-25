"""
build_vitalsign.py

【目的】
    從 chartevents 重建 vitalsign 衍生表，將生命徵象相關 itemid（心跳、
    血壓、呼吸速率、體溫、SpO2、血糖）依 mimic-code 邏輯彙整為每個
    (subject_id, stay_id, charttime) 一列的寬表，並清除超出生理合理
    範圍的離群值。

【輸出欄位】
    heart_rate, sbp, dbp, mbp, sbp_ni, dbp_ni, mbp_ni, resp_rate,
    temperature, temperature_site, spo2, glucose

【輸出檔案】
    - data/derived/vitalsign.parquet
"""

import duckdb
from pathlib import Path
import os
MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR")
if not MIMIC_DATA_DIR:
    raise RuntimeError(
        "Environment variable MIMIC_DATA_DIR is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local MIMIC-IV raw-data / DuckDB project root."
    )


# === 路徑設定 ===
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
base_dir = Path(rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1")
icu_parquet = base_dir / "icu_parquet" / "chartevents.parquet"
output_dir = base_dir / "derived"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "vitalsign.parquet"

# === DuckDB 初始化 ===
con = duckdb.connect()

print("🚀 建立 vitalsign 衍生表（依照 mimic-code）")

# === SQL（依生理合理範圍過濾離群值後，依時間點取平均） ===
vitalsign_sql = f"""
COPY (
SELECT
  ce.subject_id,
  ce.stay_id,
  ce.charttime,

  -- Heart Rate
  AVG(CASE WHEN itemid IN (220045) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS heart_rate,

  -- Systolic / Diastolic / Mean BP
  AVG(CASE WHEN itemid IN (220179, 220050, 225309) AND valuenum > 0 AND valuenum < 400 THEN valuenum END) AS sbp,
  AVG(CASE WHEN itemid IN (220180, 220051, 225310) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS dbp,
  AVG(CASE WHEN itemid IN (220052, 220181, 225312) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS mbp,

  -- Non-invasive BP
  AVG(CASE WHEN itemid = 220179 AND valuenum > 0 AND valuenum < 400 THEN valuenum END) AS sbp_ni,
  AVG(CASE WHEN itemid = 220180 AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS dbp_ni,
  AVG(CASE WHEN itemid = 220181 AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS mbp_ni,

  -- Respiratory Rate
  AVG(CASE WHEN itemid IN (220210, 224690) AND valuenum > 0 AND valuenum < 70 THEN valuenum END) AS resp_rate,

  -- Temperature (convert F → C)
  ROUND(
    TRY_CAST(AVG(
      CASE
        WHEN itemid IN (223761) AND valuenum > 70 AND valuenum < 120 THEN (valuenum - 32) / 1.8
        WHEN itemid IN (223762) AND valuenum > 10 AND valuenum < 50 THEN valuenum
      END
    ) AS DECIMAL),
    2
  ) AS temperature,

  MAX(CASE WHEN itemid = 224642 THEN value END) AS temperature_site,

  -- SpO2
  AVG(CASE WHEN itemid IN (220277) AND valuenum > 0 AND valuenum <= 100 THEN valuenum END) AS spo2,

  -- Glucose
  AVG(CASE WHEN itemid IN (225664, 220621, 226537) AND valuenum > 0 THEN valuenum END) AS glucose

FROM read_parquet('{icu_parquet}') AS ce

WHERE ce.stay_id IS NOT NULL
  AND ce.itemid IN (
    220045, 225309, 225310, 225312, 220050, 220051, 220052,
    220179, 220180, 220181, 220210, 224690, 220277, 225664,
    220621, 226537, 223762, 223761, 224642
  )

GROUP BY ce.subject_id, ce.stay_id, ce.charttime
)
TO '{output_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
"""

# === 執行 ===
con.execute(vitalsign_sql)
print("✅ 轉換完成:", output_file)

# === 檢查前 5 筆 ===
sample = con.execute(f"SELECT * FROM read_parquet('{output_file}') LIMIT 5").df()
print(sample)
