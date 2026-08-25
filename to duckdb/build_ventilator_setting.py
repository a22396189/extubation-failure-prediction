"""
build_ventilator_setting.py

【目的】
    從 chartevents 重建 ventilator_setting 衍生表：彙整呼吸器設定相關
    itemid（呼吸速率、潮氣容積、PEEP、FiO2、呼吸器模式等）為每個
    (subject_id, charttime) 一列的寬表。SQL 邏輯移植自 MIT-LCP
    mimic-code 的 ventilator_setting.sql（DuckDB 版本）。

【輸入】
    - chartevents.parquet（透過 mimiciv_icu_chartevents view 讀取）

【輸出】
    - DuckDB 資料庫中的 ventilator_setting 表，供 build_ventilation.py
      與時序特徵萃取腳本使用
"""

import duckdb
import os
MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR")
if not MIMIC_DATA_DIR:
    raise RuntimeError(
        "Environment variable MIMIC_DATA_DIR is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local MIMIC-IV raw-data / DuckDB project root."
    )


# === 連線到 DuckDB 資料庫 ===
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
con = duckdb.connect(rf"{MIMIC_DATA_DIR}\mimic.duckdb")

# === 掛載 chartevents parquet 作為 view ===
con.execute("""
CREATE OR REPLACE VIEW mimiciv_icu_chartevents AS
SELECT * FROM read_parquet(f'{MIMIC_DATA_DIR}/data/mimic-iv-3.1/icu_parquet/chartevents.parquet');
""")

# === MIT-LCP ventilator_setting SQL (DuckDB 版本) ===
ventset_sql = r"""
DROP TABLE IF EXISTS ventilator_setting;

CREATE TABLE ventilator_setting AS
-- 步驟1：篩選呼吸器設定相關 itemid，並清除超出合理範圍的離群值
--   itemid 223835 = FiO2：統一換算為百分比（0.2~1 視為小數，20~100 視為百分比）
--   itemid 220339/224700 = PEEP：超過 100 或小於 0 視為異常值
WITH ce AS (
  SELECT
    subject_id,
    stay_id,
    charttime,
    itemid,
    value,
    CASE
      WHEN itemid = 223835 THEN CASE
        WHEN valuenum >= 0.20 AND valuenum <= 1 THEN valuenum * 100
        WHEN valuenum > 1 AND valuenum < 20 THEN NULL
        WHEN valuenum >= 20 AND valuenum <= 100 THEN valuenum
        ELSE NULL
      END
      WHEN itemid IN (220339, 224700) THEN CASE
        WHEN valuenum > 100 THEN NULL
        WHEN valuenum < 0 THEN NULL
        ELSE valuenum
      END
      ELSE valuenum
    END AS valuenum,
    valueuom,
    storetime
  FROM mimiciv_icu_chartevents
  WHERE value IS NOT NULL
    AND stay_id IS NOT NULL
    AND itemid IN (
      224688, 224689, 224690, 224687, 224685, 224684, 224686, 224696,
      220339, 224700, 223835, 223849, 229314, 223848, 224691
    )
)
-- 步驟2：依 itemid 轉置為寬表，每個 (subject_id, charttime) 一列
SELECT
  subject_id,
  MAX(stay_id) AS stay_id,
  charttime,
  MAX(CASE WHEN itemid = 224688 THEN valuenum END) AS respiratory_rate_set,
  MAX(CASE WHEN itemid = 224690 THEN valuenum END) AS respiratory_rate_total,
  MAX(CASE WHEN itemid = 224689 THEN valuenum END) AS respiratory_rate_spontaneous,
  MAX(CASE WHEN itemid = 224687 THEN valuenum END) AS minute_volume,
  MAX(CASE WHEN itemid = 224684 THEN valuenum END) AS tidal_volume_set,
  MAX(CASE WHEN itemid = 224685 THEN valuenum END) AS tidal_volume_observed,
  MAX(CASE WHEN itemid = 224686 THEN valuenum END) AS tidal_volume_spontaneous,
  MAX(CASE WHEN itemid = 224696 THEN valuenum END) AS plateau_pressure,
  MAX(CASE WHEN itemid IN (220339, 224700) THEN valuenum END) AS peep,
  MAX(CASE WHEN itemid = 223835 THEN valuenum END) AS fio2,
  MAX(CASE WHEN itemid = 224691 THEN valuenum END) AS flow_rate,
  MAX(CASE WHEN itemid = 223849 THEN value END) AS ventilator_mode,
  MAX(CASE WHEN itemid = 229314 THEN value END) AS ventilator_mode_hamilton,
  MAX(CASE WHEN itemid = 223848 THEN value END) AS ventilator_type
FROM ce
GROUP BY
  subject_id,
  charttime;
"""

print("🚀 建立 ventilator_setting 表中 ...")
con.execute(ventset_sql)
print("✅ 建立完成！")

# === 快速檢查前 5 筆 ===
df = con.execute("SELECT * FROM ventilator_setting LIMIT 5").df()
print(df)
