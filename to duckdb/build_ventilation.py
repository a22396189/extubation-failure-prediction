"""
build_ventilation.py

【目的】
    整合 ventilator_setting 與 oxygen_delivery 兩表，依 MIT-LCP mimic-code
    的邏輯判斷病人每個時間點的呼吸支持狀態（InvasiveVent / NonInvasiveVent /
    HFNC / SupplementalOxygen / Tracheostomy / None），並將連續相同狀態的
    紀錄合併為一段區間（start/end time），建立 ventilation 表。

【前置依賴】
    需先執行 build_ventilator_setting.py 與 build_oxygen_delivery.py，
    產生 ventilator_setting、oxygen_delivery 兩張表。

【輸出】
    - DuckDB 資料庫中的 ventilation 表（每列代表一段呼吸支持區間）
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

# === 掛載 derived tables ===
print("🚀 檢查前置表...")
print(con.execute("SHOW TABLES").df())

# === MIT-LCP ventilation SQL (DuckDB 版本) ===
ventilation_sql = r"""
DROP TABLE IF EXISTS ventilation;

CREATE TABLE ventilation AS
-- 步驟1：合併兩張來源表出現過的所有 (stay_id, charttime) 時間點
WITH tm AS (
  SELECT stay_id, charttime FROM ventilator_setting
  UNION
  SELECT stay_id, charttime FROM oxygen_delivery
),
-- 步驟2：依給氧裝置 / 呼吸器模式，將每個時間點分類為呼吸支持狀態
vs AS (
  SELECT
    tm.stay_id,
    tm.charttime,
    od.o2_delivery_device_1,
    COALESCE(vs.ventilator_mode, vs.ventilator_mode_hamilton) AS vent_mode,
    CASE
      WHEN od.o2_delivery_device_1 IN ('Tracheostomy tube', 'Trach mask ')
        THEN 'Tracheostomy'
      WHEN od.o2_delivery_device_1 IN ('Endotracheal tube')
        OR vs.ventilator_mode IN (
          '(S) CMV', 'APRV', 'APRV/Biphasic+ApnPress', 'APRV/Biphasic+ApnVol', 'APV (cmv)',
          'Ambient', 'Apnea Ventilation', 'CMV', 'CMV/ASSIST', 'CMV/ASSIST/AutoFlow', 'CMV/AutoFlow',
          'CPAP/PPS', 'CPAP/PSV', 'CPAP/PSV+Apn TCPL', 'CPAP/PSV+ApnPres', 'CPAP/PSV+ApnVol',
          'MMV', 'MMV/AutoFlow', 'MMV/PSV', 'MMV/PSV/AutoFlow', 'P-CMV', 'PCV+', 'PCV+/PSV',
          'PCV+Assist', 'PRES/AC', 'PRVC/AC', 'PRVC/SIMV', 'PSV/SBT', 'SIMV', 'SIMV/AutoFlow',
          'SIMV/PRES', 'SIMV/PSV', 'SIMV/PSV/AutoFlow', 'SIMV/VOL', 'SYNCHRON MASTER',
          'SYNCHRON SLAVE', 'VOL/AC'
        )
        OR vs.ventilator_mode_hamilton IN (
          'APRV', 'APV (cmv)', 'Ambient', '(S) CMV', 'P-CMV', 'SIMV', 'APV (simv)',
          'P-SIMV', 'VS', 'ASV'
        )
        THEN 'InvasiveVent'
      WHEN od.o2_delivery_device_1 IN ('Bipap mask ', 'CPAP mask ')
        OR vs.ventilator_mode_hamilton IN ('DuoPaP', 'NIV', 'NIV-ST')
        THEN 'NonInvasiveVent'
      WHEN od.o2_delivery_device_1 IN ('High flow nasal cannula')
        THEN 'HFNC'
      WHEN od.o2_delivery_device_1 IN (
          'Non-rebreather', 'Face tent', 'Aerosol-cool', 'Venti mask ',
          'Medium conc mask ', 'Ultrasonic neb', 'Vapomist', 'Oxymizer',
          'High flow neb', 'Nasal cannula'
        )
        THEN 'SupplementalOxygen'
      WHEN od.o2_delivery_device_1 IN ('None')
        THEN 'None'
      ELSE NULL
    END AS ventilation_status
  FROM tm
  LEFT JOIN ventilator_setting AS vs
    ON tm.stay_id = vs.stay_id AND tm.charttime = vs.charttime
  LEFT JOIN oxygen_delivery AS od
    ON tm.stay_id = od.stay_id AND tm.charttime = od.charttime
),
-- 步驟3：標記狀態改變點（與前一筆狀態不同，或間隔 >=14 小時視為新事件）
vd0 AS (
  SELECT
    stay_id,
    charttime,
    LAG(charttime, 1) OVER (PARTITION BY stay_id, ventilation_status ORDER BY charttime NULLS FIRST) AS charttime_lag,
    LEAD(charttime, 1) OVER w AS charttime_lead,
    ventilation_status,
    LAG(ventilation_status, 1) OVER w AS ventilation_status_lag
  FROM vs
  WHERE ventilation_status IS NOT NULL
  WINDOW w AS (PARTITION BY stay_id ORDER BY charttime NULLS FIRST)
),
-- 步驟4：計算與前一筆的時間差，決定是否視為新的呼吸支持事件（new_ventilation_event=1）
vd1 AS (
  SELECT
    stay_id,
    charttime,
    charttime_lag,
    charttime_lead,
    ventilation_status,
    (DATE_DIFF('microseconds', charttime_lag, charttime) / 3600000000.0) AS ventduration_hr,
    CASE
      WHEN ventilation_status_lag IS NULL THEN 1
      WHEN DATE_DIFF('microseconds', charttime_lag, charttime) / 3600000000.0 >= 14 THEN 1
      WHEN ventilation_status_lag <> ventilation_status THEN 1
      ELSE 0
    END AS new_ventilation_event
  FROM vd0
),
-- 步驟5：以 new_ventilation_event 的累加和作為事件序號 vent_seq，供後續分組
vd2 AS (
  SELECT
    stay_id,
    charttime,
    charttime_lead,
    ventilation_status,
    ventduration_hr,
    new_ventilation_event,
    SUM(new_ventilation_event) OVER (PARTITION BY stay_id ORDER BY charttime NULLS FIRST) AS vent_seq
  FROM vd1
)
-- 步驟6：依 vent_seq 分組，將同一段連續事件彙整成 starttime/endtime 一筆紀錄
SELECT
  stay_id,
  MIN(charttime) AS starttime,
  MAX(
    CASE
      WHEN charttime_lead IS NULL
        OR DATE_DIFF('microseconds', charttime, charttime_lead) / 3600000000.0 >= 14
      THEN charttime
      ELSE charttime_lead
    END
  ) AS endtime,
  MAX(ventilation_status) AS ventilation_status
FROM vd2
GROUP BY stay_id, vent_seq
HAVING MIN(charttime) <> MAX(charttime);
"""

# === 執行 ===
print("🚀 建立 ventilation 表中 ...")
con.execute(ventilation_sql)
print("✅ ventilation 建立完成！")

# === 檢查結果 ===
df = con.execute("SELECT * FROM ventilation LIMIT 10").df()
print(df)
