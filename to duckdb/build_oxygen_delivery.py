"""
build_oxygen_delivery.py

【目的】
    從 MIMIC-IV ICU chartevents 重建 oxygen_delivery 衍生表，記錄病人於
    每個時間點使用的給氧方式（如 Nasal cannula、Endotracheal tube 等）
    與氧氣流量。SQL 邏輯移植自 MIT-LCP mimic-code 專案的
    oxygen_delivery.sql，改寫為 DuckDB 語法。

【輸入】
    - chartevents.parquet（MIMIC-IV ICU 事件表）

【輸出】
    - DuckDB 資料庫中的 oxygen_delivery 表，供 build_ventilation.py 等
      後續腳本 join 使用

【注意】
    - 路徑為原分析機器上的絕對路徑，於其他環境執行前請先修改
"""

import duckdb

# === 連線到 DuckDB 資料庫 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
con = duckdb.connect(r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb")

# === 掛載 chartevents parquet 檔作為 view ===
con.execute("""
CREATE OR REPLACE VIEW mimiciv_icu_chartevents AS
SELECT * FROM read_parquet('C:/Users/your-username/Desktop/extubation_project/data/mimic-iv-3.1/icu_parquet/chartevents.parquet');
""")

# === MIT-LCP oxygen_delivery SQL（DuckDB 版本） ===
oxygen_sql = r"""
DROP TABLE IF EXISTS oxygen_delivery;
CREATE TABLE oxygen_delivery AS
-- 步驟1：取出氧氣流量相關 itemid（223834=O2 flow, 227582 合併為同一 itemid, 227287=額外流量），
--        同一 itemid 的重複量測合併為一筆
WITH ce_stg1 AS (
  SELECT
    subject_id,
    stay_id,
    charttime,
    CASE WHEN itemid IN (223834, 227582) THEN 223834 ELSE itemid END AS itemid,
    value,
    valuenum,
    valueuom,
    storetime
  FROM mimiciv_icu_chartevents
  WHERE value IS NOT NULL
    AND itemid IN (223834, 227582, 227287)
),
-- 步驟2：同一時間點若有多筆紀錄，取最後 storetime 的一筆為準
ce_stg2 AS (
  SELECT
    subject_id,
    stay_id,
    charttime,
    itemid,
    value,
    valuenum,
    valueuom,
    ROW_NUMBER() OVER (PARTITION BY subject_id, charttime, itemid ORDER BY storetime DESC) AS rn
  FROM ce_stg1
),
-- 步驟3：取出給氧裝置（itemid=226732），同一時間可能有多個裝置，依序編號 rn=1..4
o2 AS (
  SELECT
    subject_id,
    stay_id,
    charttime,
    itemid,
    value AS o2_device,
    ROW_NUMBER() OVER (PARTITION BY subject_id, charttime, itemid ORDER BY value NULLS FIRST) AS rn
  FROM mimiciv_icu_chartevents
  WHERE itemid = 226732
),
-- 步驟4：以 subject_id + charttime 全外聯接流量與裝置資料
stg AS (
  SELECT
    COALESCE(ce.subject_id, o2.subject_id) AS subject_id,
    COALESCE(ce.stay_id, o2.stay_id) AS stay_id,
    COALESCE(ce.charttime, o2.charttime) AS charttime,
    COALESCE(ce.itemid, o2.itemid) AS itemid,
    ce.value,
    ce.valuenum,
    o2.o2_device,
    o2.rn
  FROM ce_stg2 AS ce
  FULL OUTER JOIN o2
    ON ce.subject_id = o2.subject_id AND ce.charttime = o2.charttime
  WHERE ce.rn = 1
)
-- 步驟5：依 subject_id + charttime 彙整成單列，最多列出 4 個同時使用的給氧裝置
SELECT
  subject_id,
  MAX(stay_id) AS stay_id,
  charttime,
  MAX(CASE WHEN itemid = 223834 THEN valuenum END) AS o2_flow,
  MAX(CASE WHEN itemid = 227287 THEN valuenum END) AS o2_flow_additional,
  MAX(CASE WHEN rn = 1 THEN o2_device END) AS o2_delivery_device_1,
  MAX(CASE WHEN rn = 2 THEN o2_device END) AS o2_delivery_device_2,
  MAX(CASE WHEN rn = 3 THEN o2_device END) AS o2_delivery_device_3,
  MAX(CASE WHEN rn = 4 THEN o2_device END) AS o2_delivery_device_4
FROM stg
GROUP BY
  subject_id,
  charttime;
"""

print("🚀 建立 oxygen_delivery 表中 ...")
con.execute(oxygen_sql)
print("✅ 建立完成！")

# === 快速檢查前 5 筆 ===
df = con.execute("SELECT * FROM oxygen_delivery LIMIT 5").df()
print(df)
