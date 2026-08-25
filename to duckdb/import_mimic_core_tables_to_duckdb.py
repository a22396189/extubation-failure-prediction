"""
import_mimic_core_tables_to_duckdb.py

【目的】
    將 MIMIC-IV 核心 CSV 表（admissions、patients、icustays）匯入本機
    DuckDB 資料庫，供後續各 build_*.py 腳本以 SQL join 存取。已存在的
    表會跳過匯入（create_table_if_not_exists），避免重複匯入。

【輸出】
    - DuckDB 資料庫中的 mimiciv_hosp.admissions / mimiciv_hosp.patients /
      mimiciv_icu.icustays 三張表，並建立常用 join 欄位索引
"""

import os
import duckdb
MIMIC_DATA_DIR = os.environ.get("MIMIC_DATA_DIR")
if not MIMIC_DATA_DIR:
    raise RuntimeError(
        "Environment variable MIMIC_DATA_DIR is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local MIMIC-IV raw-data / DuckDB project root."
    )


# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
DB_PATH = rf"{MIMIC_DATA_DIR}\mimic.duckdb"

admissions_csv = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\admissions.csv"
patients_csv   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\patients.csv"
icustays_csv   = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\icu\icustays.csv"

con = duckdb.connect(DB_PATH)

# 建 schema
con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_hosp;")
con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_icu;")

def create_table_if_not_exists(schema_table: str, csv_path: str):
    """若 schema_table 尚不存在於 DuckDB 中，從 csv_path 讀入並建表；已存在則跳過。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    schema, table = schema_table.split(".")
    exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchone()[0]

    if exists:
        print(f"[SKIP] {schema_table} already exists.")
        return

    print(f"[IMPORT] Creating {schema_table} from {csv_path}")
    con.execute(f"""
        CREATE TABLE {schema_table} AS
        SELECT * FROM read_csv_auto('{csv_path}', header=True, sample_size=-1);
    """)

create_table_if_not_exists("mimiciv_hosp.admissions", admissions_csv)
create_table_if_not_exists("mimiciv_hosp.patients", patients_csv)
create_table_if_not_exists("mimiciv_icu.icustays", icustays_csv)

# 可選：建立索引加速 join（失敗也沒關係）
try:
    con.execute("CREATE INDEX IF NOT EXISTS idx_icustays_stay_id ON mimiciv_icu.icustays(stay_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_icustays_subj ON mimiciv_icu.icustays(subject_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_adm_subj_hadm ON mimiciv_hosp.admissions(subject_id, hadm_id);")
except Exception as e:
    print("[INFO] Index creation skipped/failed (OK):", e)

print("\n[DONE] Imported core tables into", DB_PATH)

# 驗證
print(con.execute("SELECT COUNT(*) AS n_icustays FROM mimiciv_icu.icustays;").df())
print(con.execute("SELECT COUNT(*) AS n_adm FROM mimiciv_hosp.admissions;").df())
print(con.execute("SELECT COUNT(*) AS n_pat FROM mimiciv_hosp.patients;").df())
