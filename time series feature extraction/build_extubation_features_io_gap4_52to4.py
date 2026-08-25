# =============================================================
# build_extubation_features_io_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的輸入輸出量，
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - inputevents.csv (MIMIC-IV icu)：靜脈輸液、藥物等攝入量（mL）
#     時間欄位為 starttime，取 amount / totalamount 加總
#   - outputevents.csv (MIMIC-IV icu)：尿液及其他排出量（mL）
#     時間欄位為 charttime，含特殊處理（itemid=227488 取負值）
#
# 【特徵計算方式】
#   - input_ml：每時間點各輸入事件加總（無紀錄填 0）
#   - urineoutput：每時間點各輸出事件加總（無紀錄填 0）
#   - io_balance：input_ml − urineoutput（液體平衡）
#   ⚠️ 無紀錄以 0 填補（非 NaN），代表該段時間確實無輸入/輸出事件
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 每位病人固定輸出 12 列
#
# 【特徵欄位】
#   input_ml, urineoutput, io_balance（共 3 個連續變數）
#
# 【輸出】
#   - extubation_features_io_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：DuckDB 讀取 inputevents，依 stay_id + starttime 加總 input_ml
#   Step 2：DuckDB 讀取 outputevents，依 stay_id + charttime 加總 urineoutput
#   Step 3：outer join 合併，fillna(0)，計算 io_balance
#   Step 4：逐人篩選觀測窗，floor 分箱，每 bin 平均，補齊 12 bins
#   Step 5：合併 Extubation_failure 標籤，輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_io_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）、每 4 小時平均的輸入輸出 (I/O) 特徵（共12段）
#
# ✅ gap=4h（嚴格）：完全不含 [extub_time-4h, extub_time) 的資料（含剛好 -4h 也排除）
# ✅ 選擇 A：time_bin 代表「區間左端點」（floor 分箱）
#    12 bins: -52, -48, ..., -8
#    對應區間: [-52,-48), [-48,-44), ..., [-8,-4)
# 若該時間段無資料，補齊 NaN
# ==========================================================

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
duckdb_path = rf"{MIMIC_DATA_DIR}\mimic.duckdb"
inputevents_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\icu\inputevents.csv"
outputevents_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\icu\outputevents.csv"
extub_path = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_io_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8

# === 連線 DuckDB ===
con = duckdb.connect(duckdb_path)

print("🚀 讀取 Input/Output 資料 (這可能需要一點時間)...")

# === 1️⃣ 讀取 Inputevents (攝入量) ===
# 注意：inputevents 的時間欄位在不同版本可能是 starttime 或 charttime
input_df = con.execute(f"""
SELECT
    stay_id,
    starttime AS charttime,
    SUM(
        CASE
            WHEN amount IS NOT NULL AND amount > 0 THEN amount
            WHEN totalamount IS NOT NULL AND totalamount > 0 THEN totalamount
            ELSE NULL
        END
    ) AS input_ml
FROM read_csv_auto('{inputevents_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
WHERE (amount > 0 OR totalamount > 0)
GROUP BY stay_id, starttime
""").df()

# === 2️⃣ 讀取 Outputevents (尿量與排出量) ===
output_df = con.execute(f"""
SELECT
    stay_id,
    charttime,
    SUM(
        CASE
            WHEN itemid = 227488 AND value > 0 THEN -1 * value
            WHEN value > 0 THEN value
            ELSE NULL
        END
    ) AS urineoutput
FROM read_csv_auto('{outputevents_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
WHERE itemid IN (
    226559, 226560, 226561, 226584, 226563, 226564, 226565,
    226567, 226557, 226558, 227488, 227489
)
GROUP BY stay_id, charttime
""").df()

# === 3️⃣ 合併 Input 與 Output ===
print("🔄 合併 Input/Output 並計算 Balance...")

input_df["charttime"] = pd.to_datetime(input_df["charttime"], errors="coerce")
output_df["charttime"] = pd.to_datetime(output_df["charttime"], errors="coerce")

io_df = pd.merge(input_df, output_df, on=["stay_id", "charttime"], how="outer")

# 注意：若你希望「沒有記錄」代表 0（例如沒有輸入事件），用 0；若希望未知則用 NaN
# 你原本使用 0，我保留相同行為
io_df["input_ml"] = io_df["input_ml"].fillna(0)
io_df["urineoutput"] = io_df["urineoutput"].fillna(0)
io_df["io_balance"] = io_df["input_ml"] - io_df["urineoutput"]

print(f"✅ I/O 原始事件時間點共 {len(io_df):,} 筆")

# === 4️⃣ 讀取拔管時間 ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ 拔管名單共 {len(extub):,} 筆")

# === 5️⃣ 主迴圈: 每位病人拔管前 [52h, 4h) ===
records = []
print("⏳ 開始進行時間窗格切割 (-52h to -4h, gap=4h, 12 bins)...")

for _, row in tqdm(extub.iterrows(), total=len(extub), desc="💧 Processing patients"):
    sid = row["subject_id"]
    stay = row["stay_id"]
    extub_time = row["extubation_time"]

    # 觀測窗: [extub-52h, extub-4h)（嚴格右開）
    start_time = extub_time - pd.Timedelta(hours=(WINDOW_H + GAP_H))  # 52h
    end_time = extub_time - pd.Timedelta(hours=GAP_H)                # 4h

    # ✅ 嚴格：charttime < end_time（不含等於 end_time）
    sub = io_df[
        (io_df["stay_id"] == stay)
        & (io_df["charttime"] >= start_time)
        & (io_df["charttime"] < end_time)
    ].copy()

    if sub.empty:
        # 完全無資料：補齊 12 bins（全 NaN）
        empty_df = pd.DataFrame({"time_bin": expected_bins})
        empty_df["input_ml"] = np.nan
        empty_df["urineoutput"] = np.nan
        empty_df["io_balance"] = np.nan
        empty_df["subject_id"] = sid
        empty_df["stay_id"] = stay
        records.append(empty_df)
        continue

    # === 相對時間與分段 ===
    sub["hours_from_extub"] = (sub["charttime"] - extub_time).dt.total_seconds() / 3600.0

    # 防呆：嚴格不允許 >= -4h
    if (sub["hours_from_extub"] >= -GAP_H).any():
        raise ValueError(f"Leakage detected (>= -{GAP_H}h): subject_id={sid}, stay_id={stay}")

    # 選擇 A：floor 分箱（time_bin=左端點）
    sub["time_bin"] = (np.floor(sub["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

    # === 每 4 小時平均 ===
    agg = (
        sub.groupby("time_bin")[["input_ml", "urineoutput", "io_balance"]]
           .mean()
           .reindex(expected_bins, fill_value=np.nan)
           .reset_index()  # time_bin
    )

    agg["subject_id"] = sid
    agg["stay_id"] = stay
    records.append(agg)

# === 6️⃣ 合併結果並加入標籤 ===
if len(records) == 0:
    print("⚠️ 無資料可用")
else:
    all_df = pd.concat(records, ignore_index=True)

    all_df = all_df.merge(
        extub[["subject_id", "stay_id", "Extubation_failure"]],
        on=["subject_id", "stay_id"],
        how="left"
    )

    print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
    print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

    # 檢查是否每位病人都有 12 bins
    cnt = all_df.groupby(["subject_id", "stay_id"]).size()
    bad = cnt[cnt != len(expected_bins)]
    if len(bad) > 0:
        print(f"⚠️ 有 {len(bad)} 位病人 time_bin 筆數不是 12（可能是 merge 或資料異常）")
        print(bad.head(10))

    # === 儲存 ===
    all_df.to_csv(output_path, index=False)
    print(f"📁 已輸出至: {output_path}")

    print("\n🔍 前 10 筆資料：")
    print(all_df[["subject_id", "stay_id", "time_bin", "io_balance", "Extubation_failure"]].head(10))
