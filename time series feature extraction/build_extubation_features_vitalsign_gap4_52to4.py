# =============================================================
# build_extubation_features_vitalsign_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的生理徵象，
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 extubation_time / Extubation_failure
#   - vitalsign.parquet (MIMIC-IV derived)：含 heart_rate / sbp / dbp / mbp /
#     resp_rate / temperature / spo2 / glucose，以 stay_id + charttime 為索引
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 無資料的 bin 補 NaN，每位病人固定輸出 12 列
#
# 【特徵欄位】
#   heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2, glucose
#   （共 8 個連續變數）
#
# 【輸出】
#   - extubation_features_vitalsign_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：讀取 vitalsign.parquet 與拔管名單
#   Step 2：逐人以 DuckDB SQL 擷取觀測窗內資料
#   Step 3：floor 分箱、每 bin 平均，補齊 12 bins
#   Step 4：合併 Extubation_failure 標籤，輸出 CSV
# =============================================================

# ============================================================
# File: build_extubation_features_vitalsign_gap4_52to4.py
# 功能: 建立拔管前 52–4 小時、每 4 小時平均的生理特徵（固定 12 區段）
#      ✅ 嚴格 gap=4h：完全不含 [extub_time-4h, extub_time) 之間的資料
#
# 設計選擇 A（推薦）：
#   - time_bin 代表「區間左端點」（用 floor 分箱）
#   - 12 個 bins：-52, -48, ..., -8
#     對應區間：
#       -52 -> [-52, -48)
#       -48 -> [-48, -44)
#       ...
#        -8 -> [ -8,  -4)   ← 最後一段（仍符合 gap=4h，未使用 [-4,0)）
#
# 觀測窗（leakage-safe）：
#   - 使用 [t-52h, t-4h)（左閉右開）
# ============================================================

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
vital_path = Path(rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\vitalsign.parquet")
extub_path = Path(rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv")

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_vitalsign_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h = 48h (不含 gap)

# ✅ 固定 12 bins：-52, -48, ..., -8（最後一段代表 [-8,-4)）
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)
# 等價於：np.arange(-52, -8 + 1, 4)

num_cols = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose"]

# === 連線至 DuckDB 並讀取資料 ===
con = duckdb.connect(duckdb_path)

print("🚀 載入 vitalsign parquet...")
vital = con.execute(f"SELECT * FROM read_parquet('{vital_path}')").df()
print(f"✅ 共 {len(vital):,} 筆記錄")
print(f"✅ vitalsign 欄位：{list(vital.columns)}")

print("🚀 讀取拔管結果 extubation_outcome.csv ...")
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ 載入拔管結果，共 {len(extub):,} 位病人")
print(f"✅ extub 欄位：{list(extub.columns)}")

# 註冊 vital 表供 SQL 使用
con.register("vital", vital)

records = []
print("🔄 開始處理每位病人的 Vital Signs...")

for _, row in tqdm(extub.iterrows(), total=len(extub), desc="🩺 Processing patients"):
    sid = row["subject_id"]
    stay = row["stay_id"]
    extub_time = row["extubation_time"]

    # 目標區間： [extub-52h, extub-4h)  (嚴格右開，排除 >= -4h 的資料)
    start_time = extub_time - pd.Timedelta(hours=(WINDOW_H + GAP_H))  # extub - 52h
    end_time = extub_time - pd.Timedelta(hours=GAP_H)                # extub - 4h

    # ✅ 嚴格 gap：charttime < end_time（不含等於 end_time）
    query = f"""
        SELECT charttime, {', '.join(num_cols)}
        FROM vital
        WHERE subject_id = {sid}
          AND stay_id = {stay}
          AND charttime >= TIMESTAMP '{start_time}'
          AND charttime <  TIMESTAMP '{end_time}'
    """
    df = con.execute(query).df()

    if df.empty:
        # 即使無資料，也要建立空白 12 bins
        empty_df = pd.DataFrame({"time_bin": expected_bins})
        empty_df[num_cols] = np.nan
        empty_df["subject_id"] = sid
        empty_df["stay_id"] = stay
        records.append(empty_df)
        continue

    # 計算相對拔管時間（負值代表拔管前）
    df["hours_from_extub"] = (df["charttime"] - extub_time).dt.total_seconds() / 3600.0

    # 防呆檢查：嚴格不允許任何 >= -4h 的點
    if (df["hours_from_extub"] >= -GAP_H).any():
        raise ValueError(f"Leakage detected (>= -{GAP_H}h): subject_id={sid}, stay_id={stay}")

    # === 選擇 A：floor 分箱（time_bin 是區間左端點）===
    # 例如：
    #   -51.9 -> -52  表示落在 [-52,-48)
    #    -8.1 -> -12  表示落在 [-12,-8)
    #    -4.0001 -> -8 表示落在 [-8,-4)（但 >=-4h 已被 SQL 排除）
    df["time_bin"] = (np.floor(df["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

    # 聚合：每個 time_bin 的 4 小時平均
    agg_df = (
        df.groupby("time_bin")[num_cols]
          .mean()
          .reindex(expected_bins, fill_value=np.nan)  # 補齊 12 bins
          .reset_index()  # 產生 time_bin 欄位
          .assign(subject_id=sid, stay_id=stay)
    )
    records.append(agg_df)

# === 合併所有病人資料 ===
all_df = pd.concat(records, ignore_index=True)

# 合併拔管失敗標籤（stay-level）
print("🔗 合併拔管失敗標籤 Extubation_failure ...")
all_df = all_df.merge(
    extub[["subject_id", "stay_id", "Extubation_failure"]],
    on=["subject_id", "stay_id"],
    how="inner"
)

print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

# === 檢查每位病人是否都有 12 bins ===
cnt = all_df.groupby(["subject_id", "stay_id"]).size()
bad = cnt[cnt != len(expected_bins)]
if len(bad) > 0:
    print(f"⚠️ 有 {len(bad)} 位病人 time_bin 筆數不是 12（可能是 extub merge 或資料異常）")
    print(bad.head(10))

# === 儲存 CSV ===
all_df.to_csv(output_path, index=False)
print(f"📁 已輸出至: {output_path}")

# 預覽
print(all_df[["subject_id", "stay_id", "time_bin", "heart_rate", "Extubation_failure"]].head(15))
