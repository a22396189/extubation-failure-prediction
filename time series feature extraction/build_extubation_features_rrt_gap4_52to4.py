# =============================================================
# build_extubation_features_rrt_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）內，
#   每個 4 小時區段是否接受腎臟替代療法（RRT / 血液透析），
#   輸出二元旗標（0/1），作為時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - rrt.parquet (MIMIC-IV derived)：RRT 紀錄，含 stay_id / charttime / dialysis_active
#
# 【特徵計算方式】
#   - 每個 time_bin 內，若任一筆 dialysis_active == 1 → Hemodialysis_use = 1
#   - 若該 stay_id 在觀測窗內完全無 RRT 紀錄 → 補 0（非 NaN）
#   ⚠️ 無紀錄以 0 填補（非 NaN），代表該段時間確實未接受透析
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 每位病人固定輸出 12 列
#
# 【特徵欄位】
#   Hemodialysis_use（0/1 二元旗標，共 1 個變數）
#
# 【輸出】
#   - extubation_features_rrt_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：讀取 rrt.parquet 與拔管名單
#   Step 2：逐人篩選觀測窗，floor 分箱
#   Step 3：每 bin 取 max（任一筆 active → 1），補齊 12 bins（預設 0）
#   Step 4：輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_rrt_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）、每 4 小時區間內是否接受血液透析 (Yes/No)（共12段）
#
# ✅ gap=4h（嚴格）：完全不含 [extub_time-4h, extub_time) 的資料（含剛好 -4h 也排除）
# ✅ 選擇 A：time_bin 代表「區間左端點」（floor 分箱）
#    12 bins: -52, -48, ..., -8
#    對應區間: [-52,-48), [-48,-44), ..., [-8,-4)
#
# 事件聚合策略：
#   - 每個 time_bin 內只要任一筆 dialysis_active==1 => Hemodialysis_use=1（用 max 聚合）
#   - 若該病人/區段完全無紀錄 => 補 0
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
rrt_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\rrt.parquet"
extub_path = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_rrt_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8

# === 1️⃣ 讀取血液透析資料 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 rrt.parquet ...")
rrt = con.execute(f"SELECT * FROM read_parquet('{rrt_path}')").df()
rrt["charttime"] = pd.to_datetime(rrt["charttime"], errors="coerce")
print(f"✅ 共 {len(rrt):,} 筆 RRT 紀錄，欄位：{list(rrt.columns)}")

# === 2️⃣ 讀取拔管 outcome ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ 共 {len(extub):,} 位病人拔管紀錄")

records = []
print("🔄 開始處理每位病人的 RRT 狀態 (gap4_52to4)...")

# === 3️⃣ 主迴圈：每位病人生成 12 段時間區間 ===
for _, row in tqdm(extub.iterrows(), total=len(extub), desc="🩸 Processing patients"):
    sid = row["subject_id"]
    stay = row["stay_id"]
    extub_time = row["extubation_time"]
    failure_label = row["Extubation_failure"]

    # ✅ 嚴格觀測窗: [extub-52h, extub-4h) => -52 <= h < -4
    sub = rrt[rrt["stay_id"] == stay].copy()

    if sub.empty:
        sub_df = pd.DataFrame({"time_bin": expected_bins, "Hemodialysis_use": 0})
    else:
        sub["hours_from_extub"] = (sub["charttime"] - extub_time).dt.total_seconds() / 3600.0

        # 嚴格篩選：-52 <= h < -4
        sub = sub[
            (sub["hours_from_extub"] >= -(WINDOW_H + GAP_H)) &
            (sub["hours_from_extub"] < -GAP_H)
        ].copy()

        if sub.empty:
            sub_df = pd.DataFrame({"time_bin": expected_bins, "Hemodialysis_use": 0})
        else:
            # 防呆：不允許任何 >= -4h
            if (sub["hours_from_extub"] >= -GAP_H).any():
                raise ValueError(f"Leakage detected (>= -{GAP_H}h): subject_id={sid}, stay_id={stay}")

            # 只要 dialysis_active==1 就算使用
            sub["Hemodialysis_use"] = (sub["dialysis_active"] == 1).astype(int)

            # 選擇 A：floor 分箱（左端點）
            sub["time_bin"] = (np.floor(sub["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

            # 每 4 小時聚合：任一筆=1 => 1
            sub_df = (
                sub.groupby("time_bin")["Hemodialysis_use"]
                   .max()
                   .reindex(expected_bins, fill_value=0)
                   .reset_index()
            )

    sub_df["subject_id"] = sid
    sub_df["stay_id"] = stay
    sub_df["Extubation_failure"] = failure_label
    records.append(sub_df)

# === 4️⃣ 合併所有病人資料 ===
all_df = pd.concat(records, ignore_index=True)
print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

# === 5️⃣ 輸出結果 ===
all_df.to_csv(output_path, index=False)
print(f"📁 Saved to: {output_path}")

print("\n🔍 前 10 筆資料：")
print(all_df[["subject_id", "stay_id", "time_bin", "Hemodialysis_use", "Extubation_failure"]].head(10))
