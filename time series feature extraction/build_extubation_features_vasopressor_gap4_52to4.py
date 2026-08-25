# =============================================================
# build_extubation_features_vasopressor_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）內，
#   每個 4 小時區段是否使用升壓劑（Vasopressor），
#   輸出二元旗標（0/1），作為時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - vasoactive_agent.parquet (MIMIC-IV derived)：升壓劑紀錄，
#     含 stay_id / starttime / endtime（連續使用區間格式）
#
# 【特徵計算方式（Overlap Logic）】
#   每個 time_bin 對應區間 [bin_start, bin_end)，
#   判斷升壓劑是否與該區間重疊：
#     Vasopressor_use = 1  iff  starttime < bin_end  AND  endtime > bin_start
#   - 無任何升壓劑紀錄的 stay → 12 個 bin 皆為 0
#   ⚠️ 無紀錄以 0 填補（非 NaN），代表該段時間確實未使用升壓劑
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 每位病人固定輸出 12 列
#
# 【特徵欄位】
#   Vasopressor_use（0/1 二元旗標，共 1 個變數）
#
# 【輸出】
#   - extubation_features_vasopressor_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：讀取 vasoactive_agent.parquet 與拔管名單
#   Step 2：逐人、逐 bin 以 overlap 條件判斷是否使用升壓劑
#   Step 3：建立完整 12 bins 結果（無紀錄預設 0）
#   Step 4：輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_vasopressor_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）內升壓劑使用情況（每 4 小時區段 Yes/No，共12段）
#
# ✅ gap=4h（嚴格）：完全不含 [extub_time-4h, extub_time) 的資料（含剛好 -4h 也排除）
# ✅ 選擇 A：time_bin 代表「區間左端點」
#    12 bins: -52, -48, ..., -8
#    對應區間: [-52,-48), [-48,-44), ..., [-8,-4)
#
# 用藥重疊判定（Overlap logic）：
#   使用 = 1  iff  starttime < bin_end AND endtime > bin_start
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
vaso_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\vasoactive_agent.parquet"
extub_path = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_vasopressor_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h
time_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8

# === 1️⃣ 讀取資料 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 vasoactive_agent.parquet ...")
vaso = con.execute(f"SELECT * FROM read_parquet('{vaso_path}')").df()

vaso["starttime"] = pd.to_datetime(vaso["starttime"], errors="coerce")
vaso["endtime"] = pd.to_datetime(vaso["endtime"], errors="coerce")

print(f"✅ 共 {len(vaso):,} 筆 vasoactive 紀錄，欄位：{list(vaso.columns)}")

# === 2️⃣ 讀取拔管 outcome ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ 共 {len(extub):,} 位病人拔管紀錄")

records = []
print("🔄 開始計算時間區間特徵 (vasopressor use, gap4_52to4)...")

for _, row in tqdm(extub.iterrows(), total=len(extub), desc="💊 Processing patients"):
    stay = row["stay_id"]
    sid = row["subject_id"]
    extub_time = row["extubation_time"]
    failure_label = row["Extubation_failure"]

    # 該 stay 的用藥紀錄
    sub = vaso[vaso["stay_id"] == stay].copy()

    # 若完全無升壓劑資料 → 12 段皆為 0
    if sub.empty:
        for tb in time_bins:
            records.append({
                "subject_id": sid,
                "stay_id": stay,
                "time_bin": tb,
                "Vasopressor_use": 0,
                "Extubation_failure": failure_label
            })
        continue

    # 逐 bin 判斷是否重疊
    for tb in time_bins:
        bin_start = extub_time + pd.Timedelta(hours=int(tb))
        bin_end = bin_start + pd.Timedelta(hours=BIN_H)

        # ✅ 重疊條件：start < bin_end 且 end > bin_start
        overlap = sub[(sub["starttime"] < bin_end) & (sub["endtime"] > bin_start)]
        used = 1 if not overlap.empty else 0

        records.append({
            "subject_id": sid,
            "stay_id": stay,
            "time_bin": int(tb),
            "Vasopressor_use": used,
            "Extubation_failure": failure_label
        })

# === 3️⃣ 匯出結果 ===
df_out = pd.DataFrame(records)
df_out.sort_values(["stay_id", "time_bin"], inplace=True)
df_out.to_csv(output_path, index=False)

print(f"✅ Vasopressor time-bin features generated successfully: {len(df_out):,} rows")
print(f"📁 Saved to: {output_path}")
print(f"✅ bins = {list(time_bins)}")

print("\n🔍 前 12 筆資料預覽：")
print(df_out[["subject_id", "stay_id", "time_bin", "Vasopressor_use", "Extubation_failure"]].head(12))
