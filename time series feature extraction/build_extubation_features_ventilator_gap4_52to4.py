# =============================================================
# build_extubation_features_ventilator_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的呼吸器設定參數，
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值；
#   並從 Cohort 的 MV_days 回推各時間點的「累積機械通氣天數（MV_day）」，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - ventilator_setting.parquet (MIMIC-IV derived)：呼吸器設定，
#     含 subject_id / stay_id / charttime / fio2 / mean_airway_pressure / peep / tidal_volume_set
#   - mv_day_unique_subject_filtered.csv：Cohort 篩選後的 MV_days（拔管當下的總機械通氣天數）
#     ⚠️ 不重新計算 MV_days，直接引用此欄位以確保與 Cohort 一致
#
# 【MV_day 回推邏輯】
#   MV_day（某時間點）= MV_days_at_extubation − (−time_bin / 24)
#   即以拔管當下為基準，往前推算每個 time_bin 對應的累積 MV 天數
#   ⚠️ clip(lower=0) 避免極早期 bin 出現負值
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 無資料的 bin 補 NaN，每位病人固定輸出 12 列
#
# 【特徵欄位】
#   FiO2, MAP（Mean Airway Pressure）, PEEP, Tidal_Volume, MV_day
#   （共 5 個連續變數）
#   ⚠️ MAP 此處代表 Mean Airway Pressure，非 Mean Arterial Pressure（後者為 mbp）
#
# 【輸出】
#   - extubation_features_ventilator_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：讀取 ventilator_setting.parquet
#   Step 2：合併拔管時間，計算相對時間，嚴格篩選觀測窗
#   Step 3：floor 分箱，補齊 12 bins，每 bin 平均
#   Step 4：從 Cohort 引入 MV_days，回推各 time_bin 的累積 MV_day
#   Step 5：合併 Extubation_failure 標籤，輸出 CSV 並驗證 MV_day 遞增性
# =============================================================

# ============================================================
# build_extubation_features_ventilator_gap4_52to4.py
# 功能:
#   Extract ventilator-related features for extubation task
#   - Observation window: [t-52h, t-4h)  (strict gap=4h, left-closed right-open)
#   - 4h bins, 12 bins: -52, -48, ..., -8  (Option A: time_bin is left edge)
#   - MV_day is derived from Cohort MV_days_at_extubation (NO recomputation)
# ============================================================

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 路徑設定
# ============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
duckdb_path = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"
ventset_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\derived\ventilator_setting.parquet"
extub_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_outcome.csv"
cohort_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\mv_day_unique_subject_filtered.csv"

output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_ventilator_gap4_52to4.csv"

# ============================================================
# 參數
# ============================================================
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8

# ============================================================
# 連線 DuckDB
# ============================================================
con = duckdb.connect(duckdb_path)

# ============================================================
# 1️⃣ 讀取 ventilator_setting.parquet
# ============================================================
print("🚀 讀取 ventilator_setting.parquet ...")
vset = con.execute(f"SELECT * FROM read_parquet('{ventset_path}')").df()
print(f"✅ ventilator_setting rows: {len(vset):,}, cols: {list(vset.columns)}")

# ============================================================
# 2️⃣ 讀取 extubation outcome 並 merge extubation_time
# ============================================================
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ extub rows: {len(extub):,}")

vset = vset.merge(
    extub[["subject_id", "stay_id", "extubation_time"]],
    on=["subject_id", "stay_id"],
    how="inner"
)

# ============================================================
# 3️⃣ 計算距離拔管時間（小時）並嚴格篩選 [-52, -4)
# ============================================================
vset["charttime"] = pd.to_datetime(vset["charttime"], errors="coerce")
vset["hours_from_extub"] = (vset["charttime"] - vset["extubation_time"]).dt.total_seconds() / 3600.0

# ✅ 嚴格觀測窗：-52 <= h < -4
vset = vset[
    (vset["hours_from_extub"] >= -(WINDOW_H + GAP_H)) &
    (vset["hours_from_extub"] < -GAP_H)
].copy()

# 防呆：不允許任何 >= -4
if (vset["hours_from_extub"] >= -GAP_H).any():
    raise ValueError("Leakage detected: found hours_from_extub >= -4h after filtering")

# ✅ Option A：time_bin=左端點（floor 分箱）
vset["time_bin"] = (np.floor(vset["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

# ============================================================
# 4️⃣ 定義 ventilator 特徵
# ============================================================
features = {
    "fio2": "FiO2",
    "mean_airway_pressure": "MAP",
    "peep": "PEEP",
    "tidal_volume_set": "Tidal_Volume"
}

for col in features.keys():
    if col not in vset.columns:
        vset[col] = np.nan
        print(f"⚠️ 缺少欄位 {col}，自動補 NaN")

# ============================================================
# 5️⃣ 補齊完整時間序列 (-52..-8)
# ============================================================
print("🔄 補齊時間序列缺值 (-52h to -4h, gap=4h, 12 bins)...")

records = []
all_pairs = extub[["subject_id", "stay_id"]].drop_duplicates()

for _, r in all_pairs.iterrows():
    sid, stay = r["subject_id"], r["stay_id"]
    subdf = vset[(vset["subject_id"] == sid) & (vset["stay_id"] == stay)]

    if not subdf.empty:
        subagg = (
            subdf.groupby("time_bin")[list(features.keys())]
            .mean()
            .reindex(expected_bins, fill_value=np.nan)
            .reset_index()
        )
    else:
        subagg = pd.DataFrame({"time_bin": expected_bins})
        for c in features.keys():
            subagg[c] = np.nan

    subagg["subject_id"] = sid
    subagg["stay_id"] = stay
    records.append(subagg)

all_df = pd.concat(records, ignore_index=True)
all_df.rename(columns=features, inplace=True)

print(f"✅ 完成 ventilator 特徵彙整，共 {len(all_df):,} 筆資料")

# ============================================================
# 6️⃣ 直接引用 Cohort 的 MV_days（拔管當下的總 MV_days）並回推 MV_day
# ============================================================
print("📘 直接引用 Cohort 的 MV_days，回推各時間點累積 MV 天數 ...")

cohort_df = pd.read_csv(cohort_path)

# Cohort 提供的是「拔管當下的總 MV_days（唯一可信）」→ 改名避免混淆
cohort_mv = cohort_df[["stay_id", "MV_days"]].copy()
cohort_mv.rename(columns={"MV_days": "MV_days_at_extubation"}, inplace=True)

# merge 到時間序列表
all_df = all_df.merge(cohort_mv, on="stay_id", how="left")

# time_bin 為負（小時），表示距離拔管還有 -time_bin 小時
# 回推該時間點累積 MV 天數（以拔管當下為基準往回減）
all_df["MV_day"] = all_df["MV_days_at_extubation"] - (-all_df["time_bin"] / 24.0)

# 保護性處理：避免負值
all_df["MV_day"] = all_df["MV_day"].clip(lower=0)

# 移除中間欄位
all_df.drop(columns=["MV_days_at_extubation"], inplace=True)

# ============================================================
# 7️⃣ 合併拔管結果
# ============================================================
all_df = all_df.merge(
    extub[["subject_id", "stay_id", "Extubation_failure"]],
    on=["subject_id", "stay_id"],
    how="inner"
)

# ============================================================
# 8️⃣ 輸出結果
# ============================================================
all_df.to_csv(output_path, index=False)

print(f"✅ Ventilator features generated successfully: {len(all_df):,} rows")
print(f"📁 Saved to: {output_path}")
print(f"✅ bins = {list(expected_bins)}")

# ============================================================
# 9️⃣ 驗證
# ============================================================
print("🔍 驗證 MV_day 是否隨 time_bin 推進而遞增（越接近拔管 MV_day 越大）：")
print(
    all_df.sort_values(["stay_id", "time_bin"])
         [["stay_id", "time_bin", "MV_day"]]
         .head(12)
)
print(f"🔍 MV_day 最小值: {all_df['MV_day'].min()} (應 ≥ 0)")

print("\n🔍 前 10 筆資料預覽：")
print(
    all_df[["subject_id", "stay_id", "time_bin", "FiO2", "MAP", "PEEP", "Tidal_Volume", "MV_day", "Extubation_failure"]]
    .head(10)
)
