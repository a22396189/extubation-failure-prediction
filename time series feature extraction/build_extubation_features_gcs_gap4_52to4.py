# =============================================================
# build_extubation_features_gcs_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的格拉斯哥昏迷指數（GCS），
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - chartevents.parquet (MIMIC-IV icu)：含三個 GCS 子項 itemid：
#       223900 = GCS Verbal（插管無法言語者賦值 0）
#       223901 = GCS Motor
#       220739 = GCS Eyes
#
# 【GCS 計算方式】
#   GCS = Eyes + Verbal + Motor（三子項以 stay_id + charttime 對齊後加總）
#   - 插管病人（No Response-ETT）：Verbal 賦值 0（非 NaN）
#   - 三子項以 outer join 合併，min_count=1 確保至少有一項才加總
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 無資料的 bin 補 NaN，每位病人固定輸出 12 列
#
# 【特徵欄位】
#   GCS（共 1 個連續變數，範圍 3–15）
#
# 【輸出】
#   - extubation_features_gcs_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：從 chartevents 讀取三個 GCS 子項
#   Step 2：分離 Motor / Eyes / Verbal，處理插管無法言語特殊值
#   Step 3：以 stay_id + charttime outer join 合併三子項，加總為 GCS
#   Step 4：逐人篩選觀測窗，floor 分箱，每 bin 平均，補齊 12 bins
#   Step 5：合併 Extubation_failure 標籤，輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_gcs_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）、每 4 小時平均的 GCS 特徵（共12段）
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

# === 路徑設定 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
duckdb_path = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"
chartevents_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\icu_parquet\chartevents.parquet"
extub_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_outcome.csv"

output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_gcs_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8

# === 1️⃣ 連線與讀取 GCS 相關欄位 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 chartevents (GCS 欄位)...")

gcs_df = con.execute(f"""
SELECT
    subject_id,
    stay_id,
    charttime,
    itemid,
    valuenum,
    value
FROM read_parquet('{chartevents_path}')
WHERE itemid IN (223900, 223901, 220739)  -- verbal, motor, eyes
  AND (valuenum IS NOT NULL OR value IS NOT NULL)
""").df()

gcs_df["charttime"] = pd.to_datetime(gcs_df["charttime"], errors="coerce")

# === 2️⃣ 各子項整理 ===
gcsmotor = (
    gcs_df[gcs_df["itemid"] == 223901][["stay_id", "charttime", "valuenum"]]
    .rename(columns={"valuenum": "gcsmotor"})
    .copy()
)

gcseyes = (
    gcs_df[gcs_df["itemid"] == 220739][["stay_id", "charttime", "valuenum"]]
    .rename(columns={"valuenum": "gcseyes"})
    .copy()
)

gcsverbal = gcs_df[gcs_df["itemid"] == 223900][["stay_id", "charttime", "valuenum", "value"]].copy()

# 處理插管無法言語的狀況 (No Response-ETT)
gcsverbal.loc[gcsverbal["value"] == "No Response-ETT", "valuenum"] = 0
gcsverbal = gcsverbal[["stay_id", "charttime", "valuenum"]].rename(columns={"valuenum": "gcsverbal"})

# === 3️⃣ 合併三個子項（以 stay_id+charttime 為鍵） ===
merged = gcsmotor.merge(gcseyes, on=["stay_id", "charttime"], how="outer")
merged = merged.merge(gcsverbal, on=["stay_id", "charttime"], how="outer")

# GCS = 眼 + 言語 + 動作
merged["gcs"] = merged[["gcsmotor", "gcsverbal", "gcseyes"]].sum(axis=1, min_count=1)

# === 4️⃣ 讀取拔管時間與標籤 ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
print(f"✅ 共 {len(extub):,} 位病人拔管紀錄")

records = []
print("🔄 開始計算時間區間平均 (GCS)...")

for _, row in tqdm(extub.iterrows(), total=len(extub), desc="🧠 Processing patients"):
    sid = row["subject_id"]
    stay = row["stay_id"]
    extub_time = row["extubation_time"]
    failure_label = row["Extubation_failure"]

    # 觀測窗: [extub-52h, extub-4h)  (嚴格右開)
    start_window = extub_time - pd.Timedelta(hours=(WINDOW_H + GAP_H))  # 52h
    end_window = extub_time - pd.Timedelta(hours=GAP_H)                # 4h

    sub = merged[merged["stay_id"] == stay].copy()

    if sub.empty:
        sub_df = pd.DataFrame({"time_bin": expected_bins, "GCS": np.nan})
    else:
        sub["hours_from_extub"] = (sub["charttime"] - extub_time).dt.total_seconds() / 3600.0

        # ✅ 嚴格篩選：-52 <= h < -4
        sub = sub[
            (sub["hours_from_extub"] >= -(WINDOW_H + GAP_H)) &
            (sub["hours_from_extub"] < -GAP_H)
        ].copy()

        # 防呆：嚴格不允許任何 >= -4h
        if (sub["hours_from_extub"] >= -GAP_H).any():
            raise ValueError(f"Leakage detected (>= -{GAP_H}h): subject_id={sid}, stay_id={stay}")

        # 選擇 A：floor 分箱（左端點）
        sub["time_bin"] = (np.floor(sub["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

        sub_df = (
            sub.groupby("time_bin")["gcs"]
               .mean()
               .reindex(expected_bins, fill_value=np.nan)
               .reset_index()
               .rename(columns={"gcs": "GCS"})
        )

    sub_df["subject_id"] = sid
    sub_df["stay_id"] = stay
    sub_df["Extubation_failure"] = failure_label
    records.append(sub_df)

# === 5️⃣ 合併所有病人 ===
all_df = pd.concat(records, ignore_index=True)
print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

# === 6️⃣ 輸出 ===
all_df.to_csv(output_path, index=False)
print(f"📁 Saved to: {output_path}")

print("\n🔍 前 10 筆資料：")
print(all_df[["subject_id", "stay_id", "time_bin", "GCS", "Extubation_failure"]].head(10))
