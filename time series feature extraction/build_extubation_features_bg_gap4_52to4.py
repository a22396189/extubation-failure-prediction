# =============================================================
# build_extubation_features_bg_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的動脈血氣分析數值，
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - bg.parquet (MIMIC-IV derived)：動脈血氣分析，以 subject_id + charttime 為索引
#     （若缺 stay_id 欄位，透過 stay_subject_map.csv 補入）
#   - stay_subject_map.csv：subject_id → stay_id 對應表（本 Cohort 為 1對1）
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 無資料的 bin 補 NaN，每位病人固定輸出 12 列
#
# 【特徵欄位】
#   pH, PaO2, PaCO2, BE（Base Excess）, PaO2_FiO2_Ratio
#   （共 5 個連續變數）
#
# 【輸出】
#   - extubation_features_bg_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：讀取 bg.parquet 與拔管名單
#   Step 2：若 bg 缺 stay_id，透過 stay_subject_map 補入
#   Step 3：合併拔管時間，計算相對時間，嚴格篩選觀測窗
#   Step 4：floor 分箱、每 bin 平均
#   Step 5：補齊所有病人的完整 12 bins，輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_bg_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）、每 4 小時平均的血氣特徵（共12段）
#   ✅ gap=4h：完全不含 [extub_time-4h, extub_time) 的資料（含剛好 -4h 也排除）
#   ✅ 選擇 A：time_bin 代表「區間左端點」（floor 分箱）
#      12 bins: -52, -48, ..., -8
#      對應區間: [-52,-48), [-48,-44), ..., [-8,-4)
#   若某時間段無資料，補齊 NaN
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
bg_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\derived\bg.parquet"
extub_path = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"
stay_map_path = rf"{EXTUBATION_ROOT}\data\outputs\stay_subject_map.csv"

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_bg_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h

# ✅ 12 bins（左端點）：-52, -48, ..., -8
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8 step4

# === 1️⃣ 讀取資料 ===
con = duckdb.connect(duckdb_path)
print("🚀 讀取 bg.parquet ...")
bg = con.execute(f"SELECT * FROM read_parquet('{bg_path}')").df()
print(f"✅ 共 {len(bg):,} 筆資料，欄位：{list(bg.columns)}")

# === 2️⃣ 讀取拔管 outcome 與 stay 對應表 ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
stay_map = pd.read_csv(stay_map_path)

# === 3️⃣ 若血氣資料缺 stay_id，依 subject_id 對應 ===
# ⚠️ 注意：subject_id → stay_id 可能一對多，若你的 stay_map 不是唯一對應，會造成重複列。
# 若你確認每個 subject_id 在研究期間只對應一個 stay_id 才安全。
if "stay_id" not in bg.columns:
    print("🧩 bg 缺 stay_id，將血氣資料對應到 stay_id ...")
    bg = bg.merge(stay_map, on="subject_id", how="left")

# === 4️⃣ 合併拔管時間與 outcome（stay-level）===
bg = bg.merge(
    extub[["stay_id", "extubation_time", "Extubation_failure"]],
    on="stay_id",
    how="inner"
)

# === 5️⃣ 時間處理與嚴格篩選區間 ===
bg["charttime"] = pd.to_datetime(bg["charttime"], errors="coerce")
bg["hours_from_extub"] = (bg["charttime"] - bg["extubation_time"]).dt.total_seconds() / 3600.0

# ✅ 嚴格觀測窗: [-52h, -4h)  =>  -52 <= h < -4
bg = bg[(bg["hours_from_extub"] >= -(WINDOW_H + GAP_H)) & (bg["hours_from_extub"] < -GAP_H)]

# 防呆：理論上不該出現 >= -4h
if (bg["hours_from_extub"] >= -GAP_H).any():
    raise ValueError("Leakage detected: found hours_from_extub >= -4h after filtering")

# === 6️⃣ 定義特徵欄位（bg.parquet 欄位名 → 輸出欄位名） ===
features = {
    "ph": "pH",
    "pao2": "PaO2",
    "paco2": "PaCO2",
    "baseexcess": "BE",
    # 注意：有些版本可能叫 pao2fio2ratio / pao2_fio2_ratio / pafi / pf_ratio
    "pao2_fio2_ratio": "PaO2_FiO2_Ratio"
}

# 若欄位不存在，補 NaN
for c in features.keys():
    if c not in bg.columns:
        bg[c] = np.nan
        print(f"⚠️ 欄位缺失：{c}，自動補上 NaN")

# === 7️⃣ time_bin（選擇 A：左端點 floor 分箱）===
# 例如：
#   -51.9 -> -52  ([-52,-48))
#   -8.1  -> -12  ([-12,-8))
#   -4.0001 -> -8 ([-8,-4))  (但 >=-4 已被排除)
bg["time_bin"] = (np.floor(bg["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

# === 8️⃣ 每 4 小時平均（stay-level）===
agg_df = (
    bg.groupby(["subject_id", "stay_id", "time_bin"], as_index=False)[list(features.keys())]
      .mean()
      .rename(columns=features)
)

# === 9️⃣ 補齊完整 12 bins（-52..-8）===
records = []

all_pairs = extub[["subject_id", "stay_id", "Extubation_failure"]].drop_duplicates()
print(f"🔄 開始補齊時間序列 (共 {len(all_pairs)} 位病人)...")

for _, r in tqdm(all_pairs.iterrows(), total=len(all_pairs), desc="🧪 Padding bins"):
    sid = r["subject_id"]
    stay = r["stay_id"]
    failure_val = r["Extubation_failure"]

    subdf = agg_df[(agg_df["subject_id"] == sid) & (agg_df["stay_id"] == stay)]

    if not subdf.empty:
        subagg = (
            subdf.set_index("time_bin")
                 .reindex(expected_bins, fill_value=np.nan)
                 .reset_index()
        )
    else:
        subagg = pd.DataFrame({"time_bin": expected_bins})
        for col in features.values():
            subagg[col] = np.nan

    subagg["subject_id"] = sid
    subagg["stay_id"] = stay
    subagg["Extubation_failure"] = failure_val
    records.append(subagg)

all_df = pd.concat(records, ignore_index=True)

print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

# === 🔟 檢查每位病人是否都有 12 bins ===
cnt = all_df.groupby(["subject_id", "stay_id"]).size()
bad = cnt[cnt != len(expected_bins)]
if len(bad) > 0:
    print(f"⚠️ 有 {len(bad)} 位病人 time_bin 筆數不是 12（可能是 merge/重複 stay_id 導致）")
    print(bad.head(10))

# === 11️⃣ 輸出結果 ===
all_df.to_csv(output_path, index=False)
print(f"📁 已輸出至: {output_path}")

print("\n🔍 前 10 筆資料：")
print(all_df[["subject_id", "stay_id", "time_bin", "PaO2", "Extubation_failure"]].head(10))
