# =============================================================
# build_extubation_features_lab_gap4_52to4.py
#
# 【目的】
#   擷取每位 ICU 病人拔管前 52 至 4 小時（嚴格 gap=4h）的實驗室檢驗數值，
#   以每 4 小時為一個時間區段（共 12 bins）計算平均值，
#   作為 Transformer / LSTM 等時序模型的動態輸入特徵。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / extubation_time / Extubation_failure
#   - labevents.csv (MIMIC-IV hosp)：實驗室事件，含 subject_id / hadm_id / itemid / valuenum
#     （無 stay_id，需透過 stay_subject_map.csv 對應）
#   - stay_subject_map.csv：subject_id → stay_id 對應表（本 Cohort 為 1對1）
#
# 【觀測窗設計】
#   - 觀測窗：[t−52h, t−4h)，嚴格左閉右開（gap=4h 防止 leakage）
#   - 分箱：floor 分箱，time_bin 代表區間左端點（-52, -48, ..., -8）
#   - 同一 itemid 群組內多個 itemid 取 mean（coalesce 策略）
#   - 無資料的 bin 補 NaN，每位病人固定輸出 12 列
#
# 【特徵欄位與 ItemID 對應】
#   Cr（Creatinine）：50912, 52546
#   WBC：51301, 51300
#   Hb（Hemoglobin）：51222, 50811, 51640
#   PLT（Platelet）：51265, 53189
#   AnionGap：50868, 52500
#   Lactate：50813, 52442, 53154
#   Glucose：50931, 50809, 52569, 51478
#   （共 7 個連續變數）
#
# 【輸出】
#   - extubation_features_lab_gap4_52to4.csv：
#     長格式，每位病人 12 列（每列一個 time_bin），含 Extubation_failure 標籤
#
# 【執行步驟】
#   Step 1：定義目標 ItemID 群組，DuckDB 篩選讀取 labevents
#   Step 2：pivot 轉寬表（多 itemid → 單欄位，取 mean）
#   Step 3：透過 stay_subject_map 補入 stay_id，合併拔管時間
#   Step 4：嚴格篩選觀測窗，floor 分箱，每 bin 平均
#   Step 5：補齊所有病人的完整 12 bins，輸出 CSV
# =============================================================

# ==========================================================
# build_extubation_features_lab_gap4_52to4.py
# 功能:
#   產生拔管前 52–4 小時（嚴格右開）、每 4 小時平均的實驗室特徵（共12段）
#   (V2: 擴充 ItemIDs + pivot_table mean 作為 coalesce/合併策略)
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
# 若 labevents 為壓縮檔，請改成 .csv.gz（duckdb 的 read_csv_auto 可讀 gz）
labevents_path = rf"{MIMIC_DATA_DIR}\data\mimic-iv-3.1\hosp\labevents.csv"
extub_path = rf"{EXTUBATION_ROOT}\data\outputs\extubation_outcome.csv"
stay_map_path = rf"{EXTUBATION_ROOT}\data\outputs\stay_subject_map.csv"

output_dir = Path(rf"{EXTUBATION_ROOT}\data\outputs\gap4_52to4")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "extubation_features_lab_gap4_52to4.csv"

# === 參數 ===
BIN_H = 4
GAP_H = 4
WINDOW_H = 48  # 12 bins × 4h

# ✅ 12 bins（左端點）：-52, -48, ..., -8
expected_bins = np.arange(-(WINDOW_H + GAP_H), -GAP_H, BIN_H).astype(int)  # -52..-8 step4

# === 連線 DuckDB ===
con = duckdb.connect(duckdb_path)
print("🚀 從 labevents 建立實驗室特徵 (gap4_52to4, V2) ...")

# === 1️⃣ 定義 ItemID 群組 ===
target_items = {
    "Creatinine": [50912, 52546],
    "WBC": [51301, 51300],  # 51301=White Blood Cells, 51300=WBC Count
    "Hemoglobin": [51222, 50811, 51640],  # 51222=Hgb, 50811=Hgb(BG), 51640=Hgb(Heme)
    "Platelet": [51265, 53189],  # 51265=Platelet, 53189=Platelet(Chem)
    "AnionGap": [50868, 52500],
    "Lactate": [50813, 52442, 53154],
    "Glucose": [50931, 50809, 52569, 51478],
}

all_itemids = [x for ids in target_items.values() for x in ids]
id_list_str = ",".join(map(str, all_itemids))

# === 2️⃣ 讀取拔管 outcome 與 stay 對應表 ===
extub = pd.read_csv(extub_path, parse_dates=["extubation_time"])
stay_map = pd.read_csv(stay_map_path)

print(f"✅ extub 共 {len(extub):,} 筆 (拔管事件/病人)")

# === 3️⃣ 讀取 labevents (只抓需要的 itemid) ===
# 注意：labevents 很大，這段會花時間與記憶體
print("📥 執行 SQL 查詢讀取 labevents（只取目標 itemid）...")
lab_df = con.execute(f"""
SELECT
    subject_id,
    hadm_id,
    charttime,
    itemid,
    valuenum
FROM read_csv_auto('{labevents_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
WHERE itemid IN ({id_list_str})
  AND valuenum IS NOT NULL
  AND valuenum > 0
""").df()
print(f"✅ 原始 lab 讀取完成: {len(lab_df):,} 筆")

# === 4️⃣ 映射 feature_name ===
itemid_to_feature = {i: feat for feat, ids in target_items.items() for i in ids}
lab_df["feature_name"] = lab_df["itemid"].map(itemid_to_feature)

# === 5️⃣ Pivot 成寬表（同一時間點多個來源 => mean coalesce）===
print("🔄 轉換為寬表格式 (Pivot)...")
lab_wide = lab_df.pivot_table(
    index=["subject_id", "hadm_id", "charttime"],
    columns="feature_name",
    values="valuenum",
    aggfunc="mean",
).reset_index()
print(f"📊 pivot 後時間點資料: {len(lab_wide):,} 筆")

# === 6️⃣ 對應 stay_id 與拔管時間 ===
# ⚠️ 注意：subject_id → stay_id 可能一對多，若 stay_map 不是唯一對應，會造成重複列
lab_wide = lab_wide.merge(stay_map, on="subject_id", how="left")

# 只保留研究 cohort（inner）
lab_wide = lab_wide.merge(
    extub[["stay_id", "extubation_time", "Extubation_failure"]],
    on="stay_id",
    how="inner",
)

# === 7️⃣ 計算相對時間並嚴格篩選觀測窗 ===
lab_wide["charttime"] = pd.to_datetime(lab_wide["charttime"], errors="coerce")
lab_wide["hours_from_extub"] = (lab_wide["charttime"] - lab_wide["extubation_time"]).dt.total_seconds() / 3600.0

# ✅ 嚴格觀測窗: [-52h, -4h)  =>  -52 <= h < -4
lab_wide = lab_wide[
    (lab_wide["hours_from_extub"] >= -(WINDOW_H + GAP_H)) &
    (lab_wide["hours_from_extub"] < -GAP_H)
].copy()

# === 8️⃣ time_bin（選擇 A：左端點 floor 分箱）===
lab_wide["time_bin"] = (np.floor(lab_wide["hours_from_extub"] / BIN_H) * BIN_H).astype(int)

# === 9️⃣ 輸出欄位命名（和你原本相容）===
rename_map = {
    "Creatinine": "Cr",
    "WBC": "WBC",
    "Hemoglobin": "Hb",
    "Platelet": "PLT",
    "AnionGap": "AnionGap",
    "Lactate": "Lactate",
    "Glucose": "Glucose",
}
# 確保缺欄位也存在（若某 feature pivot 後沒出現）
for k in rename_map.keys():
    if k not in lab_wide.columns:
        lab_wide[k] = np.nan
        print(f"⚠️ pivot 後缺欄位：{k}，自動補 NaN")

# === 🔟 每 4 小時平均（stay-level）===
agg_df = (
    lab_wide.groupby(["subject_id", "stay_id", "time_bin"], as_index=False)[list(rename_map.keys())]
            .mean()
            .rename(columns=rename_map)
)

# === 11️⃣ 補齊完整 12 bins（-52..-8）===
print("⏳ 開始補齊時間序列 (-52h to -4h, gap=4h, 12 bins)...")

final_features = list(rename_map.values())  # ["Cr","WBC","Hb","PLT","AnionGap","Lactate","Glucose"]

all_pairs = extub[["subject_id", "stay_id", "Extubation_failure"]].drop_duplicates()
records = []

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
        for c in final_features:
            subagg[c] = np.nan

    subagg["subject_id"] = sid
    subagg["stay_id"] = stay
    subagg["Extubation_failure"] = failure_val
    records.append(subagg)

all_df = pd.concat(records, ignore_index=True)
print(f"✅ 完成特徵彙整，共 {len(all_df):,} 筆時間片段資料")
print(f"✅ 預期每位病人 12 筆（time_bin = {list(expected_bins)}）")

# 可選：檢查每位病人是否都有 12 bins
cnt = all_df.groupby(["subject_id", "stay_id"]).size()
bad = cnt[cnt != len(expected_bins)]
if len(bad) > 0:
    print(f"⚠️ 有 {len(bad)} 位病人 time_bin 筆數不是 12（可能是 stay_map 一對多造成重複 / merge 問題）")
    print(bad.head(10))

# === 12️⃣ 輸出 ===
all_df.to_csv(output_path, index=False)
print(f"📁 已輸出至: {output_path}")

print("\n🔍 前 10 筆資料：")
print(all_df[["subject_id", "stay_id", "time_bin", "WBC", "Extubation_failure"]].head(10))
