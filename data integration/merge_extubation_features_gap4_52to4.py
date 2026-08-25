# =============================================================
# merge_extubation_features_gap4_52to4.py
#
# 【目的】
#   將所有靜態特徵與時間序列特徵合併為一張大表，
#   作為模型訓練的最終輸入資料。
#
# 【輸入】
#   靜態特徵（每位病人 1 列，以 subject_id + stay_id 合併）：
#   - extubation_features_age.csv
#   - extubation_features_bmi.csv
#   - extubation_features_sex.csv
#   - extubation_features_CCI.csv（Charlson Comorbidity Index）
#   ⚠️ primary_diagnosis 不作為模型特徵，不納入合併
#
#   時間序列特徵（每位病人 12 列，以 subject_id + stay_id + time_bin 合併）：
#   - extubation_features_vitalsign_gap4_52to4.csv
#   - extubation_features_bg_gap4_52to4.csv
#   - extubation_features_lab_gap4_52to4.csv
#   - extubation_features_ventilator_gap4_52to4.csv
#   - extubation_features_vasopressor_gap4_52to4.csv
#   - extubation_features_rrt_gap4_52to4.csv
#   - extubation_features_io_gap4_52to4.csv
#   - extubation_features_gcs_gap4_52to4.csv
#
# 【合併策略】
#   - 時序特徵間：left join（以第一個時序檔為基準，避免 outer join 產生幽靈列）
#   - 靜態特徵加入時序：left join（保留所有時序列，靜態缺失補 NaN）
#   - Extubation_failure 標籤以時序第一個檔案為準，後續重複欄位自動移除
#
# 【輸出】
#   - extubation_features_merged_gap4_52to4.csv：
#     長格式，每位病人 12 列（N × 12 列），含所有特徵與標籤
#
# 【執行步驟】
#   Step 1：依序 left join 各靜態特徵檔
#   Step 2：依序 left join 各時序特徵檔
#   Step 3：將靜態特徵 left join 至時序大表
#   Step 4：驗證標籤完整性，輸出 CSV
# =============================================================

import pandas as pd
from pathlib import Path

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
static_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
ts_dir     = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4")
output_path = ts_dir / "extubation_features_merged_gap4_52to4.csv"

# =============================================================
# 檔案清單
# =============================================================

# 靜態特徵（每人 1 列）
# ⚠️ primary_diagnosis 不作為模型特徵，不納入此清單
static_files = [
    static_dir / "extubation_features_age.csv",
    static_dir / "extubation_features_bmi.csv",
    static_dir / "extubation_features_sex.csv",
    static_dir / "extubation_features_CCI.csv",   # Charlson Comorbidity Index
]

# 時序特徵（每人 12 列，time_bin: -52 到 -8）
ts_files = [
    ts_dir / "extubation_features_vitalsign_gap4_52to4.csv",
    ts_dir / "extubation_features_bg_gap4_52to4.csv",
    ts_dir / "extubation_features_lab_gap4_52to4.csv",
    ts_dir / "extubation_features_ventilator_gap4_52to4.csv",
    ts_dir / "extubation_features_vasopressor_gap4_52to4.csv",
    ts_dir / "extubation_features_rrt_gap4_52to4.csv",
    ts_dir / "extubation_features_io_gap4_52to4.csv",
    ts_dir / "extubation_features_gcs_gap4_52to4.csv",
]

TARGET_COL = "Extubation_failure"
ID_COLS    = ["subject_id", "stay_id"]
KEY_TS     = ["subject_id", "stay_id", "time_bin"]

# =============================================================
# Helper：safe read
# =============================================================
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ⚠️ 找不到檔案：{path.name}，跳過。")
        return None
    df = pd.read_csv(path)
    print(f"  📖 {path.name}  shape={df.shape}")
    return df

# =============================================================
# Step 1：合併靜態特徵
# =============================================================
print("=" * 60)
print("Step 1：合併靜態特徵（left join on subject_id + stay_id）")
print("=" * 60)

static_df = None

for path in static_files:
    df = safe_read_csv(path)
    if df is None:
        continue

    # 防呆：確保 key 欄位存在
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"❌ {path.name} 缺少鍵欄位 '{c}'")

    if static_df is None:
        static_df = df
    else:
        # 移除重複的標籤欄，避免 _x/_y 後綴
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        # left join：以第一個靜態檔（age）為基準，確保人數不膨脹
        static_df = static_df.merge(df, on=ID_COLS, how="left")

if static_df is None:
    print("⚠️ 沒有任何靜態檔案成功讀取，靜態特徵將缺失。")
else:
    n_patients = static_df["stay_id"].nunique()
    print(f"✅ 靜態特徵合併完成：{static_df.shape}，涵蓋 {n_patients:,} 位病人")

# =============================================================
# Step 2：合併時間序列特徵
# =============================================================
print("\n" + "=" * 60)
print("Step 2：合併時間序列特徵（left join on subject_id + stay_id + time_bin）")
print("=" * 60)

ts_df = None

for path in ts_files:
    df = safe_read_csv(path)
    if df is None:
        continue

    # 防呆：確保 key 欄位存在
    for c in KEY_TS:
        if c not in df.columns:
            raise ValueError(f"❌ {path.name} 缺少鍵欄位 '{c}'")

    if ts_df is None:
        ts_df = df
    else:
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        # left join：以第一個時序檔（vitalsign）為基準，避免 outer join 產生幽靈列
        ts_df = ts_df.merge(df, on=KEY_TS, how="left")

if ts_df is None:
    raise RuntimeError("❌ 沒有任何時間序列檔案成功讀取，無法產生 merged dataset")

n_ts_patients = ts_df["stay_id"].nunique()
print(f"✅ 時序特徵合併完成：{ts_df.shape}，涵蓋 {n_ts_patients:,} 位病人（應 = {n_ts_patients} × 12 = {n_ts_patients*12:,} 列）")

# =============================================================
# Step 3：將靜態特徵 left join 至時序大表
# =============================================================
print("\n" + "=" * 60)
print("Step 3：靜態特徵 left join 至時序大表")
print("=" * 60)

if static_df is None:
    final_df = ts_df.copy()
    print("⚠️ 無靜態特徵可合併，僅保留時序資料。")
else:
    # 移除靜態表內的標籤欄（標籤由時序表的第一欄持有）
    static_df_clean = static_df.drop(columns=[TARGET_COL], errors="ignore")
    final_df = ts_df.merge(static_df_clean, on=ID_COLS, how="left")
    print(f"✅ 最終合併完成：{final_df.shape}")

# =============================================================
# Step 4：驗證與輸出
# =============================================================
print("\n" + "=" * 60)
print("Step 4：驗證與輸出")
print("=" * 60)

# 確認標籤存在
if TARGET_COL not in final_df.columns:
    raise RuntimeError(
        f"❌ 最終資料表缺少 '{TARGET_COL}' 欄位！\n"
        "   請確認至少一個時序特徵檔案內含有 Extubation_failure。"
    )

# 檢查標籤缺失（left join 下不應出現）
missing_target = final_df[TARGET_COL].isna().sum()
if missing_target > 0:
    print(f"⚠️ 警告：有 {missing_target:,} 筆資料缺少標籤，請檢查時序來源檔。")
else:
    print(f"✅ 標籤完整，無缺失。")

# 確認每個 stay_id 剛好 12 列
bin_counts = final_df.groupby("stay_id")["time_bin"].count()
not_12 = bin_counts[bin_counts != 12]
if not not_12.empty:
    print(f"⚠️ 警告：有 {len(not_12)} 個 stay_id 的 time_bin 數量不等於 12：")
    print(not_12.head(10))
else:
    print(f"✅ 所有 stay_id 均有完整 12 個 time_bin。")

# 靜態特徵缺失摘要
static_feature_cols = ["age", "BMI", "sex", "Charlson_Score"]
print("\n📊 靜態特徵缺失摘要（每 stay 應各僅 1 值）：")
for col in static_feature_cols:
    if col in final_df.columns:
        n_miss = final_df[col].isna().sum()
        print(f"   {col:20s}  缺失 {n_miss:>6,} 筆 ({n_miss/len(final_df)*100:.1f}%)")

# 輸出
ts_dir.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"\n🎉 所有特徵合併完成！")
print(f"📊 最終資料維度：{final_df.shape}")
print(f"📁 輸出至：{output_path}")

# 預覽
feature_cols  = [c for c in final_df.columns if c not in KEY_TS + [TARGET_COL]]
preview_cols  = KEY_TS + feature_cols[:5] + [TARGET_COL]
print("\n🔍 前 5 筆資料預覽：")
print(final_df[preview_cols].head(5))
