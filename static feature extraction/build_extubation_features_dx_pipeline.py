# =============================================================
# build_extubation_features_dx_pipeline.py
#
# 【目的】
#   將每位 ICU 病人的主要診斷（Primary Diagnosis）從原始 ICD 碼
#   依序對應至 CCS / CCSR 官方分類，最終收斂為 10 大臨床類別，
#   供後續 Subgroup Analysis 使用（不作為模型輸入特徵）。
#
# 【輸入】
#   - extubation_outcome.csv：最終 Cohort，含 stay_id / subject_id
#   - diagnoses_icd.csv (MIMIC-IV hosp)：ICD 診斷碼，seq_num=1 為主診斷
#   - icustays.parquet (MIMIC-IV icu)：stay_id → hadm_id 精確對應橋樑
#   - DXCCSR_v2026-1.csv (AHRQ 官方)：ICD-10 → CCSR 映射表
#   - $dxref 2015.csv (AHRQ 官方)：ICD-9 → CCS 映射表
#
# 【ICD 映射說明】
#   ICD-9  → CCS（Clinical Classifications Software）
#             單層分類，共約 17 大類，以數字編號（如 2=Septicemia、122=Pneumonia）
#   ICD-10 → CCSR（CCS Refined）
#             細分為 530+ 類別，前 3 碼為 Body System 前綴（如 RSP=Respiratory）
#
# 【最終分類（13 大類）】
#   Infectious / Respiratory / Circulatory / Neurological / Digestive /
#   Genitourinary / Neoplasms / Endocrine / Hematologic /
#   Musculoskeletal / Mental / Injury_Poisoning / Others/Unknown
#
# 【輸出】
#   - extubation_features_final_categories.csv：含 DxGroup_Major（主診斷大類）
#   - （可選）checkpoint_1_raw_icd.csv：Step 1 中間結果，供 debug 用
#   - （可選）checkpoint_2_ccs_mapped.csv：Step 2 中間結果，供 debug 用
#
# 【執行步驟】
#   Step 1：從 MIMIC-IV 抽取主診斷 ICD 碼（seq_num=1）
#   Step 2：載入 AHRQ 官方映射表，將 ICD 碼對應至 CCS/CCSR 分類碼
#   Step 3：將 CCS/CCSR 分類碼收斂為 10 大臨床類別（DxGroup_Major）
#   Step 4：統計摘要與輸出最終 CSV
#
# 【注意】
#   本程式合併自以下三個舊版腳本（已可退役）：
#     - build_extubation_features_diagnosis.py
#     - build_extubation_features_primary_diagnosis_mapping.py
#     - build_extubation_features_final_categories.py
# =============================================================

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# 設定開關
# =============================================================
# 設為 True 時，Step 1 與 Step 2 的中間結果會存成 CSV，供 debug 使用
SAVE_CHECKPOINTS = True

# =============================================================
# 路徑設定
# =============================================================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
duckdb_path   = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"

# 輸入：MIMIC-IV 原始資料
diag_path     = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\hosp\diagnoses_icd.csv"
icustays_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\icu\icustays.parquet"
extub_path    = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_outcome.csv"

# 輸入：AHRQ 官方映射表
ccsr_map_file = Path(r"C:\Users\your-username\Desktop\extubation_project\data\outputs\DXCCSR_v2026-1.csv")
ccs9_map_file = Path(r"C:\Users\your-username\Desktop\extubation_project\data\outputs\$dxref 2015.csv")

# 輸出
output_dir  = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path       = output_dir / "extubation_features_final_categories.csv"
checkpoint1_path  = output_dir / "checkpoint_1_raw_icd.csv"
checkpoint2_path  = output_dir / "checkpoint_2_ccs_mapped.csv"

# =============================================================
# 工具函數
# =============================================================
def strip_col_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """去除欄位名稱的單/雙引號與前後空白。"""
    df.columns = [c.strip().strip("'\"") for c in df.columns]
    return df


def clean_icd_code(code) -> str | None:
    """
    統一清理診斷碼格式：
      1. strip()          — 去最外層空白
      2. strip("'\"")     — 去單/雙引號
      3. strip()          — 去引號內層空白（e.g. "'0389 '" → "0389"）
      4. upper()          — 轉大寫
      5. replace('.','')  — 去小數點（ICD-10 格式差異）
    回傳 None 代表無效碼。
    """
    if pd.isna(code):
        return None
    s = str(code).strip().strip("'\"").strip().upper().replace('.', '')
    return s if s not in ('', 'UNKNOWN') else None


# =============================================================
# Step 1：從 MIMIC-IV 抽取主診斷 ICD 碼（seq_num = 1）
# =============================================================
print("=" * 60)
print("Step 1：抽取主診斷 ICD 碼（seq_num=1）")
print("=" * 60)

con = duckdb.connect(duckdb_path)

# 讀取 seq_num=1 的主診斷紀錄
diag_df = con.execute(f"""
    SELECT hadm_id, icd_code, icd_version
    FROM read_csv_auto('{diag_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
    WHERE seq_num = 1
""").df()
print(f"✅ 主診斷紀錄: {len(diag_df):,} 筆（涵蓋 {diag_df['hadm_id'].nunique():,} 個 hadm_id）")

# 讀取拔管名單
extub = pd.read_csv(extub_path)
print(f"✅ 拔管名單: {len(extub):,} 筆")

# stay_id → hadm_id（精確查表，不用 subject_id 時間推算）
icu_stays = con.execute(f"""
    SELECT stay_id, hadm_id
    FROM read_parquet('{icustays_path}')
""").df()

if "hadm_id" not in extub.columns:
    extub = extub.merge(icu_stays, on="stay_id", how="left")

missing_hadm = extub["hadm_id"].isna().sum()
if missing_hadm > 0:
    print(f"⚠️  {missing_hadm} 筆 stay_id 無對應 hadm_id，將被排除")
    extub = extub.dropna(subset=["hadm_id"])

# hadm_id → ICD code
diag_merged = extub[["subject_id", "stay_id", "hadm_id", "Extubation_failure"]].merge(
    diag_df, on="hadm_id", how="inner"
)

# 追蹤缺失
extub_hadm_set   = set(extub["hadm_id"].unique())
diag_hadm_set    = set(diag_df["hadm_id"].unique())
missing_in_diag  = extub_hadm_set - diag_hadm_set

print(f"\n🔍 缺失追蹤：")
print(f"   diagnoses_icd 中完全沒有: {len(missing_in_diag)} 筆 hadm_id（標記為 UNKNOWN）")

# 為無診斷記錄的 stay_id 加入 UNKNOWN 佔位列
null_codes = diag_merged["icd_code"].isna().sum()
if null_codes > 0:
    diag_merged = diag_merged.dropna(subset=["icd_code"])

diag_merged["Primary_Diagnosis"] = diag_merged["icd_code"].astype(str).str.strip().str.upper()
diag_merged["ICD_Version"] = diag_merged["icd_version"].map({9: "ICD-9", 10: "ICD-10"}).fillna("Unknown")

# 每個 stay_id 只保留一筆
diag_unique = diag_merged.groupby("stay_id", as_index=False).first()[[
    "subject_id", "stay_id", "Primary_Diagnosis", "ICD_Version", "Extubation_failure"
]]

# 補入 UNKNOWN 佔位
if len(missing_in_diag) > 0:
    missing_rows = extub[extub["hadm_id"].isin(missing_in_diag)][
        ["subject_id", "stay_id", "Extubation_failure"]
    ].copy()
    missing_rows["Primary_Diagnosis"] = "UNKNOWN"
    missing_rows["ICD_Version"]       = "Unknown"
    diag_unique = pd.concat([diag_unique, missing_rows], ignore_index=True)

print(f"✅ Step 1 完成：{len(diag_unique):,} 筆")
print(f"   ICD-9 : {(diag_unique['ICD_Version']=='ICD-9').sum():,}")
print(f"   ICD-10: {(diag_unique['ICD_Version']=='ICD-10').sum():,}")
print(f"   UNKNOWN: {(diag_unique['ICD_Version']=='Unknown').sum():,}")

if SAVE_CHECKPOINTS:
    diag_unique.to_csv(checkpoint1_path, index=False, encoding='utf-8-sig')
    print(f"💾 [Checkpoint 1] 儲存至: {checkpoint1_path}")

# =============================================================
# Step 2：載入 AHRQ 官方映射表，將 ICD 碼對應至 CCS/CCSR 分類碼
# =============================================================
print("\n" + "=" * 60)
print("Step 2：ICD → CCS / CCSR 官方映射")
print("=" * 60)

# 清理原始診斷碼格式
diag_unique["diag_clean"] = diag_unique["Primary_Diagnosis"].apply(clean_icd_code)

# ── 讀取 ICD-10 → CCSR 映射表 ───────────────────────────────
print("  讀取 CCSR (ICD-10) 映射表...")
ccsr_map_dict = {}
try:
    ccsr_df = pd.read_csv(ccsr_map_file, dtype=str, on_bad_lines='skip')
    ccsr_df = strip_col_quotes(ccsr_df)
    ccsr_cols = ccsr_df.columns.tolist()

    code_col = next(
        (c for c in ccsr_cols
         if 'ICD-10' in c.upper() and 'CODE' in c.upper() and 'DESCRIPTION' not in c.upper()),
        ccsr_cols[0]
    )
    ccsr_col = next(
        (c for c in ccsr_cols
         if 'DEFAULT' in c.upper() and 'CCSR' in c.upper()
         and 'IP' in c.upper() and 'DESCRIPTION' not in c.upper()),
        None
    )
    if ccsr_col is None:
        ccsr_col = next(
            (c for c in ccsr_cols
             if 'CCSR' in c.upper() and 'CATEGORY' in c.upper()
             and 'DESCRIPTION' not in c.upper()),
            None
        )
    if ccsr_col is None:
        raise ValueError("找不到 CCSR 分類欄位")

    for _, row in ccsr_df.iterrows():
        code = clean_icd_code(row[code_col])
        ccsr = clean_icd_code(row[ccsr_col])
        if code and ccsr and ccsr not in ('', 'NAN'):
            ccsr_map_dict[code] = ccsr

    print(f"  ✅ CCSR 映射字典: {len(ccsr_map_dict):,} 筆（ICD-10 代碼欄: {code_col}，CCSR 欄: {ccsr_col}）")
except Exception as e:
    print(f"  ❌ CCSR 映射表載入失敗: {e}")

# ── 讀取 ICD-9 → CCS 映射表 ─────────────────────────────────
print("  讀取 CCS (ICD-9) 映射表...")
ccs9_map_dict = {}
try:
    ccs9_df = pd.read_csv(ccs9_map_file, dtype=str, skiprows=1)
    ccs9_df = strip_col_quotes(ccs9_df)

    for _, row in ccs9_df.iterrows():
        code = clean_icd_code(row['ICD-9-CM CODE'])
        ccs  = clean_icd_code(row['CCS CATEGORY'])
        if code and ccs and ccs not in ('', '0', 'NAN'):
            ccs9_map_dict[code] = ccs

    print(f"  ✅ CCS 映射字典: {len(ccs9_map_dict):,} 筆")
except Exception as e:
    print(f"  ❌ CCS 映射表載入失敗: {e}")

# ── 執行映射 ─────────────────────────────────────────────────
def map_diagnosis(row):
    code = row["diag_clean"]
    if pd.isna(code):
        return "UNKNOWN"
    if row["ICD_Version"] == "ICD-10":
        return ccsr_map_dict.get(code, "CCSR_UNMAPPED")
    elif row["ICD_Version"] == "ICD-9":
        return ccs9_map_dict.get(code, "CCS_UNMAPPED")
    return "UNKNOWN"

diag_unique["DxGroup_Code"] = diag_unique.apply(map_diagnosis, axis=1)

# 映射統計
icd10_df = diag_unique[diag_unique["ICD_Version"] == "ICD-10"]
icd9_df  = diag_unique[diag_unique["ICD_Version"] == "ICD-9"]

mapped10 = ~icd10_df["DxGroup_Code"].isin(["CCSR_UNMAPPED", "UNKNOWN"])
mapped9  = ~icd9_df["DxGroup_Code"].isin(["CCS_UNMAPPED",  "UNKNOWN"])

print(f"\n  【ICD-10 (CCSR)】 {len(icd10_df):,} 筆")
print(f"    ✅ 成功映射: {mapped10.sum():,}  ({100*mapped10.mean():.1f}%)")
print(f"    ❌ 無法映射: {(icd10_df['DxGroup_Code']=='CCSR_UNMAPPED').sum():,}")
print(f"  【ICD-9  (CCS)】  {len(icd9_df):,} 筆")
if len(icd9_df) > 0:
    print(f"    ✅ 成功映射: {mapped9.sum():,}  ({100*mapped9.mean():.1f}%)")
    print(f"    ❌ 無法映射: {(icd9_df['DxGroup_Code']=='CCS_UNMAPPED').sum():,}")

# 未映射碼前 10 分析
for tag, label in [("CCSR_UNMAPPED", "ICD-10 未映射"), ("CCS_UNMAPPED", "ICD-9 未映射")]:
    sub = diag_unique[diag_unique["DxGroup_Code"] == tag]
    if len(sub) > 0:
        top = sub["diag_clean"].value_counts().head(10)
        print(f"\n  🔍 {label}（共 {len(sub):,} 筆，前 10 碼）：")
        for code, cnt in top.items():
            print(f"      {code}: {cnt} 筆")

if SAVE_CHECKPOINTS:
    diag_unique.to_csv(checkpoint2_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 [Checkpoint 2] 儲存至: {checkpoint2_path}")

# =============================================================
# Step 3：收斂為 10 大臨床類別（DxGroup_Major）
# =============================================================
print("\n" + "=" * 60)
print("Step 3：收斂為 10 大臨床類別（DxGroup_Major）")
print("=" * 60)

# CCSR 前綴 → 大類（ICD-10 用）
CCSR_PREFIX_MAP = {
    'INF': 'Infectious',
    'RSP': 'Respiratory',
    'CIR': 'Circulatory',
    'NVS': 'Neurological',
    'MNS': 'Neurological',
    'DIG': 'Digestive',
    'GEN': 'Genitourinary',
    'INJ': 'Injury_Poisoning',
    'EXT': 'Injury_Poisoning',
    'NEO': 'Neoplasms',
    'END': 'Endocrine',
    'BLD': 'Hematologic',
    'MUS': 'Musculoskeletal',
    'SKN': 'Musculoskeletal',
    'MHD': 'Mental',
    'PRG': 'Others/Unknown',
    'NEW': 'Others/Unknown',
    'CON': 'Others/Unknown',
    'SYM': 'Others/Unknown',
    'FAC': 'Others/Unknown',
}

def map_to_major_category(row):
    group_code = str(row["DxGroup_Code"]).strip()
    icd_ver    = str(row["ICD_Version"]).strip()

    if group_code in ("UNKNOWN", "CCSR_UNMAPPED", "CCS_UNMAPPED", "nan", ""):
        return "Others/Unknown"

    # ICD-10：依 CCSR 前綴 3 碼
    if icd_ver == "ICD-10":
        prefix = group_code[:3].upper()
        return CCSR_PREFIX_MAP.get(prefix, "Others/Unknown")

    # ICD-9：依 CCS Single-Level 編號區間
    if icd_ver == "ICD-9":
        try:
            n = int(group_code)
            if   1 <= n <=   9: return "Infectious"
            if  11 <= n <=  47: return "Neoplasms"
            if  48 <= n <=  58: return "Endocrine"
            if  59 <= n <=  64: return "Hematologic"
            if  65 <= n <=  75: return "Mental"
            if  76 <= n <=  95: return "Neurological"
            if  96 <= n <= 121: return "Circulatory"
            if 122 <= n <= 134: return "Respiratory"
            if 135 <= n <= 155: return "Digestive"
            if 156 <= n <= 175: return "Genitourinary"
            if 196 <= n <= 212: return "Musculoskeletal"
            if 225 <= n <= 260: return "Injury_Poisoning"
        except ValueError:
            return "Others/Unknown"

    return "Others/Unknown"

diag_unique["DxGroup_Major"] = diag_unique.apply(map_to_major_category, axis=1)

print("✅ Step 3 完成")

# =============================================================
# Step 4：統計摘要與輸出最終 CSV
# =============================================================
print("\n" + "=" * 60)
print("Step 4：統計摘要")
print("=" * 60)

# 疾病大類別分佈
print("\n📊 疾病大類別分佈：")
counts = diag_unique["DxGroup_Major"].value_counts()
total  = len(diag_unique)
for cat, cnt in counts.items():
    bar = "█" * max(1, int(cnt / counts.max() * 30))
    print(f"  {cat:<22} {cnt:>5,} ({100*cnt/total:.1f}%)  {bar}")

# 各類別拔管失敗率
print("\n💡 各疾病大類別之拔管失敗率：")
summary = (
    diag_unique.groupby("DxGroup_Major")["Extubation_failure"]
    .agg(失敗數="sum", 總數="count", 失敗率="mean")
    .sort_values("失敗率", ascending=False)
)
for cat, row in summary.iterrows():
    print(f"  {cat:<22}: {row['失敗率']:.2%}  ({int(row['失敗數'])}/{int(row['總數'])})")

# 整體映射成功率
success = ~diag_unique["DxGroup_Code"].isin(["UNKNOWN", "CCSR_UNMAPPED", "CCS_UNMAPPED"])
print(f"\n📈 整體映射成功率: {success.sum():,} / {total:,} ({100*success.mean():.2f}%)")

# 輸出最終 CSV
output_cols = [
    "subject_id", "stay_id",
    "Primary_Diagnosis", "ICD_Version",
    "diag_clean", "DxGroup_Code", "DxGroup_Major",
    "Extubation_failure"
]
diag_unique[output_cols].to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ 最終特徵檔已儲存: {output_path}")
print(f"\n🔍 前 10 筆預覽：")
print(diag_unique[output_cols].head(10).to_string(index=False))
