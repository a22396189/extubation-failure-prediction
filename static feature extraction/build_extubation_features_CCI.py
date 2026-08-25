# =============================================================
# build_extubation_features_CCI.py
#
# 【目的】
#   依照 MIMIC-IV 官方 charlson.sql 邏輯，為每位 ICU 病人計算
#   Charlson Comorbidity Index（CCI）總分，作為靜態特徵之一。
#
# 【輸入】
#   - diagnoses_icd.csv (MIMIC-IV hosp)：ICD-9/ICD-10 診斷碼，
#     用於識別 17 種共病症是否存在
#   - icustays.csv (MIMIC-IV icu)：stay_id → hadm_id 精確對應橋樑
#   - extubation_features_age.csv：已計算好的年齡特徵，
#     含 stay_id、age、Extubation_failure，避免重複計算年齡
#
# 【Charlson Score 計算邏輯】
#   (1) 從 diagnoses_icd 抓取 hadm_id 層級的 17 種共病 binary flags：
#       MI、CHF、PVD、CVD、Dementia、COPD、Rheumatic、Peptic Ulcer、
#       Mild Liver、Diabetes（單純/複雜）、Paraplegia、Renal Disease、
#       Cancer、Severe Liver、Metastatic Solid Tumor、AIDS
#   (2) 計算年齡分數（Age Score）：
#       ≤50 → 0、51-60 → 1、61-70 → 2、71-80 → 3、>80 → 4
#   (3) 加權合計：
#       各共病基礎分 1 分（部分加重）：
#         Severe Liver 取代 Mild Liver（×3）
#         Diabetes Complex 取代 Simple（×2）
#         Metastatic Tumor 取代 Cancer（×6 vs ×2）
#         Paraplegia ×2、Renal Disease ×2、AIDS ×6
#
# 【輸出】
#   - extubation_features_charlson.csv：
#       含 Charlson_Score（連續值）+ 17 個共病 binary flags +
#       subject_id、stay_id、Extubation_failure
#
# 【執行步驟】
#   Step 1：DuckDB SQL 提取 17 種共病 flags（hadm_id 層級）
#   Step 2：讀取年齡特徵與 icustays 橋樑表，對應 stay_id → hadm_id
#   Step 3：計算 Age Score 與 Charlson_Score 總分
#   Step 4：輸出最終 CSV
# =============================================================
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# === 路徑設定 ===
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
duckdb_path = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"
diag_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\hosp\diagnoses_icd.csv"
icu_stays_path = r"C:\Users\your-username\Desktop\extubation_project\data\mimic-iv-3.1\icu\icustays.csv"

# 讀取已生成的年齡特徵檔 (裡面已有 age, stay_id, subject_id, Extubation_failure)
age_file_path = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\extubation_features_age.csv"

output_dir = Path(r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs")
output_path = output_dir / "extubation_features_CCI.csv"

con = duckdb.connect(duckdb_path)

# ==========================================================
# 1️⃣ 定義官方 SQL 邏輯 (提取共病症 Flags)
# ==========================================================
# 從 diagnoses_icd 抓取 hadm_id 層級的疾病標記
charlson_query = """
WITH diag_data AS (
    SELECT 
        hadm_id,
        icd_code,
        icd_version
    FROM read_csv_auto('{diag_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)
),
com AS (
    SELECT
        hadm_id,
        -- 1. Myocardial infarction
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('410', '412')) 
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('I21', 'I22')) 
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) = 'I252') 
            THEN 1 ELSE 0 END) AS MI,

        -- 2. Congestive heart failure
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) = '428')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 5) IN ('39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) BETWEEN '4254' AND '4259')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('I43', 'I50'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290'))
            THEN 1 ELSE 0 END) AS CHF,

        -- 3. Peripheral vascular disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('440', '441'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('0930', '4373', '4471', '5571', '5579', 'V434'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) BETWEEN '4431' AND '4439')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('I70', 'I71'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959'))
            THEN 1 ELSE 0 END) AS PVD,

        -- 4. Cerebrovascular disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) BETWEEN '430' AND '438')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 5) = '36234')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('G45', 'G46'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'I60' AND 'I69')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) = 'H340')
            THEN 1 ELSE 0 END) AS CVD,

        -- 5. Dementia
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) = '290')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('2941', '3312'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('F00', 'F01', 'F02', 'F03', 'G30'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('F051', 'G311'))
            THEN 1 ELSE 0 END) AS Dementia,

        -- 6. Chronic pulmonary disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) BETWEEN '490' AND '505')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('4168', '4169', '5064', '5081', '5088'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'J40' AND 'J47')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'J60' AND 'J67')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('I278', 'I279', 'J684', 'J701', 'J703'))
            THEN 1 ELSE 0 END) AS COPD,

        -- 7. Rheumatic disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) = '725')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '7148'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('M05', 'M06', 'M32', 'M33', 'M34'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('M315', 'M351', 'M353', 'M360'))
            THEN 1 ELSE 0 END) AS Rheumatic,

        -- 8. Peptic ulcer disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('531', '532', '533', '534'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('K25', 'K26', 'K27', 'K28'))
            THEN 1 ELSE 0 END) AS Peptic_Ulcer,

        -- 9. Mild liver disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('570', '571'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('0706', '0709', '5733', '5734', '5738', '5739', 'V427'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 5) IN ('07022', '07023', '07032', '07033', '07044', '07054'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('B18', 'K73', 'K74'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944'))
            THEN 1 ELSE 0 END) AS Mild_Liver,

        -- 10. Diabetes without CC
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('2500', '2501', '2502', '2503', '2508', '2509'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('E100', 'E101', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149'))
            THEN 1 ELSE 0 END) AS Diabetes_Simple,

        -- 11. Diabetes with CC
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('2504', '2505', '2506', '2507'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('E102', 'E103', 'E104', 'E105', 'E107', 'E112', 'E113', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147'))
            THEN 1 ELSE 0 END) AS Diabetes_Complex,

        -- 12. Paraplegia
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('342', '343'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('3341', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('G81', 'G82'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('G041', 'G114', 'G801', 'G802', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839'))
            THEN 1 ELSE 0 END) AS Paraplegia,

        -- 13. Renal disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('582', '585', '586', 'V56'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('5880', 'V420', 'V451'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) BETWEEN '5830' AND '5837')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 5) IN ('40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('N18', 'N19'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992'))
            THEN 1 ELSE 0 END) AS Renal_Disease,

        -- 14. Cancer
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) BETWEEN '140' AND '172')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) BETWEEN '1740' AND '1958')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 3) BETWEEN '200' AND '208')
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) = '2386')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('C43', 'C88'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C00' AND 'C26')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C30' AND 'C34')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C37' AND 'C41')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C45' AND 'C58')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C60' AND 'C76')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C81' AND 'C85')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) BETWEEN 'C90' AND 'C97')
            THEN 1 ELSE 0 END) AS Cancer,

        -- 15. Severe liver disease
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 4) IN ('4560', '4561', '4562'))
                   OR (icd_version=9 AND SUBSTR(icd_code, 1, 4) BETWEEN '5722' AND '5728')
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 4) IN ('I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767'))
            THEN 1 ELSE 0 END) AS Severe_Liver,

        -- 16. Metastatic solid tumor
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('196', '197', '198', '199'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('C77', 'C78', 'C79', 'C80'))
            THEN 1 ELSE 0 END) AS Metastatic_Solid_Tumor,

        -- 17. AIDS
        MAX(CASE WHEN (icd_version=9 AND SUBSTR(icd_code, 1, 3) IN ('042', '043', '044'))
                   OR (icd_version=10 AND SUBSTR(icd_code, 1, 3) IN ('B20', 'B21', 'B22', 'B24'))
            THEN 1 ELSE 0 END) AS AIDS

    FROM diag_data
    GROUP BY hadm_id
)
SELECT * FROM com
"""

print("🚀 執行官方 SQL 提取共病特徵...")
cci_components = con.execute(charlson_query.format(diag_path=diag_path)).df()
print(f"✅ 共病特徵提取完成: {len(cci_components):,} 筆住院")

# ==========================================================
# 2️⃣ 讀取年齡與橋樑表，並計算總分
# ==========================================================
print("🔗 讀取 extubation_features_age.csv 與 icustays ...")

# 讀取年齡特徵檔 (這個檔案已經有 stay_id, age, Extubation_failure)
age_df = pd.read_csv(age_file_path)

# 讀取橋樑表 (用來對應 stay_id 和 hadm_id)
icu_stays = con.execute(f"SELECT stay_id, hadm_id FROM read_csv_auto('{icu_stays_path}', SAMPLE_SIZE=-1, IGNORE_ERRORS=true)").df()

# Step A: 將年齡檔 (stay_id) 關聯到 hadm_id
# 注意：merge 使用 inner 可以確保只保留我們 cohort 中的病人
merged_df = age_df.merge(icu_stays, on="stay_id", how="left")

# Step B: 將共病特徵 (hadm_id) 關聯進來
merged_df = merged_df.merge(cci_components, on="hadm_id", how="left")

# Step C: 計算年齡分數 (Age Score) - 根據 Charlson 定義
# <=50: 0, 51-60: 1, 61-70: 2, 71-80: 3, >80: 4
def calculate_age_score(age):
    if age <= 50: return 0
    elif age <= 60: return 1
    elif age <= 70: return 2
    elif age <= 80: return 3
    else: return 4

merged_df["age_score"] = merged_df["age"].apply(calculate_age_score)

# Step D: 填補缺失 (沒對應到的共病視為 0)
comorb_cols = [
    "MI", "CHF", "PVD", "CVD", "Dementia", "COPD", "Rheumatic", "Peptic_Ulcer",
    "Mild_Liver", "Diabetes_Simple", "Diabetes_Complex", "Paraplegia", 
    "Renal_Disease", "Cancer", "Severe_Liver", "Metastatic_Solid_Tumor", "AIDS"
]
merged_df[comorb_cols] = merged_df[comorb_cols].fillna(0)

# Step E: 計算 CCI 總分 (依照官方 SQL 權重邏輯)
merged_df["Charlson_Score"] = (
    merged_df["age_score"] +
    merged_df["MI"] + merged_df["CHF"] + merged_df["PVD"] + merged_df["CVD"] + 
    merged_df["Dementia"] + merged_df["COPD"] + merged_df["Rheumatic"] + merged_df["Peptic_Ulcer"] +
    np.maximum(merged_df["Mild_Liver"], 3 * merged_df["Severe_Liver"]) +
    np.maximum(merged_df["Diabetes_Simple"], 2 * merged_df["Diabetes_Complex"]) +
    np.maximum(2 * merged_df["Cancer"], 6 * merged_df["Metastatic_Solid_Tumor"]) +
    2 * merged_df["Paraplegia"] + 2 * merged_df["Renal_Disease"] +
    6 * merged_df["AIDS"]
)

# ==========================================================
# 3️⃣ 輸出結果
# ==========================================================
# 保留所有 Flags 供 Transformer 使用，保留 Score 供分析，保留 Extubation_failure
# 我們從 age_df 繼承了 Extubation_failure，所以直接輸出即可
output_cols = ["subject_id", "stay_id", "Charlson_Score"] + comorb_cols + ["Extubation_failure"]

final_df_out = merged_df[output_cols].drop_duplicates(subset=["stay_id"])

final_df_out.to_csv(output_path, index=False)
print(f"✅ 最終檔案已建立！")
print(f"📊 輸出筆數: {len(final_df_out)}")
print(f"📁 儲存至: {output_path}")
print(final_df_out.head())