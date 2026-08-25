"""
意見一：拔管失敗後極早期死亡個案剔除敏感度分析 — 建立排除名單與新 csv

【排除定義】
    拔管後 <= 6 小時死亡（hours_to_death <= 6，含時間戳異常的負值個案，
    依使用者決定一併排除，不於文中另外區分）。

【輸入】
    - hours_to_death_distribution.csv（compute_hours_to_death.py 產出）
    - extubation_features_imputed_gap4_52to4.csv（既有 imputed feature csv，不修改）

【輸出（新檔案，不覆蓋原始檔案）】
    - extubation_features_imputed_excl_earlydeath6h.csv
"""

import pandas as pd

# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
STATS_DIR = r"C:\Users\your-username\Desktop\extubation_failure_prediction\stats"
DATA_DIR = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4"

death_time_df = pd.read_csv(f"{STATS_DIR}\\hours_to_death_distribution.csv")

THRESHOLD_HOURS = 6
early_death_ids = death_time_df.loc[
    death_time_df["hours_to_death"] <= THRESHOLD_HOURS, "stay_id"
]
print(f"排除門檻: hours_to_death <= {THRESHOLD_HOURS}h")
print(f"需剔除的 stay_id 數量: {early_death_ids.nunique():,}")

input_path = f"{DATA_DIR}\\extubation_features_imputed_gap4_52to4.csv"
output_path = f"{DATA_DIR}\\extubation_features_imputed_excl_earlydeath6h.csv"

df = pd.read_csv(input_path)
print(f"\n原始 imputed csv: {len(df):,} 列, {df['stay_id'].nunique():,} 位病人")

df_excl = df[~df["stay_id"].isin(early_death_ids)].copy()
print(f"剔除後: {len(df_excl):,} 列, {df_excl['stay_id'].nunique():,} 位病人")
print(f"實際從 imputed csv 中剔除的病人數: "
      f"{df['stay_id'].nunique() - df_excl['stay_id'].nunique():,}")

# Extubation_failure 比率變化（剔除的都是 EF=1 個案，比率應下降）
orig_rate = df.groupby("stay_id")["Extubation_failure"].first().mean()
new_rate = df_excl.groupby("stay_id")["Extubation_failure"].first().mean()
print(f"\nExtubation_failure 比率: {orig_rate:.4f} -> {new_rate:.4f}")

df_excl.to_csv(output_path, index=False)
print(f"\n✅ 已輸出新檔案（原始檔案未被修改）: {output_path}")
