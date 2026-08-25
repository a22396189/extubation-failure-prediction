"""
意見六：拔管後死亡個案的 hours-to-death 分布

【目的】
    為意見一（拔管失敗後極早期死亡個案：剔除敏感度分析）提供操作型定義的
    門檻依據 —— 算出「因 48 小時內死亡而被標記為 Extubation_failure」的
    病人，實際death time 距離拔管時間有多久，以決定合理的「極早期死亡」
    排除門檻（如 ≤6h）。

【資料來源】
    extubation_outcome.csv（由 build_extubation_outcome.py 產出，未修改）
    欄位：subject_id, stay_id, extubation_time, deathtime, died_within_48h, ...

【不修改任何原始檔案，僅讀取既有 csv 並輸出統計結果】
"""

import pandas as pd

OUTCOME_CSV = (
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
    r"C:\Users\your-username\Desktop\extubation_failure_prediction"
    r"\data\outputs\extubation_outcome.csv"
)

df = pd.read_csv(OUTCOME_CSV, parse_dates=["extubation_time", "deathtime"])

print(f"總病人數: {len(df):,}")
print(f"died_within_48h == 1 的病人數: {(df['died_within_48h'] == 1).sum():,}")

died = df.loc[df["died_within_48h"] == 1].copy()
died["hours_to_death"] = (
    (died["deathtime"] - died["extubation_time"]).dt.total_seconds() / 3600
)

print("\n" + "=" * 60)
print("hours_to_death 描述性統計（拔管後 48h 內死亡個案）")
print("=" * 60)
print(died["hours_to_death"].describe())

print("\n分位數：")
for q in [0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95]:
    print(f"  {int(q*100):3d}th percentile: {died['hours_to_death'].quantile(q):.2f} h")

# ------------------------------------------------------------
# ⚠️ 資料異常檢查：deathtime 早於 extubation_time（hours_to_death < 0）
# build_extubation_outcome.py 只排除 extubation_time == deathtime，
# 對 extubation_time > deathtime（記錄異常）選擇保留，這裡需要單獨列出來看。
# ------------------------------------------------------------
n_died = len(died)
neg = died.loc[died["hours_to_death"] < 0]
print(f"\n⚠️ hours_to_death < 0（deathtime 早於 extubation_time，屬記錄異常）: "
      f"{len(neg)} 人 ({len(neg)/n_died*100:.1f}% of died_within_48h)")
if len(neg) > 0:
    print(neg["hours_to_death"].describe())

print("\n常見門檻下，落在門檻內（即會被剔除）的人數與佔比：")
n_ef = (df["Extubation_failure"] == 1).sum()
n_total = len(df)
for threshold in [3, 6, 12, 24, 48]:
    n_within = (died["hours_to_death"] <= threshold).sum()
    print(
        f"  ≤{threshold:2d}h: {n_within:4d} 人"
        f"  ({n_within/n_died*100:5.1f}% of died_within_48h,"
        f" {n_within/n_ef*100:5.1f}% of all EF=1,"
        f" {n_within/n_total*100:5.2f}% of全部病人)"
    )

# 輸出明細，供後續步驟（列出需剔除的 stay_id）直接使用
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
out_path = (
    r"C:\Users\your-username\Desktop\extubation_failure_prediction"
    r"\stats\hours_to_death_distribution.csv"
)
died[["subject_id", "stay_id", "extubation_time", "deathtime", "hours_to_death"]].to_csv(
    out_path, index=False
)
print(f"\n✅ 明細已輸出: {out_path}")
