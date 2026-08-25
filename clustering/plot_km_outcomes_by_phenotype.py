import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# =========================
# Paths
# =========================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
BASE     = r"C:\Users\your-username\Desktop\extubation_failure_prediction"

# DuckDB 共用資料庫（MIMIC-IV）
DB_PATH  = r"C:\Users\your-username\Desktop\extubation_project\mimic.duckdb"

CLUSTER_CSV          = rf"{BASE}\results\phenotyping\cluster_assignments.csv"
EXTUBATION_OUTCOME_CSV = rf"{BASE}\data\outputs\extubation_outcome.csv"

# 輸出
OUT_ASSIGN_CSV = rf"{BASE}\results\phenotyping\cluster_outcomes_28d.csv"
FIG_DIR        = rf"{BASE}\results\phenotyping\figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# 設定
READMIT_WINDOW_DAYS = 28
READMIT_START       = "icu_outtime"   # 或 "extubation_time"

# =========================
# Load cluster assignments + extubation time
# =========================
df_cluster = pd.read_csv(CLUSTER_CSV)[["stay_id", "cluster"]].copy()
df_cluster["stay_id"] = df_cluster["stay_id"].astype(int)

df_ext = pd.read_csv(EXTUBATION_OUTCOME_CSV)[["stay_id", "extubation_time"]].copy()
df_ext["stay_id"]        = df_ext["stay_id"].astype(int)
df_ext["extubation_time"] = pd.to_datetime(df_ext["extubation_time"], errors="coerce")

df0 = df_cluster.merge(df_ext, on="stay_id", how="left")
df0 = df0.dropna(subset=["extubation_time"]).copy()

print(f"[Info] Matched {len(df0)} patients with extubation_time")
print(f"[Info] Cluster distribution:\n{df0['cluster'].value_counts().sort_index()}")

# =========================
# SQL: 28-day mortality + ICU readmission
# =========================
con = duckdb.connect(DB_PATH)
con.register("cluster_ext_df", df0)

sql = f"""
WITH base AS (
  SELECT
    c.stay_id,
    c.cluster,
    c.extubation_time,
    i.subject_id,
    i.hadm_id,
    i.intime  AS icu_intime,
    i.outtime AS icu_outtime,
    a.deathtime,
    p.dod
  FROM cluster_ext_df c
  JOIN mimiciv_icu.icustays i
    ON i.stay_id = c.stay_id
  LEFT JOIN mimiciv_hosp.admissions a
    ON a.subject_id = i.subject_id AND a.hadm_id = i.hadm_id
  LEFT JOIN mimiciv_hosp.patients p
    ON p.subject_id = i.subject_id
),
base2 AS (
  SELECT
    *,
    COALESCE(deathtime, dod) AS death_time
  FROM base
),
readmit AS (
  SELECT
    b.stay_id,
    MIN(i2.intime) AS next_icu_intime
  FROM base2 b
  JOIN mimiciv_icu.icustays i2
    ON i2.subject_id = b.subject_id
   AND i2.intime > b.icu_outtime
  GROUP BY b.stay_id
)
SELECT
  b.stay_id,
  b.cluster,
  b.subject_id,
  b.hadm_id,
  b.extubation_time,
  b.icu_intime,
  b.icu_outtime,
  b.death_time,
  r.next_icu_intime,

  -- 28-day mortality
  CASE
    WHEN b.death_time IS NULL THEN 0
    WHEN b.death_time < b.extubation_time THEN 0
    WHEN b.death_time <= b.extubation_time + INTERVAL '{READMIT_WINDOW_DAYS} days' THEN 1
    ELSE 0
  END AS mortality_28d,

  -- Time to death (capped at window; 0 if died before extubation)
  CASE
    WHEN b.death_time IS NULL THEN {READMIT_WINDOW_DAYS}::DOUBLE
    WHEN b.death_time < b.extubation_time THEN 0.0
    ELSE LEAST(
      {READMIT_WINDOW_DAYS}::DOUBLE,
      EXTRACT(EPOCH FROM (b.death_time - b.extubation_time)) / 86400.0
    )
  END AS t_death_days,

  -- 28-day ICU readmission
  CASE
    WHEN r.next_icu_intime IS NULL THEN 0
    WHEN '{READMIT_START}' = 'icu_outtime' THEN
      CASE
        WHEN r.next_icu_intime <= b.icu_outtime + INTERVAL '{READMIT_WINDOW_DAYS} days' THEN 1
        ELSE 0
      END
    ELSE
      CASE
        WHEN r.next_icu_intime <= b.extubation_time + INTERVAL '{READMIT_WINDOW_DAYS} days' THEN 1
        ELSE 0
      END
  END AS icu_readmit_28d

FROM base2 b
LEFT JOIN readmit r ON r.stay_id = b.stay_id
"""

df_out = con.execute(sql).df()
con.close()

print(f"\n[Query] Returned {len(df_out)} rows")

# =========================
# Save outcomes
# =========================
df_out.to_csv(OUT_ASSIGN_CSV, index=False, encoding="utf-8-sig")
print(f"✓ Saved: {OUT_ASSIGN_CSV}")

# Summary per cluster
grp = df_out.groupby("cluster").agg(
    n               = ("stay_id",        "count"),
    mortality_28d   = ("mortality_28d",   "mean"),
    icu_readmit_28d = ("icu_readmit_28d", "mean"),
).reset_index()
grp["mortality_pct"]  = grp["mortality_28d"]  * 100
grp["readmit_pct"]    = grp["icu_readmit_28d"] * 100
print("\n[Summary by cluster]")
print(grp[["cluster", "n", "mortality_pct", "readmit_pct"]].to_string(index=False))

# =========================
# Plot 1: Kaplan-Meier survival by cluster
# =========================
df_km = df_out[["cluster", "t_death_days", "mortality_28d"]].copy()
df_km = df_km.rename(columns={"t_death_days": "T", "mortality_28d": "E"})
df_km["cluster"] = df_km["cluster"].astype(int)

lr   = multivariate_logrank_test(df_km["T"], df_km["cluster"], df_km["E"])
pval = lr.p_value

fig, ax = plt.subplots(figsize=(7, 5))
kmf = KaplanMeierFitter()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for i, cid in enumerate(sorted(df_km["cluster"].unique())):
    sub = df_km[df_km["cluster"] == cid]
    kmf.fit(sub["T"], event_observed=sub["E"], label=f"Cluster {cid}  (n={len(sub)})")
    kmf.plot_survival_function(ci_show=False, ax=ax, color=colors[i % len(colors)])

ax.set_xlim(0, READMIT_WINDOW_DAYS)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Days since extubation", fontsize=12)
ax.set_ylabel("Survival probability", fontsize=12)
ax.set_title(
    f"28-day Survival by Extubation Failure Phenotype\nlog-rank p = {pval:.4g}",
    fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

km_png = Path(FIG_DIR) / f"KM_survival_28d_by_cluster_{READMIT_START}_win{READMIT_WINDOW_DAYS}d.png"
km_pdf = Path(FIG_DIR) / f"KM_survival_28d_by_cluster_{READMIT_START}_win{READMIT_WINDOW_DAYS}d.pdf"
plt.savefig(km_png, dpi=300, bbox_inches="tight")
plt.savefig(km_pdf, bbox_inches="tight")
print(f"\n✓ KM plot saved: {km_png}")
plt.close()

# =========================
# Plot 2: Bar chart — mortality + readmission by cluster
# =========================
x     = np.arange(len(grp))
width = 0.38

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, grp["mortality_pct"],  width,
       label="28-day mortality (%)",          color="#d62728", alpha=0.85)
ax.bar(x + width / 2, grp["readmit_pct"],    width,
       label=f"ICU readmission ≤{READMIT_WINDOW_DAYS}d (%)", color="#1f77b4", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"Cluster {int(c)}\n(n={int(n)})" for c, n in zip(grp["cluster"], grp["n"])],
    fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_title("Outcomes by Extubation Failure Phenotype", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, ls="--")
plt.tight_layout()

bar_png = Path(FIG_DIR) / f"Bar_outcomes_by_cluster_{READMIT_START}_win{READMIT_WINDOW_DAYS}d.png"
bar_pdf = Path(FIG_DIR) / f"Bar_outcomes_by_cluster_{READMIT_START}_win{READMIT_WINDOW_DAYS}d.pdf"
plt.savefig(bar_png, dpi=300, bbox_inches="tight")
plt.savefig(bar_pdf, bbox_inches="tight")
print(f"✓ Bar chart saved: {bar_png}")
plt.close()

print("\n✅ 完成。請參考 summary 選出最低風險 cluster 作為 plot_cluster_odds_ratios.py 的 REF_CLUSTER。")
