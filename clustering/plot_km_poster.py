#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_km_poster.py

從已儲存的 cluster_outcomes_28d.csv 產生 poster 用高品質 KM 存活曲線。
不需要重新連接 DuckDB，直接讀取結果。

執行：
python "C:/Users/your-username/Desktop/extubation_failure_prediction/clustering/plot_km_poster.py"
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# =========================
# Paths
# =========================
# 【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，於其他環境執行前請依實際檔案存放位置調整
BASE           = r"C:\Users\your-username\Desktop\extubation_failure_prediction"
OUTCOMES_CSV   = rf"{BASE}\results\phenotyping\cluster_outcomes_28d.csv"
FIG_DIR        = rf"{BASE}\results\phenotyping\figures"
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

READMIT_WINDOW_DAYS = 28

# =========================
# 可自訂的 phenotype 標籤
# cluster id → 短標籤（根據 KM 結果由低到高風險命名）
# 依存活率排序：C2(best) > C0 > C3 > C1(worst)
# 改為 Cluster 1–4（嚴重度排序：1=最差/Critical，4=最輕/Low Risk）
# 與 poster 內文「Cluster 1 = worst prognosis」一致
# =========================
# ⚠️  跑完 plot_km_outcomes_by_phenotype.py 確認 28 天死亡率後再更新此對應表
# 目前根據臨床剖析（FiO2 / vasopressor / lactate）暫定排序：
#   C1（FiO2=0.63, vasopressor=77.6%, lactate=3.02）→ 最差
#   C0（FiO2=0.44, vasopressor=59.3%, lactate=2.05）→ 次差
#   C3（FiO2=0.41, vasopressor=23.4%, GCS=8.40）   → 次佳
#   C2（FiO2=0.43, vasopressor=33.1%, PaO2=112）   → 最佳
PHENOTYPE_LABELS = {
    1: "Phenotype 1 (Critical)",       # C1 → 28d mortality 83.3%
    0: "Phenotype 2 (High Risk)",      # C0 → 28d mortality 72.8%
    2: "Phenotype 3 (Moderate Risk)",  # C2 → 28d mortality 50.0%
    3: "Phenotype 4 (Low Risk)",       # C3 → 28d mortality 40.2%
}

# 顏色：從高風險→低風險（紅→橙→藍→綠），colorblind-friendly
PHENOTYPE_COLORS = {
    1: "#c0392b",   # red    — Critical     (83.3%)
    0: "#e67e22",   # orange — High Risk    (72.8%)
    2: "#2980b9",   # blue   — Moderate     (50.0%)
    3: "#27ae60",   # green  — Low Risk     (40.2%)
}

# 繪圖順序（由高風險到低風險，圖例由上到下為差到好）
PLOT_ORDER = [1, 0, 2, 3]

# =========================
# Load
# =========================
df_out = pd.read_csv(OUTCOMES_CSV)
df_out["cluster"] = df_out["cluster"].astype(int)

# =========================
# Log-rank test
# =========================
df_km = df_out[["cluster", "t_death_days", "mortality_28d"]].dropna().copy()
df_km.rename(columns={"t_death_days": "T", "mortality_28d": "E"}, inplace=True)

lr   = multivariate_logrank_test(df_km["T"], df_km["cluster"], df_km["E"])
pval = lr.p_value
pval_str = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"

# =========================
# Poster KM plot
# =========================
matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

fig, ax = plt.subplots(figsize=(8, 5.5))

kmf = KaplanMeierFitter()
cluster_n = {}

for cid in PLOT_ORDER:
    sub = df_km[df_km["cluster"] == cid].copy()
    if len(sub) == 0:
        continue
    n = len(sub)
    cluster_n[cid] = n
    label = PHENOTYPE_LABELS.get(cid, f"Cluster {cid}")
    # 去掉換行符，在括號前加 n
    legend_label = label.replace("\n", " ") + f"  (n={n})"
    kmf.fit(sub["T"], event_observed=sub["E"], label=legend_label)
    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.12,
        color=PHENOTYPE_COLORS.get(cid, "grey"),
        lw=2.5,
    )

# Axes formatting
ax.set_xlim(0, READMIT_WINDOW_DAYS)
ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("Days after Extubation", fontsize=14, labelpad=6)
ax.set_ylabel("Survival Probability", fontsize=14, labelpad=6)
ax.xaxis.set_major_locator(mticker.MultipleLocator(7))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.tick_params(axis="both", which="major", length=5)
ax.tick_params(axis="x",    which="minor", length=3)

# 移除上方與右側邊框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 標題（poster 用短標題）
ax.set_title(f"28-Day Survival by Extubation Failure Phenotype\nlog-rank {pval_str}",
             fontweight="bold", fontsize=15, pad=10)

# 圖例
leg = ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="lightgrey",
    title="Cluster",
    title_fontsize=11,
)

# 水平參考線（50% 存活）
ax.axhline(0.5, color="grey", lw=1, ls=":", alpha=0.6)
ax.text(READMIT_WINDOW_DAYS + 0.3, 0.5, "50%", va="center",
        color="grey", fontsize=10)

plt.tight_layout()

# =========================
# Save
# =========================
out_png = Path(FIG_DIR) / "KM_survival_28d_poster.png"
out_pdf = Path(FIG_DIR) / "KM_survival_28d_poster.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"✓ PNG: {out_png}")
print(f"✓ PDF: {out_pdf}")
plt.show()
plt.close()

# =========================
# Print 28-day survival summary
# =========================
print("\n[28-Day Survival Summary]")
print(f"{'Cluster':<10} {'Phenotype':<30} {'N':>5} {'28d Survival':>14} {'28d Mortality':>14}")
print("-" * 75)
for cid in PLOT_ORDER:
    sub = df_out[df_out["cluster"] == cid]
    surv = 1 - sub["mortality_28d"].mean()
    mort = sub["mortality_28d"].mean()
    label = PHENOTYPE_LABELS.get(cid, f"Cluster {cid}").replace("\n", " ")
    print(f"C{cid:<9} {label:<30} {len(sub):>5} {surv:>13.1%} {mort:>13.1%}")
print(f"\nlog-rank {pval_str}")
