#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
replot_umap_with_labels.py

從已儲存的 UMAP 座標 CSV 快速重畫 UMAP 圖，套用與 plot_km_poster.py 一致的
PHENOTYPE_LABELS（確定 28 天死亡率排序後才設定）。

不需重跑 KMeans 或 Transformer 模型，只讀取兩個 CSV：
  - umap_coords_{tag}.csv   ← extubation_failure_phenotyping.py 產生
  - （選用）cluster_assignments.csv

【執行步驟】
  1. 先跑 plot_km_outcomes_by_phenotype.py，確認各群 28 天死亡率排序
  2. 依排序更新下方 PHENOTYPE_LABELS / PHENOTYPE_COLORS / PLOT_ORDER
  3. 執行此腳本

【執行指令】（<EXTUBATION_PROJECT_ROOT> 為佔位符，請先設定好環境變數，見 .env.example）
  python "<EXTUBATION_PROJECT_ROOT>/clustering/replot_umap_with_labels.py"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


# =====================================================================
# Paths
# =====================================================================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
BASE        = rf"{EXTUBATION_ROOT}"
COORDS_CSV  = rf"{BASE}\results\phenotyping\umap_coords_last_embOnly.csv"
OUTPUT_DIR  = rf"{BASE}\results\phenotyping"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================================
# UMAP 圖設計：顯示 KMeans 原始 cluster 編號（C0–C3）
#
# 說明：
#   UMAP 呈現非監督學習的分群結果，legend 使用 K-means 輸出的原始 cluster 編號。
#   臨床表型標籤（Phenotype 1–4 / Critical / High Risk...）在 KM 存活曲線圖呈現。
#   兩張圖使用相同顏色，讀者可透過顏色跨圖對應。
#
# 顏色對應（與 plot_km_poster.py 完全一致）：
#   C0（72.8% mortality）→ orange → Phenotype 2 (High Risk)
#   C1（83.3% mortality）→ red    → Phenotype 1 (Critical)
#   C2（50.0% mortality）→ blue   → Phenotype 3 (Moderate Risk)
#   C3（40.2% mortality）→ green  → Phenotype 4 (Lower Risk)
# =====================================================================
PHENOTYPE_LABELS = {
    0: "C0",   # KMeans C0 → 28d mortality 72.8% (→ Phenotype 2, High Risk)
    1: "C1",   # KMeans C1 → 28d mortality 83.3% (→ Phenotype 1, Critical)
    2: "C2",   # KMeans C2 → 28d mortality 50.0% (→ Phenotype 3, Moderate)
    3: "C3",   # KMeans C3 → 28d mortality 40.2% (→ Phenotype 4, Lower Risk)
}

# 顏色與 plot_km_poster.py / KM 存活曲線完全一致（讀者可跨圖對應）
PHENOTYPE_COLORS = {
    0: "#e67e22",   # orange — C0 (→ Phenotype 2, High Risk)
    1: "#c0392b",   # red    — C1 (→ Phenotype 1, Critical)
    2: "#2980b9",   # blue   — C2 (→ Phenotype 3, Moderate)
    3: "#27ae60",   # green  — C3 (→ Phenotype 4, Lower Risk)
}

# 圖例顯示順序（依 cluster 編號 C0–C3）
PLOT_ORDER = [0, 1, 2, 3]

# =====================================================================
# Load UMAP coordinates
# =====================================================================
df = pd.read_csv(COORDS_CSV)
df["cluster"] = df["cluster"].astype(int)

print(f"[Info] Loaded {len(df)} patients from {COORDS_CSV}")
print(f"[Info] Cluster distribution:\n{df['cluster'].value_counts().sort_index()}\n")

# =====================================================================
# Plot
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 7))

for cid in PLOT_ORDER:
    sub = df[df["cluster"] == cid]
    if len(sub) == 0:
        continue
    ax.scatter(sub["umap_1"], sub["umap_2"],
               c=PHENOTYPE_COLORS.get(cid, "grey"),
               s=50, alpha=0.7, edgecolors="k", linewidths=0.3,
               label=PHENOTYPE_LABELS.get(cid, f"Cluster {cid}"))

# Legend（與 PHENOTYPE_LABELS / PHENOTYPE_COLORS 完全一致）
patches = [
    mpatches.Patch(
        color=PHENOTYPE_COLORS.get(cid, "grey"),
        label=f"{PHENOTYPE_LABELS.get(cid, f'Cluster {cid}')}  (n={len(df[df['cluster']==cid])})"
    )
    for cid in PLOT_ORDER if cid in df["cluster"].unique()
]
ax.legend(handles=patches, fontsize=10.5, title="K-means Cluster",
          title_fontsize=11, framealpha=0.9)

ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title(
    "UMAP of Extubation Failure Patients\n"
    "(Transformer Encoder Last-step Embedding, K-means Clusters)",
    fontsize=12, fontweight="bold"
)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
plt.tight_layout()

out_png = Path(OUTPUT_DIR) / "phenotype_umap_last_embOnly_labeled.png"
out_pdf = Path(OUTPUT_DIR) / "phenotype_umap_last_embOnly_labeled.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"✓ PNG: {out_png}")
print(f"✓ PDF: {out_pdf}")
