#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extubation_failure_phenotyping.py

Extubation failure phenotyping based on Transformer latent representations.

完整功能：
1. Transformer Latent Embedding 提取（Encoder last-step hidden state）
2. 自動 K 值搜尋：k=2~6，產出 Silhouette / DB Index 趨勢圖
3. Clustering 穩定性驗證：Multi-seed ARI + Bootstrap Silhouette
4. 臨床剖析：描述統計（mean±SD / %）+ Kruskal-Wallis + BH FDR 校正
5. 視覺化：UMAP 散佈圖 + Z-score 正規化特徵熱圖

設計決策：
【分群依據】
  - 只對 test set 中的 extubation failure 病人做分群
  - 分群特徵：Transformer Encoder 的 last-step hidden state（pooling=last）
    每位病患一個 d_model 維向量，代表模型從完整 48h 時序軌跡壓縮出的
    latent representation，不包含靜態特徵（age/BMI/Charlson）
  - 靜態特徵不參與分群：本模型為 Late Fusion 架構，Transformer Encoder
    只輸入動態特徵，靜態特徵（age/BMI/Charlson）由獨立的 Static MLP 分流，
    在 encoder 輸出後才融合進入分類頭。因此 encoder hidden state 學到的是
    「在靜態特徵已分流處理的前提下，動態軌跡本身提供的互補資訊」。
    以此做分群，探索的是不同的時序失敗機制，而非靜態人口學差異。
    若事後 profiling 發現各群靜態特徵仍有顯著差異，則反映模型隱含地從
    動態軌跡中捕捉到了與人口學相關的臨床資訊（具科學意義的發現）

【臨床剖析（事後描述，不影響分群）】
  - 分群完成後，用原始臨床數值（非 Embedding）對各群進行描述
  - 各臨床特徵以病人在所有 12 個 time bin 的平均值為代表
  - 靜態特徵（age/BMI/Charlson）此時才加入，作為群間差異的描述指標
  - Kruskal-Wallis + BH FDR 校正，檢驗各特徵的群間差異是否顯著

【可選：加入靜態特徵做分群（--use_static_in_cluster 1）】
  - 若研究問題為「結合人口學與時序資訊的表型分析」可啟用
  - 預設關閉（推薦），以保持 latent representation 分群的可解釋性
"""

import os

# Fix: Windows threadpoolctl / MKL DLL 相容性問題（OSError 0xc06d007f）
# 必須在 import sklearn 之前設定
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["NUMEXPR_NUM_THREADS"]   = "1"

import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── Fix import path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model training"))

from transformer_pre_extubation_risk_trajectory import (
    ExtubationTransformer, ExtubationSeqDataset,
    STATIC_COLS, DYNAMIC_COLS, TARGET, SEQ_TIME_BINS, split_by_stay_id
)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from scipy.stats import kruskal, zscore as sp_zscore
from statsmodels.stats.multitest import multipletests
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

CLUSTER_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
BINARY_COLS = {"Vasopressor_use", "Hemodialysis_use", "sex"}

# 臨床剖析特徵（完整 30 個：26 動態模型特徵 + 4 靜態特徵）
# 對應 transformer_pre_extubation_risk_trajectory.py 的 DYNAMIC_COLS + STATIC_COLS
PROFILE_FEATURES = {
    "Demographics":  ["age", "sex", "BMI", "Charlson_Score"],
    "Vitals":        ["heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS"],
    "Ventilator":    ["FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day"],
    "Blood Gas":     ["pH", "PaO2", "PaCO2", "BE", "OI"],
    "Labs":          ["Cr", "WBC", "Hb", "PLT", "AnionGap", "Lactate", "Glucose"],
    "Fluid/Therapy": ["io_balance", "Vasopressor_use", "Hemodialysis_use"],
}
ALL_PROFILE_COLS = [c for grp in PROFILE_FEATURES.values() for c in grp]


# =========================
# 1. 參數解析
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Extubation Failure Phenotyping — test set only")
    p.add_argument("--data_csv",    type=str, required=True)
    p.add_argument("--model_path",  type=str, required=True)
    p.add_argument("--output_dir",  type=str, required=True)
    # Model architecture（需與訓練時一致）
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--nhead",       type=int,   default=4)
    p.add_argument("--num_layers",  type=int,   default=3)
    p.add_argument("--dim_ff",      type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--pe_factor",   type=float, default=1.0)
    # Clustering
    p.add_argument("--n_clusters",   type=int,   default=4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n_runs",       type=int,   default=20)
    p.add_argument("--n_bootstrap",  type=int,   default=10)
    p.add_argument("--embed_pooling", type=str,  default="last",
                   choices=["last", "mean", "all"],
                   help="Transformer encoder output pooling 方式：\n"
                        "  last = 最後一個有效時間步（與訓練一致）\n"
                        "  mean = 所有有效時間步平均（捕捉整條軌跡）\n"
                        "  all  = 所有時間步 concat（最豐富，自動做 PCA 降維）")
    p.add_argument("--pca_dim",       type=int,  default=32,
                   help="embed_pooling=all 時的 PCA 目標維度（預設 32）")
    p.add_argument("--embed_source",  type=str,  default="encoder",
                   choices=["encoder", "fused"],
                   help="分群所用的 latent representation 來源：\n"
                        "  encoder = 只用 Transformer Encoder 輸出（預設，d_model 維）\n"
                        "            → 純動態軌跡的 latent space\n"
                        "  fused   = Encoder 輸出 + Static MLP 輸出（d_model+16 維）\n"
                        "            → Late Fusion 後的完整模型 latent space，\n"
                        "              與模型做最終分類決策時使用的資訊完全一致")
    p.add_argument("--use_static_in_cluster", type=int, default=0,
                   choices=[0, 1],
                   help="分群時是否直接加入原始靜態特徵（age/BMI/Charlson）：\n"
                        "  0 = 不加入原始靜態特徵（預設）\n"
                        "  1 = Embedding + 原始靜態特徵（注意：與 --embed_source fused 不同，\n"
                        "      fused 使用的是 Static MLP 的輸出，而非原始數值）")
    return p.parse_args()


# =========================
# 2. 提取分群特徵
# =========================
def extract_combined_features(model, dataloader, device,
                               pooling="last",
                               embed_source="encoder",
                               use_static=False):
    """
    從 Transformer 提取 patient representation，供分群使用。

    ── pooling（動態流 Encoder 輸出的時間步聚合方式）──
      last  → 最後一個有效時間步的 hidden state  (B, d_model)
      mean  → 所有有效時間步的加權平均            (B, d_model)
      all   → 所有時間步 flatten concat          (B, T * d_model)  ← 後續自動 PCA

    ── embed_source（分群所用的 latent representation 來源）──
      encoder → 只用 Transformer Encoder 輸出（d_model 維）
                研究問題：「模型從動態軌跡學到了什麼？」

      fused   → Encoder 輸出 + Static MLP 輸出（d_model + 16 維）
                完整 Late Fusion 後的 representation，與模型做最終分類決策
                時使用的輸入資訊完全一致。
                研究問題：「模型整合動態軌跡與靜態特徵後形成了怎樣的 latent space？」

    ── use_static（直接加入原始靜態數值，舊版選項）──
      False → 不加原始靜態特徵（預設）
      True  → Embedding + 原始 age/BMI/Charlson 數值

    回傳：embeddings (N, dim)、labels (N,)、stay_ids (N,)、static_arr (N, stat_dim)
         ※ static_arr 永遠回傳原始靜態特徵，供 profile_clusters 事後剖析使用
    """
    model.eval()
    all_embeds, all_stat, all_labels, all_sids = [], [], [], []

    with torch.no_grad():
        for x_dyn, x_stat, y, step_present, sid in dataloader:
            x_dyn  = x_dyn.to(device)
            x_stat = x_stat.to(device)
            B, T, _ = x_dyn.shape

            # ── 動態流：Encoder ──────────────────────────────────────
            h = model.dyn_proj(x_dyn)
            h = model.pos(h)
            key_pad = (step_present <= 0.0).to(device)
            z = model.encoder(h, src_key_padding_mask=key_pad)   # (B, T, d_model)

            valid = (step_present > 0.0).float().to(device)      # (B, T)

            if pooling == "last":
                idx    = (valid * torch.arange(T, device=device).unsqueeze(0)).max(dim=1).values.long()
                pooled = z[torch.arange(B, device=device), idx, :]         # (B, d_model)
            elif pooling == "mean":
                valid_3d = valid.unsqueeze(-1)
                pooled   = (z * valid_3d).sum(dim=1) / valid_3d.sum(dim=1).clamp(min=1e-6)
            else:  # "all"
                z_masked = z * valid.unsqueeze(-1)
                pooled   = z_masked.reshape(B, -1)                         # (B, T*d_model)

            # ── embed_source 選擇 ──────────────────────────────────────
            if embed_source == "fused":
                # Static MLP 輸出（16 維），與訓練時 late fusion 完全一致
                s = model.stat_proj(x_stat)                                # (B, 16)
                embed = torch.cat([pooled, s], dim=-1)                     # (B, d_model+16)
            else:
                # 只用 Encoder 輸出（純動態軌跡 latent representation）
                embed = pooled                                              # (B, d_model)

            # ── use_static（原始靜態數值，舊版選項）──────────────────
            if use_static:
                embed = torch.cat([embed, x_stat], dim=-1)

            all_embeds.append(embed.cpu().numpy())
            all_stat.append(x_stat.cpu().numpy())
            all_labels.append(y.cpu().numpy().reshape(-1))
            if isinstance(sid, torch.Tensor):
                all_sids.extend(sid.cpu().tolist())
            else:
                all_sids.extend(list(sid))

    embeds_arr = np.vstack(all_embeds)   # (N, dim)
    static_arr = np.vstack(all_stat)     # (N, stat_dim) — 永遠保留供 profiling 用

    return embeds_arr, np.concatenate(all_labels), np.array(all_sids), static_arr


# =========================
# 3. K 值最佳化
# =========================
def find_optimal_k(features, output_dir, seed, k_max=6):
    results = []
    print(f"\n[Optimizing] Testing k=2 to {k_max}...")
    for k in range(2, k_max + 1):
        km  = KMeans(n_clusters=k, random_state=seed, n_init=50).fit(features)
        sil = silhouette_score(features, km.labels_)
        db  = davies_bouldin_score(features, km.labels_)
        ch  = calinski_harabasz_score(features, km.labels_)
        results.append({"k": k, "Silhouette": round(sil, 4),
                         "DB_Index": round(db, 4), "CH_Index": round(ch, 1)})
        print(f"  k={k}: Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")

    df_k = pd.DataFrame(results)
    df_k.to_csv(os.path.join(output_dir, "k_optimization.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(df_k["k"], df_k["Silhouette"], "bo-", lw=2, ms=7, label="Silhouette ↑")
    ax2.plot(df_k["k"], df_k["DB_Index"],   "rs-", lw=2, ms=7, label="DB Index ↓")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Silhouette Score", color="b", fontsize=12)
    ax2.set_ylabel("Davies-Bouldin Index", color="r", fontsize=12)
    ax1.tick_params(axis="y", colors="b")
    ax2.tick_params(axis="y", colors="r")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.title("Clustering Optimization (Dynamic Embedding + Static Fusion)",
              fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_optimization_k.png"), dpi=300)
    plt.close()
    print(f"✓ K 值優化圖已儲存")
    return df_k


# =========================
# 3.5. Clustering 穩定性驗證
# =========================
def validate_cluster_stability(features, n_clusters, output_dir,
                                n_runs=20, n_bootstrap=10, seed=42):
    print(f"\n[Stability] 開始驗證 k={n_clusters} 穩定性...")
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000, size=n_runs).tolist()

    ref_km     = KMeans(n_clusters=n_clusters, random_state=seeds[0], n_init=50).fit(features)
    ref_labels = ref_km.labels_

    ari_scores, sil_scores = [], []
    for s in seeds[1:]:
        km  = KMeans(n_clusters=n_clusters, random_state=s, n_init=50).fit(features)
        ari = adjusted_rand_score(ref_labels, km.labels_)
        sil = silhouette_score(features, km.labels_)
        ari_scores.append(float(ari))
        sil_scores.append(float(sil))

    boot_sil = []
    n = len(features)
    n_sub = max(int(n * 0.8), n_clusters + 1)
    for _ in range(n_bootstrap):
        bs  = int(rng.integers(0, 10000))
        idx = np.random.default_rng(bs).choice(n, size=n_sub, replace=True)
        try:
            km_b = KMeans(n_clusters=n_clusters, random_state=bs, n_init=20).fit(features[idx])
            if len(np.unique(km_b.labels_)) == n_clusters:
                boot_sil.append(float(silhouette_score(features[idx], km_b.labels_)))
        except Exception:
            pass

    ari_mean  = float(np.mean(ari_scores))  if ari_scores else float("nan")
    ari_std   = float(np.std(ari_scores))   if ari_scores else float("nan")
    sil_mean  = float(np.mean(sil_scores))  if sil_scores else float("nan")
    sil_std   = float(np.std(sil_scores))   if sil_scores else float("nan")
    bsil_mean = float(np.mean(boot_sil))    if boot_sil   else float("nan")
    bsil_std  = float(np.std(boot_sil))     if boot_sil   else float("nan")

    print(f"  [Multi-Seed ARI] mean={ari_mean:.4f} ± {ari_std:.4f}  (n={len(ari_scores)})")
    print(f"  [Multi-Seed Sil] mean={sil_mean:.4f} ± {sil_std:.4f}")
    print(f"  [Bootstrap  Sil] mean={bsil_mean:.4f} ± {bsil_std:.4f}  (n={len(boot_sil)})")

    rows = ([{"run_type": "multi_seed", "run_id": i+1, "ARI": a, "Silhouette": s}
              for i, (a, s) in enumerate(zip(ari_scores, sil_scores))] +
            [{"run_type": "bootstrap", "run_id": i+1, "ARI": float("nan"), "Silhouette": s}
              for i, s in enumerate(boot_sil)])
    df_stab = pd.DataFrame(rows)
    df_stab.to_csv(os.path.join(output_dir, "cluster_stability_report.csv"),
                   index=False, encoding="utf-8-sig")

    # Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    seed_df = df_stab[df_stab["run_type"] == "multi_seed"]
    boot_df = df_stab[df_stab["run_type"] == "bootstrap"]

    axes[0].boxplot([seed_df["ARI"].dropna().tolist()], labels=[f"k={n_clusters}"])
    axes[0].axhline(0.9, color="red", lw=1.5, ls="--", alpha=0.7, label="ARI=0.9")
    axes[0].set_title(f"Multi-Seed ARI  (n={len(ari_scores)})", fontweight="bold")
    axes[0].set_ylabel("Adjusted Rand Index (ARI)")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3, ls="--")
    axes[0].legend()

    grp = {"multi-seed": seed_df["Silhouette"].dropna().tolist(),
           "bootstrap":  boot_df["Silhouette"].dropna().tolist()}
    axes[1].boxplot(list(grp.values()), labels=list(grp.keys()))
    axes[1].set_title(f"Silhouette Stability  (k={n_clusters})", fontweight="bold")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(alpha=0.3, ls="--")

    plt.suptitle(f"Cluster Stability Validation  (k={n_clusters})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_stability_boxplot.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 穩定性驗證完成")

    return {"n_clusters": n_clusters,
            "ari_mean": ari_mean, "ari_std": ari_std,
            "seed_sil_mean": sil_mean, "seed_sil_std": sil_std,
            "bootstrap_sil_mean": bsil_mean, "bootstrap_sil_std": bsil_std,
            "n_ari_runs": len(ari_scores), "n_bootstrap_runs": len(boot_sil)}


# =========================
# 4. 臨床剖析 + Kruskal-Wallis
# =========================
def profile_clusters(cluster_labels, stay_ids, df_original, output_dir):
    """
    以每位病人在所有 time bin 的均值作為代表值（非只取最後一筆），
    進行描述統計 + Kruskal-Wallis 群間差異檢定（BH FDR 校正）。
    回傳：df_sum（展示用）、df_heatmap（純數值，供熱圖用）、df_stat（KW 結果）
    """
    # 取各 stay_id 在所有 time bin 的平均
    avail_cols = [c for c in ALL_PROFILE_COLS if c in df_original.columns]
    df_mean = df_original.groupby("stay_id")[avail_cols].mean().reset_index()

    df_map    = pd.DataFrame({"stay_id": stay_ids.tolist(),
                               "cluster": cluster_labels.astype(int)})
    df_merged = df_map.merge(df_mean, on="stay_id", how="left")

    # ── 描述統計 ──
    summary_rows = []
    for cid in sorted(df_merged["cluster"].unique()):
        sub = df_merged[df_merged["cluster"] == cid]
        row = {"Cluster": f"C{cid}", "N": len(sub)}
        for col in avail_cols:
            v = sub[col].dropna()
            if len(v) == 0:
                row[col] = "N/A"
            elif col in BINARY_COLS:
                row[col] = f"{100 * v.mean():.1f}%"
            else:
                row[col] = f"{v.mean():.2f} ± {v.std():.2f}"
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(output_dir, "phenotype_summary.csv"),
                  index=False, encoding="utf-8-sig")

    # ── 純數值矩陣（供 Heatmap 用，排除二元欄位）──
    numeric_cols = [c for c in avail_cols if c not in BINARY_COLS]
    heatmap_rows = []
    for cid in sorted(df_merged["cluster"].unique()):
        sub = df_merged[df_merged["cluster"] == cid]
        heatmap_rows.append({c: float(sub[c].mean()) for c in numeric_cols})
    df_heatmap = pd.DataFrame(heatmap_rows,
                               index=[f"C{c}" for c in sorted(df_merged["cluster"].unique())])

    # ── Kruskal-Wallis + BH FDR 校正 ──
    stat_rows = []
    for col in avail_cols:
        groups = [df_merged.loc[df_merged["cluster"] == cid, col].dropna().values
                  for cid in sorted(df_merged["cluster"].unique())]
        try:
            if all(len(g) > 0 for g in groups):
                H, p = kruskal(*groups)
            else:
                H, p = float("nan"), float("nan")
        except Exception:
            H, p = float("nan"), float("nan")
        stat_rows.append({"Feature": col, "H_stat": round(H, 3) if not np.isnan(H) else float("nan"),
                           "p_value": p})

    df_stat = pd.DataFrame(stat_rows)
    valid_idx = df_stat["p_value"].notna()
    if valid_idx.sum() > 0:
        _, p_adj, _, _ = multipletests(df_stat.loc[valid_idx, "p_value"].values,
                                       method="fdr_bh")
        df_stat.loc[valid_idx, "p_adj_BH"] = p_adj

    def sig_label(x):
        if pd.isna(x):
            return "n/a"
        return "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns"))

    df_stat["Significant"] = df_stat.get("p_adj_BH",
                                          pd.Series([float("nan")] * len(df_stat))).apply(sig_label)
    df_stat.sort_values("p_adj_BH", inplace=True, ignore_index=True)
    df_stat.to_csv(os.path.join(output_dir, "kruskal_wallis_results.csv"),
                   index=False, encoding="utf-8-sig")

    if "p_adj_BH" in df_stat.columns:
        sig_count = int((df_stat["p_adj_BH"] < 0.05).sum())
    else:
        sig_count = 0
    print(f"\n[Kruskal-Wallis] 顯著差異特徵（FDR < 0.05）：{sig_count} / {len(df_stat)}")
    if sig_count > 0 and "p_adj_BH" in df_stat.columns:
        print(df_stat[df_stat["p_adj_BH"] < 0.05][
            ["Feature", "H_stat", "p_value", "p_adj_BH", "Significant"]
        ].to_string(index=False))

    return df_sum, df_heatmap, df_stat


# =========================
# 5. 視覺化
# =========================
def plot_umap(features, cluster_labels, output_dir,
              seed=42, pooling_tag="last",
              embed_source="encoder", use_static=False,
              phenotype_labels=None, phenotype_colors=None):
    """
    phenotype_labels : dict, e.g. {1: "Critical", 0: "High Risk", ...}
                       若為 None，使用預設 "Cluster {id}" 標籤
    phenotype_colors : dict, e.g. {1: "#c0392b", ...}
                       若為 None，使用預設色盤

    UMAP 2D 座標會同步儲存為 umap_coords_{fname_tag}.csv，
    供後續以 replot_umap_with_labels.py 快速重畫（不需重跑模型）。
    """
    print("\n[UMAP] 計算降維中...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, random_state=seed)
    coords  = reducer.fit_transform(features)

    # ── 儲存 UMAP 座標（供後續重畫用）──────────────────────────────
    umap_df = pd.DataFrame({
        "umap_1":  coords[:, 0],
        "umap_2":  coords[:, 1],
        "cluster": cluster_labels.astype(int),
    })

    # 決定檔名 tag
    if embed_source == "fused":
        feat_tag  = "Late-Fused (Encoder + StaticMLP)"
        fname_tag = f"{pooling_tag}_lateFused"
    elif use_static:
        feat_tag  = "Encoder + Raw Static"
        fname_tag = f"{pooling_tag}_wStatic"
    else:
        feat_tag  = "Encoder only"
        fname_tag = f"{pooling_tag}_embOnly"

    coords_csv = os.path.join(output_dir, f"umap_coords_{fname_tag}.csv")
    umap_df.to_csv(coords_csv, index=False)
    print(f"✓ UMAP 座標已儲存：{coords_csv}（可供重畫用）")

    # ── 顏色與標籤 ──────────────────────────────────────────────────
    unique_cids = sorted(np.unique(cluster_labels).tolist())

    if phenotype_colors is not None:
        color_map = {cid: phenotype_colors.get(cid, CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])
                     for i, cid in enumerate(unique_cids)}
    else:
        color_map = {cid: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                     for i, cid in enumerate(unique_cids)}

    if phenotype_labels is not None:
        label_map = {cid: phenotype_labels.get(cid, f"Cluster {cid}")
                     for cid in unique_cids}
    else:
        label_map = {cid: f"Cluster {cid}" for cid in unique_cids}

    colors = [color_map[c] for c in cluster_labels]

    # ── 繪圖 ────────────────────────────────────────────────────────
    pooling_label = {"last": "Last-step", "mean": "Mean", "all": "All-steps (PCA)"}
    subtitle = f"Transformer ({pooling_label.get(pooling_tag, pooling_tag)}) — {feat_tag}"

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=50, alpha=0.7, edgecolors="k", linewidths=0.3)
    patches = [mpatches.Patch(color=color_map[cid], label=label_map[cid])
               for cid in unique_cids]
    ax.legend(handles=patches, fontsize=11, title="Phenotype", title_fontsize=11)
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(f"Patient Phenotypes — UMAP\n({subtitle})",
                 fontsize=13, fontweight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"phenotype_umap_{fname_tag}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ UMAP 圖已儲存（phenotype_umap_{fname_tag}.png）")


def plot_heatmap(df_heatmap, output_dir):
    """Z-score 正規化特徵熱圖（只用純數值欄位，不含 binary 欄位）。"""
    data_mat = df_heatmap.values.astype(float)
    with np.errstate(invalid="ignore"):
        data_norm = sp_zscore(data_mat, axis=0)
    data_norm = np.nan_to_num(data_norm, nan=0.0)

    fig, ax = plt.subplots(figsize=(max(12, len(df_heatmap.columns) * 0.7),
                                    max(4, len(df_heatmap) * 1.2)))
    sns.heatmap(data_norm.T,
                xticklabels=df_heatmap.index,
                yticklabels=df_heatmap.columns,
                annot=True, fmt=".2f",
                cmap="RdBu_r", center=0,
                linewidths=0.4, linecolor="grey",
                ax=ax)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Phenotype Clinical Profiles (Z-score normalized)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phenotype_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Heatmap 已儲存")


# =========================
# 6. Main Pipeline
# =========================
def main():
    args   = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 載入資料 ──
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)

    # ── 切分（與 Transformer 訓練完全一致）──
    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.70, seed=args.seed)
    print(f"[Split] train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # ── Scaler：只在 train set fit ──
    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
                  if c in df.columns and c not in {"sex", "Vasopressor_use", "Hemodialysis_use"}]
    scaler = StandardScaler().fit(df[df["stay_id"].isin(train_ids)][scale_cols])

    # ── Test set DataLoader ──
    ds_test = ExtubationSeqDataset(
        df[df["stay_id"].isin(test_ids)], test_ids, scaler, scale_cols, SEQ_TIME_BINS)
    loader  = DataLoader(ds_test, batch_size=256, shuffle=False)

    # ── 載入 Transformer 模型 ──
    model = ExtubationTransformer(
        dyn_dim=52, stat_dim=len(STATIC_COLS),
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_ff=args.dim_ff,
        dropout=args.dropout, pe_factor=args.pe_factor,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"✓ 模型已載入：{args.model_path}")

    # ── 提取分群特徵（test set 全部）──
    use_static = bool(args.use_static_in_cluster)
    embed_source = args.embed_source
    print(f"\n[Embedding] pooling={args.embed_pooling}  |  "
          f"embed_source={embed_source}  |  use_static_in_cluster={use_static}")
    combined, orig_labels, sids, static_arr = extract_combined_features(
        model, loader, device,
        pooling=args.embed_pooling,
        embed_source=embed_source,
        use_static=use_static)

    if embed_source == "fused":
        mode_tag = f"Late-Fused Embedding (Encoder {args.d_model}d + StaticMLP 16d = {combined.shape[1]}d)"
    elif use_static:
        mode_tag = f"Encoder Embedding + Raw Static ({combined.shape[1]}d)"
    else:
        mode_tag = f"Encoder Embedding only ({combined.shape[1]}d)"
    print(f"[Features] 全 test set: {combined.shape[0]} 位，分群維度={combined.shape[1]}  （{mode_tag}）")

    # ── 只保留 Extubation Failure 病人 ──
    fail_mask     = orig_labels == 1
    combined_fail = combined[fail_mask]
    sids_fail     = sids[fail_mask]
    # static_arr 永遠保留完整，供 profile_clusters 事後剖析使用
    print(f"[Filter] Extubation Failure only: {combined_fail.shape[0]} 位")

    if combined_fail.shape[0] < 10:
        raise RuntimeError("Failure 病人數過少（< 10），無法做 clustering。")

    # ── all 模式：PCA 降維 ──
    if args.embed_pooling == "all":
        n_comp = min(args.pca_dim, combined_fail.shape[0] - 1, combined_fail.shape[1])
        pca = PCA(n_components=n_comp, random_state=args.seed)
        combined_fail = pca.fit_transform(combined_fail)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"[PCA] all-mode: {combined_fail.shape[1]} 維 → {n_comp} 維  "
              f"（累積解釋變異 {var_explained:.1%}）")

    # ── K 值最佳化 ──
    find_optimal_k(combined_fail, args.output_dir, args.seed)

    # ── 穩定性驗證 ──
    stability = validate_cluster_stability(
        combined_fail, args.n_clusters, args.output_dir,
        n_runs=args.n_runs, n_bootstrap=args.n_bootstrap, seed=args.seed)
    pd.DataFrame([stability]).to_csv(
        os.path.join(args.output_dir, "cluster_stability_summary.csv"),
        index=False, encoding="utf-8-sig")

    # ── 最終 KMeans ──
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=50)
    cluster_labels = km.fit_predict(combined_fail)
    final_sil = silhouette_score(combined_fail, cluster_labels)
    print(f"\n[KMeans k={args.n_clusters}] Silhouette = {final_sil:.4f}")
    for cid in range(args.n_clusters):
        print(f"  Cluster {cid}: {(cluster_labels == cid).sum()} 位")

    # ── 儲存分群結果 ──
    df_assign = pd.DataFrame({"stay_id": sids_fail.tolist(),
                               "cluster": cluster_labels.astype(int)})
    df_assign.to_csv(os.path.join(args.output_dir, "cluster_assignments.csv"),
                     index=False, encoding="utf-8-sig")
    print("✓ 分群結果已儲存：cluster_assignments.csv")

    # ── 臨床剖析 + Kruskal-Wallis ──
    df_sum, df_heatmap, df_stat = profile_clusters(
        cluster_labels, sids_fail, df, args.output_dir)

    # ── 視覺化 ──
    plot_umap(combined_fail, cluster_labels, args.output_dir,
              seed=args.seed, pooling_tag=args.embed_pooling,
              embed_source=embed_source, use_static=use_static)
    plot_heatmap(df_heatmap, args.output_dir)

    print(f"\n✅ 分析完成，所有結果儲存於: {args.output_dir}")


if __name__ == "__main__":
    # =========================================================================
    # 執行指令（使用主訓練模型，test AUROC=0.8177）
    #
    # 架構參數須與訓練時完全一致：
    #   d_model=64, nhead=4, num_layers=3, dim_ff=128
    #   dropout=0.2, pe_factor=1.0
    #   use_time_weights=0, use_causal_mask=0
    #
    # 模型路徑：
    #   results/transformer/best_transformer.pt
    #
    # 執行指令（<EXTUBATION_PROJECT_ROOT> 為佔位符，請先設定好環境變數，見 .env.example）：
    #
    # python "<EXTUBATION_PROJECT_ROOT>/clustering/extubation_failure_phenotyping.py" --data_csv "<EXTUBATION_PROJECT_ROOT>/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" --model_path "<EXTUBATION_PROJECT_ROOT>/results/transformer/best_transformer.pt" --output_dir "<EXTUBATION_PROJECT_ROOT>/results/phenotyping" --n_clusters 4 --d_model 64 --nhead 4 --num_layers 3 --dim_ff 128 --dropout 0.2 --pe_factor 1.0
    #
    # =========================================================================
    main()
