# -*- coding: utf-8 -*-
"""
plot_phenotype_risk_trajectories.py

四個 EF Phenotype 的平均動態風險軌跡比較圖。

流程：
  1. 以相同 seed=42 重建 train/val/test split，fit scaler（與訓練腳本完全一致）
  2. 載入 cluster_assignments.csv（408 位拔管失敗患者，cluster 0-3）
  3. 對每位 cluster 病人，用 prefix_cutoff=0..11 做 12 次推論 → 逐步累積風險軌跡
  4. 按 cluster 分組，計算 mean ± 95% bootstrap CI
  5. 輸出 figure_phenotype_risk_trajectories.png / .pdf

執行指令：
  python plot_phenotype_risk_trajectories.py \
    --data_csv  "...gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --model_pt  ".../results/transformer/best_transformer.pt" \
    --cluster_csv ".../results/phenotyping/cluster_assignments.csv" \
    --output_dir ".../results/phenotyping/figures"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# ── 與訓練腳本完全一致的常數 ────────────────────────────────────────────
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]
DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT",
    "AnionGap", "Lactate", "Glucose", "io_balance",
    "Vasopressor_use", "Hemodialysis_use",
]
MASK_COLS  = [f"mask_{col}" for col in DYNAMIC_COLS]
TARGET     = "Extubation_failure"
SEQ_TIME_BINS = list(range(-52, -4, 4))   # -52, -48, ..., -8  (12 bins)
SEQ_LEN    = len(SEQ_TIME_BINS)            # 12
BINARY_LIKE = ["sex", "Vasopressor_use", "Hemodialysis_use"]

# K-means cluster → Phenotype 編號 & 名稱（依論文 Table 6）
#   C1(n=108) → Phenotype 1 (Critical)
#   C0(n=136) → Phenotype 2 (High Risk)
#   C2(n=62)  → Phenotype 3 (Moderate Risk)
#   C3(n=102) → Phenotype 4 (Low Risk)
PHENOTYPE_NAMES = {
    0: "Phenotype 2 (High Risk)",
    1: "Phenotype 1 (Critical)",
    2: "Phenotype 3 (Moderate Risk)",
    3: "Phenotype 4 (Low Risk)",
}
# 顏色對齊 KM survival 圖與 UMAP 圖
PHENOTYPE_COLORS = {
    0: "#FF7F0E",   # orange   (C0, High Risk)    — matches KM orange
    1: "#D62728",   # red      (C1, Critical)     — matches KM red
    2: "#1F77B4",   # blue     (C2, Moderate)     — matches KM blue
    3: "#2CA02C",   # green    (C3, Low Risk)     — matches KM green
}
PHENOTYPE_MARKERS = {0: "s", 1: "o", 2: "^", 3: "D"}

# 圖例排序：Phenotype 1 → 2 → 3 → 4（cluster 1, 0, 2, 3）
PHENOTYPE_PLOT_ORDER = [1, 0, 2, 3]

# X 軸：各 time bin 的中點（每個 bin 代表 [t, t+4h)，中點 = t+2）
BIN_MIDPOINTS = [tb + 2 for tb in SEQ_TIME_BINS]   # -50, -46, ..., -6
# X 軸刻度：顯示 bin 起點（-52, -48, ..., -8）加上 "Extubation"
BIN_TICK_POSITIONS = list(SEQ_TIME_BINS) + [0]
BIN_TICK_LABELS    = [str(tb) for tb in SEQ_TIME_BINS] + ["Extubation"]


# ══════════════════════════════════════════════════════════════════
# 1. 模型結構（與訓練腳本完全一致）
# ══════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, factor=0.1):
        super().__init__()
        self.factor = factor
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.factor * self.pe[:, :x.size(1), :]


class TransformerRiskModel(nn.Module):
    def __init__(self, dyn_dim=52, stat_dim=4, d_model=64, nhead=4,
                 num_layers=3, dim_ff=128, dropout=0.1,
                 use_causal_mask=False, pe_factor=0.1,
                 pooling="last", train_noise=0.0):
        super().__init__()
        self.use_causal_mask = bool(use_causal_mask)
        self.pooling = pooling
        self.train_noise = float(train_noise)

        self.dyn_proj = nn.Linear(dyn_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=256, factor=pe_factor)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 16), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x_dyn, x_stat, step_present=None, prefix_cutoff=None,
                return_embedding=False):
        B, T, _ = x_dyn.shape
        h = self.dyn_proj(x_dyn)
        h = self.pos(h)

        key_padding_mask = None
        if step_present is not None:
            key_padding_mask = (step_present <= 0.0)
        if prefix_cutoff is not None:
            future_mask = torch.arange(T, device=x_dyn.device).unsqueeze(0) > int(prefix_cutoff)
            key_padding_mask = future_mask if key_padding_mask is None else (key_padding_mask | future_mask)

        if self.use_causal_mask:
            causal = torch.triu(torch.ones(T, T, device=x_dyn.device), diagonal=1).bool()
        else:
            causal = None

        z = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)

        if return_embedding:
            if self.pooling == "mean":
                if step_present is not None:
                    present_mask = (step_present > 0.0).float()
                    if prefix_cutoff is not None:
                        cutoff_mask = (torch.arange(T, device=x_dyn.device).unsqueeze(0) <= int(prefix_cutoff)).float()
                        present_mask = present_mask * cutoff_mask
                    present_mask_3d = present_mask.unsqueeze(-1)
                    pooled = (z * present_mask_3d).sum(dim=1) / present_mask_3d.sum(dim=1).clamp(min=1e-6)
                else:
                    pooled = z.mean(dim=1)
            else:  # last
                if step_present is None:
                    last_idx = torch.full((B,), T - 1, dtype=torch.long, device=x_dyn.device)
                else:
                    present = (step_present > 0.0)
                    if prefix_cutoff is not None:
                        present = present & (torch.arange(T, device=x_dyn.device).unsqueeze(0) <= int(prefix_cutoff))
                    idx = present.float() * torch.arange(T, device=x_dyn.device).unsqueeze(0)
                    last_idx = idx.max(dim=1).values.long()
                pooled = z[torch.arange(B, device=x_dyn.device), last_idx, :]
            return pooled

        # classifier path (last pooling)
        if step_present is None:
            last_idx = torch.full((B,), T - 1, dtype=torch.long, device=x_dyn.device)
        else:
            present = (step_present > 0.0)
            if prefix_cutoff is not None:
                present = present & (torch.arange(T, device=x_dyn.device).unsqueeze(0) <= int(prefix_cutoff))
            idx = present.float() * torch.arange(T, device=x_dyn.device).unsqueeze(0)
            last_idx = idx.max(dim=1).values.long()
        pooled = z[torch.arange(B, device=x_dyn.device), last_idx, :]

        s = self.stat_proj(x_stat)
        return self.classifier(torch.cat([pooled, s], dim=1))


# ══════════════════════════════════════════════════════════════════
# 2. Dataset（與訓練腳本一致）
# ══════════════════════════════════════════════════════════════════
class ExtubationSeqDataset(Dataset):
    def __init__(self, df, stay_ids, scaler, scale_cols, seq_time_bins):
        self.df = df[df["stay_id"].isin(stay_ids)].copy()
        self.seq_time_bins = list(seq_time_bins)
        self.stay_ids = sorted(list(set(stay_ids)))
        self.scaler = scaler
        self.scale_cols = list(scale_cols) if scale_cols is not None else []

        if self.scaler is not None and len(self.scale_cols) > 0 and len(self.df) > 0:
            self.df.loc[:, self.scale_cols] = self.scaler.transform(self.df[self.scale_cols])

        self.samples = []
        for sid in self.stay_ids:
            d = self.df[self.df["stay_id"] == sid]
            if d.empty:
                continue
            y = int(d[TARGET].iloc[0])
            stat = d[STATIC_COLS].iloc[0].copy()
            seq_values = []
            for tb in self.seq_time_bins:
                row = d[d["time_bin"] == tb]
                if row.empty:
                    dyn_filled    = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)
                    mask          = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)
                    step_pres_val = 0.0
                else:
                    if "bin_has_data" in row.columns:
                        dyn_filled    = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                        mask          = row[MASK_COLS].iloc[0].values.astype(np.float32)
                        step_pres_val = float(row["bin_has_data"].iloc[0])
                    else:
                        dyn_raw       = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                        mask          = (~np.isnan(dyn_raw)).astype(np.float32)
                        dyn_filled    = np.nan_to_num(dyn_raw, nan=0.0).astype(np.float32)
                        step_pres_val = float(mask.max())
                seq_values.append(np.concatenate([dyn_filled, mask]))
                step_pres_val = step_pres_val  # keep for step_present

            # rebuild step_present
            step_present_arr = []
            for tb in self.seq_time_bins:
                row = d[d["time_bin"] == tb]
                if row.empty:
                    step_present_arr.append(0.0)
                elif "bin_has_data" in row.columns:
                    step_present_arr.append(float(row["bin_has_data"].iloc[0]))
                else:
                    dyn_raw = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                    step_present_arr.append(float((~np.isnan(dyn_raw)).max()))

            self.samples.append({
                "sid": int(sid),
                "y": y,
                "x_dyn": np.stack(seq_values, axis=0).astype(np.float32),  # (12, 52)
                "x_stat": stat.values.astype(np.float32),
                "step_present": np.array(step_present_arr, dtype=np.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["x_dyn"],        dtype=torch.float32),
            torch.tensor(s["x_stat"],       dtype=torch.float32),
            torch.tensor(s["y"],            dtype=torch.float32),
            torch.tensor(s["step_present"], dtype=torch.float32),
            s["sid"],
        )


# ══════════════════════════════════════════════════════════════════
# 3. 主流程
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",    type=str, required=True)
    p.add_argument("--model_pt",    type=str, required=True)
    p.add_argument("--cluster_csv", type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="results/phenotyping/figures")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n_boot",      type=int, default=2000,
                   help="Bootstrap resamples for 95% CI")
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--nhead",       type=int,   default=4)
    p.add_argument("--num_layers",  type=int,   default=3)
    p.add_argument("--dim_ff",      type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--pe_factor",   type=float, default=1.0)
    p.add_argument("--pooling",     type=str,   default="last")
    return p.parse_args()


def build_scaler(df, seed=42):
    """以相同 split 重建 scaler，確保與訓練一致。"""
    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    y = stay_labels[TARGET].values
    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values, test_size=0.3,
        random_state=seed, stratify=y)
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    y_temp = temp_labels[TARGET].values
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values, test_size=0.5,
        random_state=seed, stratify=y_temp)

    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
                  if c not in BINARY_LIKE]
    train_df = df[df["stay_id"].isin(train_ids)]
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])
    return scaler, scale_cols, list(test_ids)


def get_trajectory(model, sample, device):
    """對單一病人跑 prefix_cutoff=0..11，回傳 12 個風險機率。"""
    x_dyn = torch.tensor(sample["x_dyn"], dtype=torch.float32).unsqueeze(0).to(device)
    x_stat = torch.tensor(sample["x_stat"], dtype=torch.float32).unsqueeze(0).to(device)
    step_present = torch.tensor(sample["step_present"], dtype=torch.float32).unsqueeze(0).to(device)

    probs = []
    model.eval()
    with torch.no_grad():
        for k in range(SEQ_LEN):
            logit = model(x_dyn, x_stat, step_present=step_present, prefix_cutoff=k)
            probs.append(float(torch.sigmoid(logit).cpu().item()))
    return np.array(probs)   # shape (12,)


def bootstrap_ci(matrix, n_boot=2000, seed=42, alpha=0.05):
    """
    matrix: (n_patients, 12) ndarray
    回傳 mean (12,), lo (12,), hi (12,)
    """
    rng = np.random.RandomState(seed)
    n = matrix.shape[0]
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(matrix[idx].mean(axis=0))
    boot_means = np.array(boot_means)   # (n_boot, 12)
    lo = np.percentile(boot_means, 100 * alpha / 2,     axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha/2), axis=0)
    mean = matrix.mean(axis=0)
    return mean, lo, hi


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. 載入資料 ────────────────────────────────────────────────
    print("Loading feature CSV ...")
    df = pd.read_csv(args.data_csv)
    print(f"  Rows={len(df):,}  |  stay_ids={df['stay_id'].nunique():,}")

    cluster_df = pd.read_csv(args.cluster_csv)
    print(f"  Cluster patients: {len(cluster_df)}  (clusters: {sorted(cluster_df['cluster'].unique())})")

    # ── 2. 重建 scaler（與訓練完全一致） ──────────────────────────
    print("Rebuilding scaler from train split (seed=42) ...")
    scaler, scale_cols, test_ids = build_scaler(df, seed=args.seed)

    # 確認 cluster 病人全在 test set
    cluster_ids = set(cluster_df["stay_id"].tolist())
    in_test = cluster_ids & set(test_ids)
    print(f"  Cluster patients in test set: {len(in_test)} / {len(cluster_ids)}")
    if len(in_test) < len(cluster_ids):
        missing = cluster_ids - in_test
        print(f"  [WARN] {len(missing)} cluster patients NOT in test set. Will still process.")

    # ── 3. 建立 Dataset（只包含 cluster 病人） ────────────────────
    print("Building dataset for cluster patients ...")
    ds = ExtubationSeqDataset(df, list(cluster_ids), scaler, scale_cols, SEQ_TIME_BINS)
    sid2sample = {s["sid"]: s for s in ds.samples}
    print(f"  Dataset samples: {len(ds.samples)}")

    # ── 4. 載入模型 ────────────────────────────────────────────────
    print(f"Loading model: {args.model_pt}")
    model = TransformerRiskModel(
        dyn_dim=len(DYNAMIC_COLS) * 2,   # 26 vars + 26 masks = 52
        stat_dim=len(STATIC_COLS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        pe_factor=args.pe_factor,
        pooling=args.pooling,
    ).to(device)
    ckpt = torch.load(args.model_pt, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("  Model loaded.")

    # ── 5. 計算每位病人的 12-step 軌跡 ────────────────────────────
    print("Computing trajectories (prefix inference) ...")
    cluster_trajectories = {c: [] for c in sorted(cluster_df["cluster"].unique())}

    for _, row in cluster_df.iterrows():
        sid = int(row["stay_id"])
        c   = int(row["cluster"])
        if sid not in sid2sample:
            print(f"  [WARN] stay_id={sid} not in dataset, skipping.")
            continue
        traj = get_trajectory(model, sid2sample[sid], device)
        cluster_trajectories[c].append(traj)

    for c, trajs in cluster_trajectories.items():
        print(f"  Cluster {c}: {len(trajs)} patients")

    # ── 6. Bootstrap CI ─────────────────────────────────────────────
    print(f"Bootstrap CI (n_boot={args.n_boot}) ...")
    results = {}
    for c, trajs in cluster_trajectories.items():
        if len(trajs) == 0:
            continue
        mat = np.array(trajs)   # (n, 12)
        mean, lo, hi = bootstrap_ci(mat, n_boot=args.n_boot, seed=args.seed)
        results[c] = {"mean": mean, "lo": lo, "hi": hi, "n": len(trajs)}
        print(f"  Cluster {c} ({PHENOTYPE_NAMES[c]}): "
              f"mean_final={mean[-1]:.3f} ({lo[-1]:.3f}–{hi[-1]:.3f})")

    # ── 7. 繪圖 ─────────────────────────────────────────────────────
    print("Plotting ...")
    x_mid = np.array(BIN_MIDPOINTS)   # -50, -46, ..., -6（bin 中點，資料點位置）

    fig, ax = plt.subplots(figsize=(11, 6))

    for c in PHENOTYPE_PLOT_ORDER:
        if c not in results:
            continue
        res    = results[c]
        name   = PHENOTYPE_NAMES[c]
        color  = PHENOTYPE_COLORS[c]
        marker = PHENOTYPE_MARKERS[c]
        n      = res["n"]
        mean, lo, hi = res["mean"], res["lo"], res["hi"]

        ax.plot(x_mid, mean, color=color, marker=marker,
                linewidth=2.2, markersize=6, label=f"{name}  (n={n})", zorder=3)
        ax.fill_between(x_mid, lo, hi, color=color, alpha=0.15, zorder=2)

    # 決策閾值虛線
    ax.axhline(y=0.4559, color="#888888", linestyle="--", linewidth=1.3,
               label="Decision threshold = 0.456", zorder=1)

    # 垂直虛線標示 Extubation 時間點（x=0）
    ax.axvline(x=0, color="#555555", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(0.5, 0.97, "Extubation", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9, color="#555555")

    ax.set_xlabel("Hours before extubation  (each point = midpoint of 4h window)",
                  fontsize=12)
    ax.set_ylabel("Predicted extubation failure probability", fontsize=12)
    ax.set_title(
        "Dynamic Risk Trajectories by EF Phenotype\n"
        "(Mean ± 95% Bootstrap CI, Transformer model)",
        fontsize=13, fontweight="bold"
    )

    # X 軸刻度：顯示 bin 起點 + Extubation
    ax.set_xticks(BIN_TICK_POSITIONS)
    ax.set_xticklabels(BIN_TICK_LABELS, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-54, 3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    png_path = os.path.join(args.output_dir, "figure_phenotype_risk_trajectories.png")
    pdf_path = os.path.join(args.output_dir, "figure_phenotype_risk_trajectories.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")

    # ── 8. 儲存數值表（供論文 supplementary 使用）──────────────────
    rows = []
    for c in PHENOTYPE_PLOT_ORDER:
        if c not in results:
            continue
        res = results[c]
        for i, tb in enumerate(SEQ_TIME_BINS):
            rows.append({
                "cluster": c,
                "phenotype": PHENOTYPE_NAMES[c],
                "time_bin": tb,
                "mean": res["mean"][i],
                "ci_lo": res["lo"][i],
                "ci_hi": res["hi"][i],
                "n": res["n"],
            })
    csv_path = os.path.join(args.output_dir, "figure_phenotype_trajectories_data.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    print("\n完成！")


if __name__ == "__main__":
    main()
