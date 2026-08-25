# -*- coding: utf-8 -*-
"""
plot_trajectory_inference_only.py

載入已訓練好的 Transformer 模型，對 test set 病人畫動態風險軌跡。
【不重新訓練，只做推論】

用途：
  1. --list_ids         列出 test set 所有可畫軌跡的 stay_id（成功 / EF 分開列出）
  2. --focus_ids        對指定 stay_id(s) 畫個別軌跡圖
  3. --compare          同時畫一位成功（EF=0）與一位失敗（EF=1）的對比圖

執行範例：

  # 列出所有可用 stay_id
  python plot_trajectory_inference_only.py \
    --data_csv   "...gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --model_pt   ".../results/transformer/best_transformer.pt" \
    --output_dir ".../results/trajectory_plots" \
    --list_ids 1

  # 畫單一病人軌跡
  python plot_trajectory_inference_only.py \
    --data_csv   "..." --model_pt "..." --output_dir "..." \
    --focus_ids 30015288

  # 畫多人軌跡
  python plot_trajectory_inference_only.py \
    --data_csv   "..." --model_pt "..." --output_dir "..." \
    --focus_ids 30015288 30042596 30056897

  # 自動選一位 EF=0 + 一位 EF=1，畫對比圖
  python plot_trajectory_inference_only.py \
    --data_csv   "..." --model_pt "..." --output_dir "..." \
    --compare 1

  # 自行指定對比的兩位病人
  python plot_trajectory_inference_only.py \
    --data_csv   "..." --model_pt "..." --output_dir "..." \
    --compare 1 --success_id 30005707 --failure_id 30015288
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# ══════════════════════════════════════════════════════════════
# 0. 與訓練腳本完全一致的常數
# ══════════════════════════════════════════════════════════════
STATIC_COLS   = ["age", "sex", "BMI", "Charlson_Score"]
DYNAMIC_COLS  = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use",
]
MASK_COLS     = [f"mask_{col}" for col in DYNAMIC_COLS]
TARGET        = "Extubation_failure"
SEQ_TIME_BINS = list(range(-52, -4, 4))   # -52, -48, ..., -8  (12 bins)
SEQ_LEN       = len(SEQ_TIME_BINS)
BINARY_LIKE   = ["sex", "Vasopressor_use", "Hemodialysis_use"]
WINDOW_HOURS  = 4

# ══════════════════════════════════════════════════════════════
# 1. 模型定義（與訓練腳本完全一致）
# ══════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, factor=0.1):
        super().__init__()
        self.factor = factor
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.factor * self.pe[:, :x.size(1), :]


class TransformerRiskModel(nn.Module):
    def __init__(self, dyn_dim=52, stat_dim=4, d_model=64, nhead=4,
                 num_layers=3, dim_ff=128, dropout=0.1,
                 use_causal_mask=False, pe_factor=0.1,
                 pooling="last", train_noise=0.0):
        super().__init__()
        self.use_causal_mask = bool(use_causal_mask)
        self.pooling    = pooling
        self.train_noise = float(train_noise)
        self.dyn_proj   = nn.Linear(dyn_dim, d_model)
        self.pos        = SinusoidalPositionalEncoding(d_model, max_len=256, factor=pe_factor)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.stat_proj  = nn.Sequential(
            nn.Linear(stat_dim, 16), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x_dyn, x_stat, step_present=None, prefix_cutoff=None):
        B, T, _ = x_dyn.shape
        h = self.dyn_proj(x_dyn)
        h = self.pos(h)

        key_padding_mask = None
        if step_present is not None:
            key_padding_mask = (step_present <= 0.0)
        if prefix_cutoff is not None:
            future_mask = torch.arange(T, device=x_dyn.device).unsqueeze(0) > int(prefix_cutoff)
            key_padding_mask = future_mask if key_padding_mask is None else (key_padding_mask | future_mask)

        causal = None
        if self.use_causal_mask:
            causal = torch.triu(torch.ones(T, T, device=x_dyn.device), diagonal=1).bool()

        z = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)

        # last-present-token pooling
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


# ══════════════════════════════════════════════════════════════
# 2. Dataset（與訓練腳本一致）
# ══════════════════════════════════════════════════════════════
class ExtubationSeqDataset(Dataset):
    def __init__(self, df, stay_ids, scaler, scale_cols, seq_time_bins):
        self.df = df[df["stay_id"].isin(stay_ids)].copy()
        self.seq_time_bins = list(seq_time_bins)
        self.stay_ids = sorted(list(set(stay_ids)))
        self.scaler = scaler
        self.scale_cols = list(scale_cols) if scale_cols else []

        if self.scaler is not None and self.scale_cols and len(self.df) > 0:
            self.df.loc[:, self.scale_cols] = self.scaler.transform(self.df[self.scale_cols])

        self.samples = []
        for sid in self.stay_ids:
            d = self.df[self.df["stay_id"] == sid]
            if d.empty:
                continue
            y    = int(d[TARGET].iloc[0])
            stat = d[STATIC_COLS].iloc[0].values.astype(np.float32)
            seq_values, step_present_arr = [], []
            for tb in self.seq_time_bins:
                row = d[d["time_bin"] == tb]
                if row.empty:
                    seq_values.append(np.zeros(len(DYNAMIC_COLS) * 2, dtype=np.float32))
                    step_present_arr.append(0.0)
                else:
                    if "bin_has_data" in row.columns:
                        dyn  = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                        mask = row[MASK_COLS].iloc[0].values.astype(np.float32)
                        pres = float(row["bin_has_data"].iloc[0])
                    else:
                        dyn_raw = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                        mask    = (~np.isnan(dyn_raw)).astype(np.float32)
                        dyn     = np.nan_to_num(dyn_raw, nan=0.0)
                        pres    = float(mask.max())
                    seq_values.append(np.concatenate([dyn, mask]))
                    step_present_arr.append(pres)

            self.samples.append({
                "sid": int(sid),
                "y":   y,
                "x_dyn":        np.stack(seq_values, axis=0).astype(np.float32),
                "x_stat":       stat,
                "step_present": np.array(step_present_arr, dtype=np.float32),
            })

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ══════════════════════════════════════════════════════════════
# 3. 工具函式
# ══════════════════════════════════════════════════════════════
def build_scaler_and_split(df, seed=42):
    """重建與訓練完全相同的 split + scaler（不做任何訓練）。"""
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
    scaler   = StandardScaler()
    scaler.fit(train_df[scale_cols])
    return list(train_ids), list(val_ids), list(test_ids), scaler, scale_cols


def get_trajectory(model, sample, device):
    """對單一病人跑 prefix_cutoff=0..11，回傳 12 個機率值。"""
    x_dyn        = torch.tensor(sample["x_dyn"],        dtype=torch.float32).unsqueeze(0).to(device)
    x_stat       = torch.tensor(sample["x_stat"],       dtype=torch.float32).unsqueeze(0).to(device)
    step_present = torch.tensor(sample["step_present"], dtype=torch.float32).unsqueeze(0).to(device)
    probs = []
    model.eval()
    with torch.no_grad():
        for k in range(SEQ_LEN):
            logit = model(x_dyn, x_stat, step_present=step_present, prefix_cutoff=k)
            probs.append(float(torch.sigmoid(logit).cpu().item()))
    return np.array(probs)


def plot_single(sample, probs, threshold, save_path, title_suffix=""):
    """畫單一病人的動態風險軌跡圖。"""
    x_mid = [tb + WINDOW_HOURS / 2 for tb in SEQ_TIME_BINS]
    label = sample["y"]
    sid   = sample["sid"]
    color = "#D62728" if label == 1 else "#1F77B4"
    label_text = "Extubation Failure (EF=1)" if label == 1 else "Successful Extubation (EF=0)"

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhspan(threshold, 1.0, alpha=0.07, color="#D62728")
    ax.axhspan(0.0, threshold, alpha=0.05, color="#2CA02C")
    ax.text(-53, (threshold + 1.0) / 2, "High-risk\nZone",
            va="center", ha="left", fontsize=9, color="#D62728", alpha=0.7)
    ax.text(-53, threshold / 2, "Low-risk\nZone",
            va="center", ha="left", fontsize=9, color="#2CA02C", alpha=0.7)

    ax.plot(x_mid, probs, marker="o", color=color, linewidth=2.5,
            markersize=6, label="Predicted failure risk")
    ax.axhline(y=threshold, color="crimson", linestyle="--", linewidth=1.8,
               label=f"Decision threshold = {threshold:.3f}")
    ax.axvline(x=0, color="#333333", linestyle=":", linewidth=1.5)
    ax.text(0.5, 1.01, "Extubation", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=9, color="#333333")

    xticks  = list(SEQ_TIME_BINS) + [0]
    xlabels = [str(t) for t in SEQ_TIME_BINS] + ["Extubation"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-55, 3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Hours before extubation  (each point = midpoint of 4h window)", fontsize=12)
    ax.set_ylabel("Predicted extubation failure probability", fontsize=12)
    ax.set_title(
        f"Dynamic Risk Trajectory — {label_text}\nstay_id = {sid}{title_suffix}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


def plot_comparison(sample_s, probs_s, sample_f, probs_f, threshold, save_path):
    """畫成功 vs EF 對比圖（同一張圖，兩條軌跡）。"""
    x_mid = [tb + WINDOW_HOURS / 2 for tb in SEQ_TIME_BINS]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.axhspan(threshold, 1.0, alpha=0.06, color="#D62728")
    ax.axhspan(0.0, threshold, alpha=0.04, color="#2CA02C")

    ax.plot(x_mid, probs_f, marker="o", color="#D62728", linewidth=2.5,
            markersize=6, label=f"EF patient (stay_id={sample_f['sid']})")
    ax.plot(x_mid, probs_s, marker="s", color="#1F77B4", linewidth=2.5,
            markersize=6, linestyle="--", label=f"Successful patient (stay_id={sample_s['sid']})")
    ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1.5,
               label=f"Decision threshold = {threshold:.3f}")
    ax.axvline(x=0, color="#333333", linestyle=":", linewidth=1.5)
    ax.text(0.5, 1.01, "Extubation", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=9, color="#333333")

    xticks  = list(SEQ_TIME_BINS) + [0]
    xlabels = [str(t) for t in SEQ_TIME_BINS] + ["Extubation"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-55, 3)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Hours before extubation  (each point = midpoint of 4h window)", fontsize=12)
    ax.set_ylabel("Predicted extubation failure probability", fontsize=12)
    ax.set_title(
        "Dynamic Risk Trajectory Comparison\nSuccessful Extubation vs. Extubation Failure",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


# ══════════════════════════════════════════════════════════════
# 4. CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Transformer trajectory inference (no retraining)")
    p.add_argument("--data_csv",   type=str, required=True)
    p.add_argument("--model_pt",   type=str, required=True,
                   help="Path to best_transformer.pt")
    p.add_argument("--output_dir", type=str, default="results/trajectory_plots")
    p.add_argument("--threshold",  type=float, default=0.4559,
                   help="Youden threshold from training (default: 0.4559)")
    p.add_argument("--seed",       type=int,   default=42)

    # 模型架構（需與訓練時一致）
    p.add_argument("--d_model",    type=int,   default=64)
    p.add_argument("--nhead",      type=int,   default=4)
    p.add_argument("--num_layers", type=int,   default=3)
    p.add_argument("--dim_ff",     type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--pe_factor",  type=float, default=1.0)
    p.add_argument("--pooling",    type=str,   default="last")

    # 功能選項
    p.add_argument("--list_ids",   type=int, default=0, choices=[0, 1],
                   help="1 = 列出 test set 所有可用 stay_id 後結束")
    p.add_argument("--focus_ids",  type=int, nargs="+", default=None,
                   help="指定一或多個 stay_id，各自畫一張軌跡圖")
    p.add_argument("--all_ef",     type=int, default=0, choices=[0, 1],
                   help="1 = 畫出 test set 所有 EF=1 病人的軌跡圖（批次輸出）")
    p.add_argument("--all_success",type=int, default=0, choices=[0, 1],
                   help="1 = 畫出 test set 所有 EF=0 病人的軌跡圖（批次輸出）")
    p.add_argument("--compare",    type=int, default=0, choices=[0, 1],
                   help="1 = 畫成功 vs EF 對比圖")
    p.add_argument("--success_id", type=int, default=None,
                   help="對比圖中的成功病人 stay_id（不指定則自動選第一位）")
    p.add_argument("--failure_id", type=int, default=None,
                   help="對比圖中的 EF 病人 stay_id（不指定則自動選第一位）")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. 載入資料 ──────────────────────────────────────────
    print("Loading data ...")
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()

    # ── 2. 重建 split + scaler（不訓練）─────────────────────
    print("Rebuilding split and scaler (seed=42, no training) ...")
    train_ids, val_ids, test_ids, scaler, scale_cols = build_scaler_and_split(df, seed=args.seed)

    test_df = df[df["stay_id"].isin(test_ids)].copy()
    ds_test = ExtubationSeqDataset(test_df, test_ids, scaler, scale_cols, SEQ_TIME_BINS)

    # stay_id → sample 快速查找
    sid2sample = {s["sid"]: s for s in ds_test.samples}

    # 分組：成功 vs EF
    success_ids = sorted([s["sid"] for s in ds_test.samples if s["y"] == 0])
    failure_ids = sorted([s["sid"] for s in ds_test.samples if s["y"] == 1])

    # ── 3. 列出可用 stay_id ──────────────────────────────────
    if args.list_ids:
        print(f"\n{'='*65}")
        print(f"Test set: {len(ds_test.samples)} patients")
        print(f"  Successful extubation (EF=0): {len(success_ids)}")
        print(f"  Extubation failure    (EF=1): {len(failure_ids)}")
        print(f"{'='*65}")

        # 儲存 CSV
        rows = [{"stay_id": sid, "label": 0, "outcome": "Success"} for sid in success_ids] + \
               [{"stay_id": sid, "label": 1, "outcome": "EF"}      for sid in failure_ids]
        csv_path = os.path.join(args.output_dir, "test_stay_ids.csv")
        pd.DataFrame(rows).sort_values("stay_id").to_csv(csv_path, index=False)
        print(f"\nSuccess IDs (first 20): {success_ids[:20]}")
        print(f"Failure IDs (first 20): {failure_ids[:20]}")
        print(f"\n✓ Full list saved: {csv_path}")
        return

    # ── 4. 載入模型 ──────────────────────────────────────────
    print(f"Loading model: {args.model_pt}")
    model = TransformerRiskModel(
        dyn_dim=len(DYNAMIC_COLS) * 2,
        stat_dim=len(STATIC_COLS),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        pe_factor=args.pe_factor,
        pooling=args.pooling,
    ).to(device)
    model.load_state_dict(torch.load(args.model_pt, map_location=device))
    model.eval()
    print(f"  Model loaded. Threshold = {args.threshold}")

    # ── 5a. 批次畫出所有 EF=1 或 EF=0 ──────────────────────
    batch_ids = []
    if args.all_ef:
        batch_ids = failure_ids
        subdir = os.path.join(args.output_dir, "all_ef")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n[Batch] 畫出所有 EF=1 病人共 {len(failure_ids)} 人 → {subdir}")
        for i, sid in enumerate(failure_ids, 1):
            sample = sid2sample[sid]
            probs  = get_trajectory(model, sample, device)
            save_path = os.path.join(subdir, f"trajectory_ef_{sid}.png")
            plot_single(sample, probs, args.threshold, save_path)
            if i % 50 == 0:
                print(f"  進度：{i}/{len(failure_ids)}")
        print(f"[Batch] 完成，共儲存 {len(failure_ids)} 張圖。")

    if args.all_success:
        subdir = os.path.join(args.output_dir, "all_success")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n[Batch] 畫出所有 EF=0 病人共 {len(success_ids)} 人 → {subdir}")
        for i, sid in enumerate(success_ids, 1):
            sample = sid2sample[sid]
            probs  = get_trajectory(model, sample, device)
            save_path = os.path.join(subdir, f"trajectory_success_{sid}.png")
            plot_single(sample, probs, args.threshold, save_path)
            if i % 50 == 0:
                print(f"  進度：{i}/{len(success_ids)}")
        print(f"[Batch] 完成，共儲存 {len(success_ids)} 張圖。")

    # ── 5b. 指定 stay_id 個別畫圖 ───────────────────────────
    if args.focus_ids:
        for sid in args.focus_ids:
            if sid not in sid2sample:
                print(f"  [WARN] stay_id={sid} not in test set, skipping.")
                continue
            sample = sid2sample[sid]
            probs  = get_trajectory(model, sample, device)
            save_path = os.path.join(args.output_dir, f"trajectory_stay_{sid}.png")
            plot_single(sample, probs, args.threshold, save_path)

    # ── 6. 對比圖（成功 vs EF）──────────────────────────────
    if args.compare:
        sid_s = args.success_id if args.success_id else (success_ids[0] if success_ids else None)
        sid_f = args.failure_id if args.failure_id else (failure_ids[0] if failure_ids else None)

        if sid_s is None or sid_f is None:
            print("[ERROR] 找不到足夠的成功或失敗病人來畫對比圖。")
            return
        if sid_s not in sid2sample:
            print(f"[ERROR] success_id={sid_s} not in test set.")
            return
        if sid_f not in sid2sample:
            print(f"[ERROR] failure_id={sid_f} not in test set.")
            return

        sample_s = sid2sample[sid_s]
        sample_f = sid2sample[sid_f]
        probs_s  = get_trajectory(model, sample_s, device)
        probs_f  = get_trajectory(model, sample_f, device)

        save_path = os.path.join(
            args.output_dir,
            f"trajectory_comparison_success{sid_s}_ef{sid_f}.png")
        plot_comparison(sample_s, probs_s, sample_f, probs_f, args.threshold, save_path)

    if not any([args.list_ids, args.focus_ids, args.all_ef, args.all_success, args.compare]):
        print("[INFO] 未指定任何動作。請使用 --list_ids 1、--focus_ids、或 --compare 1。")
        print("       執行 --list_ids 1 可取得所有可用 test set stay_id。")


if __name__ == "__main__":
    main()
