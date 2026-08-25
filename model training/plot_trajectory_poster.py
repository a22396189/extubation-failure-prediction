# -*- coding: utf-8 -*-
"""
plot_trajectory_poster.py

Poster 風格動態風險軌跡圖（不重新訓練，直接載入 best_transformer.pt）。

功能：
  --list_ids 1          列出 test set 所有可用 stay_id
  --focus_ids 30015288  畫指定病人（可多個）
  --all_ef 1            批次畫所有 EF=1 病人
  --all_success 1       批次畫所有 EF=0 病人
  --compare 1           畫成功 vs EF 對比圖
  --success_id / --failure_id  指定對比圖的病人

執行範例：
  python "%EXTUBATION_PROJECT_ROOT%/model training/plot_trajectory_poster.py" ^
    --data_csv  "%EXTUBATION_PROJECT_ROOT%/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" ^
    --model_pt  "%EXTUBATION_PROJECT_ROOT%/results/transformer/best_transformer.pt" ^
    --output_dir "%EXTUBATION_PROJECT_ROOT%/results/trajectory_plots" ^
    --focus_ids 35363177
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.ticker
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

matplotlib.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       13,
    "axes.titlesize":  15,
    "axes.labelsize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

# ══════════════════════════════════════════════════════════════
# 0. 常數（與訓練腳本完全一致）
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
SEQ_TIME_BINS = list(range(-52, -4, 4))
SEQ_LEN       = len(SEQ_TIME_BINS)
BINARY_LIKE   = ["sex", "Vasopressor_use", "Hemodialysis_use"]
WINDOW_HOURS  = 4


# ══════════════════════════════════════════════════════════════
# 1. 模型（與訓練腳本一致）
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
                 use_causal_mask=False, pe_factor=0.1, pooling="last"):
        super().__init__()
        self.use_causal_mask = bool(use_causal_mask)
        self.pooling = pooling
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

    def forward(self, x_dyn, x_stat, step_present=None, prefix_cutoff=None):
        B, T, _ = x_dyn.shape
        h = self.dyn_proj(x_dyn)
        h = self.pos(h)
        key_padding_mask = None
        if step_present is not None:
            key_padding_mask = (step_present <= 0.0)
        if prefix_cutoff is not None:
            future_mask = torch.arange(T, device=x_dyn.device).unsqueeze(0) > int(prefix_cutoff)
            key_padding_mask = (future_mask if key_padding_mask is None
                                else key_padding_mask | future_mask)
        causal = (torch.triu(torch.ones(T, T, device=x_dyn.device), diagonal=1).bool()
                  if self.use_causal_mask else None)
        z = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)
        if step_present is None:
            last_idx = torch.full((B,), T - 1, dtype=torch.long, device=x_dyn.device)
        else:
            present = (step_present > 0.0)
            if prefix_cutoff is not None:
                present = present & (torch.arange(T, device=x_dyn.device).unsqueeze(0)
                                     <= int(prefix_cutoff))
            idx = present.float() * torch.arange(T, device=x_dyn.device).unsqueeze(0)
            last_idx = idx.max(dim=1).values.long()
        pooled = z[torch.arange(B, device=x_dyn.device), last_idx, :]
        return self.classifier(torch.cat([pooled, self.stat_proj(x_stat)], dim=1))


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
                "sid": int(sid), "y": y,
                "x_dyn":        np.stack(seq_values, axis=0).astype(np.float32),
                "x_stat":       stat,
                "step_present": np.array(step_present_arr, dtype=np.float32),
            })

    def __len__(self):       return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# ══════════════════════════════════════════════════════════════
# 3. 工具函式
# ══════════════════════════════════════════════════════════════
def build_scaler_and_split(df, seed=42):
    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    y = stay_labels[TARGET].values
    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values, test_size=0.3,
        random_state=seed, stratify=y)
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values, test_size=0.5,
        random_state=seed, stratify=temp_labels[TARGET].values)
    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
                  if c not in BINARY_LIKE]
    scaler = StandardScaler()
    scaler.fit(df[df["stay_id"].isin(train_ids)][scale_cols])
    return list(train_ids), list(val_ids), list(test_ids), scaler, scale_cols


def get_trajectory(model, sample, device):
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


# ══════════════════════════════════════════════════════════════
# 4. 繪圖（Poster 風格）
# ══════════════════════════════════════════════════════════════
def plot_poster(sample, probs, threshold, save_path):
    """
    Poster 風格：
    - 閾值以上=紅，以下=藍（分段著色）
    - Peak risk 標注
    - Threshold crossing 標注（若有）
    - 標題只含 outcome，不含 stay_id
    """
    x_mid     = [tb + WINDOW_HOURS / 2 for tb in SEQ_TIME_BINS]
    probs_arr = np.array(probs)
    x_arr     = np.array(x_mid)
    label     = sample["y"]
    outcome   = "Successful Extubation" if label == 0 else "Extubation Failure"

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 背景色
    ax.axhspan(threshold, 1.05,   color="#fde8e8", alpha=0.55, zorder=0)
    ax.axhspan(-0.05, threshold,  color="#e8f5e9", alpha=0.55, zorder=0)
    ax.text(-54.5, (1.05 + threshold) / 2, "High-risk\nZone",
            color="#c0392b", fontsize=10, va="center", ha="left", style="italic", alpha=0.8)
    ax.text(-54.5, threshold / 2, "Low-risk\nZone",
            color="#27ae60", fontsize=10, va="center", ha="left", style="italic", alpha=0.8)

    # 閾值線與拔管線
    ax.axhline(y=threshold, color="#c0392b", linestyle="--", linewidth=1.8,
               label=f"Decision threshold = {threshold:.3f}", zorder=3)
    ax.axvline(x=0, color="#2c3e50", linestyle=":", linewidth=2,
               label="Extubation", zorder=3)

    # 分段著色曲線
    cross_idx = None
    for i in range(len(probs_arr) - 1):
        if probs_arr[i] >= threshold > probs_arr[i + 1]:
            cross_idx = i
            break

    if cross_idx is not None:
        ax.plot(x_arr[:cross_idx + 2], probs_arr[:cross_idx + 2],
                marker="o", ms=7, lw=2.5, color="#e74c3c",
                label="Predicted failure risk (high)", zorder=4)
        ax.plot(x_arr[cross_idx + 1:], probs_arr[cross_idx + 1:],
                marker="o", ms=7, lw=2.5, color="#2980b9",
                label="Predicted failure risk (low)", zorder=4)
        ax.plot(x_arr[cross_idx:cross_idx + 2], probs_arr[cross_idx:cross_idx + 2],
                lw=2.5, color="#e74c3c", zorder=3)
        cross_x = x_arr[cross_idx + 1]
        cross_y = probs_arr[cross_idx + 1]
        ax.annotate(f"Risk below threshold at {int(cross_x)}h",
                    xy=(cross_x, cross_y),
                    xytext=(cross_x - 14, threshold - 0.16),
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
                    color="#27ae60", fontsize=10, fontweight="bold")
    else:
        line_color   = "#e74c3c" if probs_arr.mean() >= threshold else "#2980b9"
        legend_label = ("Predicted failure risk (high)" if line_color == "#e74c3c"
                        else "Predicted failure risk (low)")
        ax.plot(x_arr, probs_arr, marker="o", ms=7, lw=2.5,
                color=line_color, label=legend_label, zorder=4)

    # 峰值標注
    peak_idx = int(np.argmax(probs_arr))
    peak_x, peak_y = x_arr[peak_idx], probs_arr[peak_idx]
    ax.annotate(f"Peak risk\n{peak_y:.2f}",
                xy=(peak_x, peak_y),
                xytext=(peak_x + 3, peak_y + 0.05),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
                color="#e74c3c", fontsize=10, fontweight="bold")

    # 軸設定
    ax.set_xlim(-56, 3)
    ax.set_ylim(-0.05, 1.1)
    xticks  = list(range(-52, 1, 4))
    xlabels = [str(t) if t != 0 else "Extubation" for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.grid(alpha=0.25, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Hours before extubation  (each point = 4h window)", fontsize=13, labelpad=6)
    ax.set_ylabel("Predicted extubation failure risk", fontsize=13, labelpad=6)
    ax.set_title(f"Dynamic Risk Trajectory — {outcome}",
                 fontweight="bold", fontsize=15, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="lightgrey", fontsize=10)

    plt.tight_layout()
    png_path = save_path if save_path.endswith(".png") else save_path + ".png"
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)
    print(f"✓ PNG: {png_path}")
    print(f"✓ PDF: {pdf_path}")


def plot_comparison_poster(sample_s, probs_s, sample_f, probs_f, threshold, save_path):
    """對比圖（Poster 風格）：成功 vs EF 各一條軌跡。"""
    x_mid = [tb + WINDOW_HOURS / 2 for tb in SEQ_TIME_BINS]
    x_arr = np.array(x_mid)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.axhspan(threshold, 1.05,  color="#fde8e8", alpha=0.45, zorder=0)
    ax.axhspan(-0.05, threshold, color="#e8f5e9", alpha=0.45, zorder=0)
    ax.axhline(y=threshold, color="#c0392b", linestyle="--", linewidth=1.8,
               label=f"Decision threshold = {threshold:.3f}", zorder=3)
    ax.axvline(x=0, color="#2c3e50", linestyle=":", linewidth=2,
               label="Extubation", zorder=3)

    ax.plot(x_arr, np.array(probs_f), marker="o", ms=7, lw=2.5, color="#e74c3c",
            label=f"EF patient  (stay_id={sample_f['sid']})", zorder=4)
    ax.plot(x_arr, np.array(probs_s), marker="s", ms=7, lw=2.5, color="#2980b9",
            linestyle="--", label=f"Success patient  (stay_id={sample_s['sid']})", zorder=4)

    ax.set_xlim(-56, 3)
    ax.set_ylim(-0.05, 1.1)
    xticks  = list(range(-52, 1, 4))
    xlabels = [str(t) if t != 0 else "Extubation" for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.grid(alpha=0.25, linestyle="--", zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Hours before extubation  (each point = 4h window)", fontsize=13, labelpad=6)
    ax.set_ylabel("Predicted extubation failure risk", fontsize=13, labelpad=6)
    ax.set_title("Dynamic Risk Trajectory Comparison\nSuccessful Extubation vs. Extubation Failure",
                 fontweight="bold", fontsize=15, pad=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="lightgrey", fontsize=10)

    plt.tight_layout()
    png_path = save_path if save_path.endswith(".png") else save_path + ".png"
    pdf_path = png_path.replace(".png", ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)
    print(f"✓ PNG: {png_path}")
    print(f"✓ PDF: {pdf_path}")


# ══════════════════════════════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="Poster-style trajectory plots (no retraining)")
    p.add_argument("--data_csv",    type=str, required=True)
    p.add_argument("--model_pt",    type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="results/trajectory_poster")
    p.add_argument("--threshold",   type=float, default=0.4559)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--d_model",     type=int,   default=64)
    p.add_argument("--nhead",       type=int,   default=4)
    p.add_argument("--num_layers",  type=int,   default=3)
    p.add_argument("--dim_ff",      type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.2)
    p.add_argument("--pe_factor",   type=float, default=1.0)
    p.add_argument("--list_ids",    type=int,   default=0, choices=[0, 1])
    p.add_argument("--focus_ids",   type=int,   nargs="+", default=None)
    p.add_argument("--all_ef",      type=int,   default=0, choices=[0, 1])
    p.add_argument("--all_success", type=int,   default=0, choices=[0, 1])
    p.add_argument("--compare",     type=int,   default=0, choices=[0, 1])
    p.add_argument("--success_id",  type=int,   default=None)
    p.add_argument("--failure_id",  type=int,   default=None)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 載入資料
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()

    # 重建 split + scaler
    _, _, test_ids, scaler, scale_cols = build_scaler_and_split(df, seed=args.seed)
    test_df = df[df["stay_id"].isin(test_ids)].copy()
    ds_test = ExtubationSeqDataset(test_df, test_ids, scaler, scale_cols, SEQ_TIME_BINS)
    sid2sample  = {s["sid"]: s for s in ds_test.samples}
    success_ids = sorted([s["sid"] for s in ds_test.samples if s["y"] == 0])
    failure_ids = sorted([s["sid"] for s in ds_test.samples if s["y"] == 1])

    # --list_ids
    if args.list_ids:
        rows = ([{"stay_id": s, "label": 0, "outcome": "Success"} for s in success_ids] +
                [{"stay_id": s, "label": 1, "outcome": "EF"}      for s in failure_ids])
        csv_path = os.path.join(args.output_dir, "test_stay_ids.csv")
        pd.DataFrame(rows).sort_values("stay_id").to_csv(csv_path, index=False)
        print(f"Success (EF=0): {len(success_ids)}  |  EF (EF=1): {len(failure_ids)}")
        print(f"Success IDs (first 20): {success_ids[:20]}")
        print(f"Failure IDs (first 20): {failure_ids[:20]}")
        print(f"✓ Full list: {csv_path}")
        return

    # 載入模型
    print(f"Loading model: {args.model_pt}")
    model = TransformerRiskModel(
        dyn_dim=len(DYNAMIC_COLS) * 2, stat_dim=len(STATIC_COLS),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, dropout=args.dropout, pe_factor=args.pe_factor,
    ).to(device)
    model.load_state_dict(torch.load(args.model_pt, map_location=device))
    model.eval()

    def run_and_plot(sid, subdir=None):
        if sid not in sid2sample:
            print(f"  [WARN] stay_id={sid} not in test set.")
            return
        sample = sid2sample[sid]
        probs  = get_trajectory(model, sample, device)
        out_dir = subdir or args.output_dir
        label_tag = "ef" if sample["y"] == 1 else "success"
        save_path = os.path.join(out_dir, f"trajectory_{label_tag}_{sid}_poster.png")
        plot_poster(sample, probs, args.threshold, save_path)

    # --all_ef
    if args.all_ef:
        subdir = os.path.join(args.output_dir, "all_ef")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n[Batch EF] {len(failure_ids)} patients → {subdir}")
        for i, sid in enumerate(failure_ids, 1):
            run_and_plot(sid, subdir)
            if i % 50 == 0:
                print(f"  {i}/{len(failure_ids)}")

    # --all_success
    if args.all_success:
        subdir = os.path.join(args.output_dir, "all_success")
        os.makedirs(subdir, exist_ok=True)
        print(f"\n[Batch Success] {len(success_ids)} patients → {subdir}")
        for i, sid in enumerate(success_ids, 1):
            run_and_plot(sid, subdir)
            if i % 50 == 0:
                print(f"  {i}/{len(success_ids)}")

    # --focus_ids
    if args.focus_ids:
        for sid in args.focus_ids:
            run_and_plot(sid)

    # --compare
    if args.compare:
        sid_s = args.success_id or (success_ids[0] if success_ids else None)
        sid_f = args.failure_id or (failure_ids[0] if failure_ids else None)
        if not sid_s or not sid_f:
            print("[ERROR] 找不到足夠病人畫對比圖。")
            return
        if sid_s not in sid2sample or sid_f not in sid2sample:
            print(f"[ERROR] stay_id not in test set: success={sid_s}, failure={sid_f}")
            return
        probs_s = get_trajectory(model, sid2sample[sid_s], device)
        probs_f = get_trajectory(model, sid2sample[sid_f], device)
        save_path = os.path.join(
            args.output_dir,
            f"trajectory_comparison_success{sid_s}_ef{sid_f}_poster.png")
        plot_comparison_poster(
            sid2sample[sid_s], probs_s,
            sid2sample[sid_f], probs_f,
            args.threshold, save_path)

    if not any([args.list_ids, args.focus_ids, args.all_ef, args.all_success, args.compare]):
        print("[INFO] 請指定 --list_ids 1、--focus_ids、--all_ef 1、或 --compare 1。")


if __name__ == "__main__":
    main()
