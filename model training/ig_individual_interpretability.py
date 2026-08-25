# -*- coding: utf-8 -*-
"""
ig_individual_interpretability.py

意見三：個案層級可解釋性（Integrated Gradients）

【目的】
    不重新訓練模型，直接對既有 best_transformer.pt 做推論層級運算：
    1. 重建 test set（與訓練/子群分析腳本相同 split/scaler 邏輯，確保結果可重現）
    2. 依決策閾值分出 TP/TN/FP/FN，各選一位代表個案（機率落在該象限中段，非邊界模糊病人）
    3. 對每位代表個案用 Integrated Gradients 計算 26 動態變數 × 12 time bin 的 attribution
    4. 輸出 4 張獨立 heatmap + 1 張 2x2 合成 heatmap

【與既有程式碼的關係】
    ExtubationTransformer / ExtubationSeqDataset / split_by_stay_id / find_best_threshold
    皆從 subgroup_analysis_transformer.py 原樣複製而來（不 import 該檔案，因其路徑含空格，
    避免模組匯入問題；為了確保 state_dict 與 split/scaler 完全相容，這裡逐字複製，
    未修改任何原始檔案）。

【輸出（於 --output_dir 下）】
    - subgroup_predictions.csv    test set 病人層級 y_true / y_prob（含 stay_id）
    - selected_cases.csv          TP/TN/FP/FN 代表個案資訊
    - IG_heatmap_{TP,TN,FP,FN}.png
    - IG_heatmap_2x2.png
    - ig_convergence.csv          各案例的 IG 收斂誤差（delta）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from captum.attr import IntegratedGradients

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# 0. 與訓練/子群分析腳本完全一致的常數與 class
# =========================================================================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]

DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]
MASK_COLS = [f"mask_{col}" for col in DYNAMIC_COLS]
TARGET = "Extubation_failure"

SEQ_TIME_BINS = list(range(-52, -4, 4))  # -52, -48, ..., -8
SEQ_LEN = len(SEQ_TIME_BINS)
BINARY_LIKE = ["sex", "Vasopressor_use", "Hemodialysis_use"]


def split_by_stay_id(df, train_ratio=0.7, seed=42):
    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    y = stay_labels[TARGET].values
    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values, test_size=(1 - train_ratio),
        random_state=seed, stratify=y
    )
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    y_temp = temp_labels[TARGET].values
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values, test_size=0.5,
        random_state=seed, stratify=y_temp
    )
    return train_ids, val_ids, test_ids


def find_best_threshold(y_true, y_prob, mode="youden"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, factor=0.1):
        super().__init__()
        self.factor = factor
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.factor * self.pe[:, :T, :]


class ExtubationTransformer(nn.Module):
    def __init__(self, dyn_dim=52, stat_dim=4, d_model=64, nhead=4, num_layers=3,
                 dim_ff=128, dropout=0.1, use_causal_mask=False, pe_factor=0.1,
                 pooling="last", train_noise=0.0):
        super().__init__()
        self.use_causal_mask = bool(use_causal_mask)
        self.pooling = pooling
        self.train_noise = float(train_noise)

        self.dyn_proj = nn.Linear(dyn_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=256, factor=pe_factor)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.stat_proj = nn.Sequential(nn.Linear(stat_dim, 16), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

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

        attn_mask = None
        if self.use_causal_mask:
            attn_mask = torch.triu(torch.ones(T, T, device=x_dyn.device) * float("-inf"), diagonal=1)

        z = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)

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
        out = self.classifier(torch.cat([pooled, s], dim=1))
        return out


class ExtubationSeqDataset(Dataset):
    def __init__(self, df, stay_ids, scaler, scale_cols, seq_time_bins):
        self.df = df[df["stay_id"].isin(stay_ids)].copy()
        self.seq_time_bins = list(seq_time_bins)
        self.stay_ids = sorted(list(set(stay_ids)))

        self.scaler = scaler
        self.scale_cols = list(scale_cols) if scale_cols is not None else []
        if self.scaler is not None and len(self.scale_cols) > 0 and len(self.df) > 0:
            self.df.loc[:, self.scale_cols] = self.scaler.transform(self.df[self.scale_cols])

        self.has_precomputed_masks = (
            all(col in self.df.columns for col in MASK_COLS) and "bin_has_data" in self.df.columns
        )

        self.samples = []
        self.sid_to_idx = {}
        for sid in self.stay_ids:
            s = self._build_one(int(sid))
            if s is not None:
                self.sid_to_idx[int(sid)] = len(self.samples)
                self.samples.append(s)

    def _build_one(self, sid: int):
        d = self.df[self.df["stay_id"] == sid].copy()
        if d.empty:
            return None
        y = int(d[TARGET].iloc[0])
        stat = d[STATIC_COLS].iloc[0].copy()

        seq_values, seq_step_present = [], []
        for tb in self.seq_time_bins:
            row = d[d["time_bin"] == tb]
            if row.empty:
                dyn_filled = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)
                mask = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)
                step_pres_val = 0.0
            elif self.has_precomputed_masks:
                dyn_filled = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                mask = row[MASK_COLS].iloc[0].values.astype(np.float32)
                step_pres_val = float(row["bin_has_data"].iloc[0])
            else:
                dyn_raw = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                mask = (~np.isnan(dyn_raw)).astype(np.float32)
                dyn_filled = np.nan_to_num(dyn_raw, nan=0.0).astype(np.float32)
                step_pres_val = 1.0 if mask.sum() > 0 else 0.0

            seq_values.append(np.concatenate([dyn_filled, mask], axis=0).astype(np.float32))
            seq_step_present.append(step_pres_val)

        seq_values = np.stack(seq_values, axis=0)
        seq_step_present = np.array(seq_step_present, dtype=np.float32)
        stat = stat.fillna(0).values.astype(np.float32)

        return {
            "sid": int(sid), "x_dyn": seq_values, "x_stat": stat,
            "y": float(y), "step_present": seq_step_present
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["x_dyn"], dtype=torch.float32),
            torch.tensor(s["x_stat"], dtype=torch.float32),
            torch.tensor([s["y"]], dtype=torch.float32),
            torch.tensor(s["step_present"], dtype=torch.float32),
            int(s["sid"]),
        )

    def get_by_sid(self, sid):
        idx = self.sid_to_idx[int(sid)]
        return self[idx]


def get_probs(model, ds, device, batch_size=256):
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs, labels, sids = [], [], []
    with torch.no_grad():
        for x_dyn, x_stat, y, step_present, sid in loader:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            step_present = step_present.to(device)
            logit = model(x_dyn, x_stat, step_present=step_present)
            p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            probs.extend(list(p))
            labels.extend(list(y.numpy().reshape(-1)))
            sids.extend(list(sid))
    return np.array(labels).astype(int), np.array(probs).astype(float), np.array(sids).astype(int)


# =========================================================================
# 0. 路徑設定（【路徑注意】以下為程式撰寫時所在機器上的檔案路徑，
#    於其他環境執行前請依實際檔案存放位置調整；亦可於執行時以對應的
#    --data_csv / --model_pt / --output_dir 參數覆寫，無需修改程式碼）
# =========================================================================
DEFAULT_DATA_CSV = r"C:\Users\your-username\Desktop\extubation_failure_prediction\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
DEFAULT_MODEL_PT = r"C:\Users\your-username\Desktop\extubation_failure_prediction\results\transformer_save_val_auroc\best_transformer.pt"
DEFAULT_OUTPUT_DIR = r"C:\Users\your-username\Desktop\extubation_failure_prediction\results\ig_individual_interpretability"


# =========================================================================
# 1. CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Individual-level Integrated Gradients interpretability.")
    p.add_argument("--data_csv", type=str, default=DEFAULT_DATA_CSV)
    p.add_argument("--model_pt", type=str, default=DEFAULT_MODEL_PT)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dim_ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pe_factor", type=float, default=1.0)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--use_causal_mask", type=int, default=0, choices=[0, 1])

    p.add_argument("--threshold", type=float, default=None,
                    help="若指定，直接使用此 threshold；否則於 val set 以 Youden index 重新尋找。")
    p.add_argument("--n_steps", type=int, default=50)

    return p.parse_args()


# =========================================================================
# 2. Main
# =========================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. 載入資料、重建 split + scaler（與訓練/子群分析腳本完全一致）───
    print("Loading data ...")
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()

    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.7, seed=args.seed)
    train_df = df[df["stay_id"].isin(train_ids)].copy()
    val_df = df[df["stay_id"].isin(val_ids)].copy()
    test_df = df[df["stay_id"].isin(test_ids)].copy()

    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"]) if c not in BINARY_LIKE]
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])

    ds_val = ExtubationSeqDataset(val_df, val_ids, scaler, scale_cols, SEQ_TIME_BINS)
    ds_test = ExtubationSeqDataset(test_df, test_ids, scaler, scale_cols, SEQ_TIME_BINS)
    print(f"Val stays: {len(ds_val)} | Test stays: {len(ds_test)}")

    # ── 2. 載入模型 ─────────────────────────────────────────────────────
    print(f"Loading model: {args.model_pt}")
    model = ExtubationTransformer(
        dyn_dim=len(DYNAMIC_COLS) * 2, stat_dim=len(STATIC_COLS),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
        use_causal_mask=bool(args.use_causal_mask), pe_factor=args.pe_factor,
        pooling=args.pooling,
    ).to(device)
    model.load_state_dict(torch.load(args.model_pt, map_location=device))
    model.eval()

    # ── 3. Threshold ────────────────────────────────────────────────────
    if args.threshold is not None:
        best_thr = float(args.threshold)
        print(f"[Threshold] 使用指定值：{best_thr:.4f}")
    else:
        yv, pv, _ = get_probs(model, ds_val, device)
        best_thr = find_best_threshold(yv, pv, mode="youden")
        print(f"[Threshold] Val set 重新尋找（Youden）：{best_thr:.4f}")

    # ── 4. Test set 推論 ────────────────────────────────────────────────
    print("Running inference on test set ...")
    yt, pt, sids_t = get_probs(model, ds_test, device)
    pred_df = pd.DataFrame({"stay_id": sids_t, "y_true": yt, "y_prob": pt})
    pred_csv = os.path.join(args.output_dir, "subgroup_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {pred_csv}")

    # ── 5. 依 threshold 分 TP/TN/FP/FN，各選中段代表個案 ──────────────────
    pred_df["pred"] = (pred_df.y_prob >= best_thr).astype(int)
    tp = pred_df[(pred_df.y_true == 1) & (pred_df.pred == 1)].sort_values("y_prob", ascending=False).reset_index(drop=True)
    tn = pred_df[(pred_df.y_true == 0) & (pred_df.pred == 0)].sort_values("y_prob", ascending=False).reset_index(drop=True)
    fp = pred_df[(pred_df.y_true == 0) & (pred_df.pred == 1)].sort_values("y_prob", ascending=False).reset_index(drop=True)
    fn = pred_df[(pred_df.y_true == 1) & (pred_df.pred == 0)].sort_values("y_prob", ascending=False).reset_index(drop=True)

    print(f"[Confusion] TP={len(tp)}  TN={len(tn)}  FP={len(fp)}  FN={len(fn)}")

    cases = {
        "TP": tp.iloc[len(tp) // 3],
        "TN": tn.iloc[len(tn) // 3],
        "FP": fp.iloc[len(fp) // 3],
        "FN": fn.iloc[len(fn) // 3],
    }
    cases_df = pd.DataFrame(cases).T.reset_index().rename(columns={"index": "case_type"})
    cases_csv = os.path.join(args.output_dir, "selected_cases.csv")
    cases_df.to_csv(cases_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {cases_csv}")
    print(cases_df[["case_type", "stay_id", "y_true", "y_prob"]])

    # ── 6. Integrated Gradients ────────────────────────────────────────
    def forward_fn(x_dyn, x_stat):
        return torch.sigmoid(model(x_dyn, x_stat))

    ig = IntegratedGradients(forward_fn)

    attr_results = {}
    delta_rows = []
    for case_type, row in cases.items():
        sid = int(row["stay_id"])
        x_dyn, x_stat, y, step_present, _sid = ds_test.get_by_sid(sid)
        x_dyn = x_dyn.unsqueeze(0).to(device)      # [1, 12, 52]
        x_stat = x_stat.unsqueeze(0).to(device)    # [1, 4]

        baseline_dyn = torch.zeros_like(x_dyn)
        attr, delta = ig.attribute(
            inputs=x_dyn, baselines=baseline_dyn,
            additional_forward_args=(x_stat,),
            n_steps=args.n_steps, return_convergence_delta=True
        )
        attr_26 = attr[0, :, :26].detach().cpu().numpy().T  # 26 features x 12 time bins
        attr_results[case_type] = attr_26

        delta_val = float(delta.detach().cpu().numpy()[0])
        delta_rows.append({
            "case_type": case_type, "stay_id": sid,
            "y_true": int(row["y_true"]), "y_prob": float(row["y_prob"]),
            "ig_delta": delta_val
        })
        print(f"[IG] {case_type} (stay_id={sid}, y_prob={row['y_prob']:.3f}): "
              f"convergence delta={delta_val:.4e}")

    delta_df = pd.DataFrame(delta_rows)
    delta_csv = os.path.join(args.output_dir, "ig_convergence.csv")
    delta_df.to_csv(delta_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {delta_csv}")

    # ── 7. Heatmaps（4 張獨立 + 1 張 2x2 合成）───────────────────────────
    x_labels = [f"{t}h" for t in SEQ_TIME_BINS]
    vmax = max(np.abs(a).max() for a in attr_results.values())

    for case_type, attr_26 in attr_results.items():
        row = cases[case_type]
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(attr_26, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=x_labels, yticklabels=DYNAMIC_COLS,
                    cbar_kws={"label": "IG attribution (+ = toward higher risk)"}, ax=ax)
        ax.set_title(f"{case_type} case (stay_id={int(row['stay_id'])}, "
                      f"y_true={int(row['y_true'])}, y_prob={row['y_prob']:.2f})")
        ax.set_xlabel("Time relative to extubation")
        ax.set_ylabel("Dynamic variable")
        plt.tight_layout()
        out_png = os.path.join(args.output_dir, f"IG_heatmap_{case_type}.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved: {out_png}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    order = ["TP", "FN", "FP", "TN"]  # 2x2: 左上TP 右上FN 左下FP 右下TN
    for ax, case_type in zip(axes.flat, order):
        attr_26 = attr_results[case_type]
        row = cases[case_type]
        sns.heatmap(attr_26, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=x_labels, yticklabels=DYNAMIC_COLS,
                    cbar_kws={"label": "IG attribution"}, ax=ax)
        ax.set_title(f"{case_type} (stay_id={int(row['stay_id'])}, "
                      f"y_true={int(row['y_true'])}, y_prob={row['y_prob']:.2f})")
        ax.set_xlabel("Time relative to extubation")
        ax.set_ylabel("Dynamic variable")
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, "IG_heatmap_2x2.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")

    # ── 8. 每個案例：影響方向最大的前 5 個 (variable, time bin) ─────────
    print("\n" + "=" * 70)
    print("每個案例 attribution 絕對值最大的前 5 個 (variable, time_bin)")
    print("=" * 70)
    top5_rows = []
    for case_type, attr_26 in attr_results.items():
        flat_idx = np.argsort(-np.abs(attr_26), axis=None)[:5]
        var_idx, time_idx = np.unravel_index(flat_idx, attr_26.shape)
        print(f"\n[{case_type}] stay_id={int(cases[case_type]['stay_id'])}")
        for vi, ti in zip(var_idx, time_idx):
            val = attr_26[vi, ti]
            direction = "toward HIGH risk" if val > 0 else "toward LOW risk"
            print(f"  {DYNAMIC_COLS[vi]:18s} @ {SEQ_TIME_BINS[ti]:>4d}h : {val:+.4f}  ({direction})")
            top5_rows.append({
                "case_type": case_type, "variable": DYNAMIC_COLS[vi],
                "time_bin": SEQ_TIME_BINS[ti], "attribution": val
            })
    pd.DataFrame(top5_rows).to_csv(
        os.path.join(args.output_dir, "top5_attributions_per_case.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\n完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
