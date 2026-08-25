#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lstm_pre_extubation_risk_trajectory.py
LSTM model for pre-extubation failure risk prediction using sequential ICU data.

Example run:
    python "C:/Users/your-username/Desktop/extubation_project_code_review/model training/lstm_pre_extubation_risk_trajectory.py" \
        --data_csv "C:/Users/your-username/Desktop/extubation_project_code_review/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
        --output_dir "C:/Users/your-username/Desktop/extubation_project_code_review/results/lstm" \
        --hidden_dim 64 --num_layers 2 --dropout 0.2 \
        --lr 5e-5 --epochs 80 --patience 15 \
        --lr_warmup_steps 5 --lr_decay 0.99 \
        --threshold_mode youden \
        --focus_stay_id 30015288
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 0. Feature config  (identical to Transformer script)
# =============================================================================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]
DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]
MASK_COLS = [f"mask_{col}" for col in DYNAMIC_COLS]   # 26 pre-computed masks
TARGET = "Extubation_failure"
SEQ_TIME_BINS = list(range(-52, -4, 4))                # [-52, -48, ..., -8] → 12 bins
SEQ_LEN = len(SEQ_TIME_BINS)
WINDOW_HOURS = 4


# =============================================================================
# 1. Utilities
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_best_threshold(y_true, y_prob, mode="youden"):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thr[np.argmax(j)])
    # F1-optimal
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    return float(thr[np.argmax(f1[:-1])])


def split_by_stay_id(df, train_ratio=0.7, seed=42):
    """
    Subject-level leakage guard: if 'subject_id' is present, verify 1-to-1
    mapping, then split by stay_id stratified on TARGET.
    """
    if "subject_id" in df.columns:
        subj_stay_count = df.groupby("subject_id")["stay_id"].nunique()
        multi_stay_count = int((subj_stay_count > 1).sum())
        if multi_stay_count > 0:
            print(f"[WARNING] {multi_stay_count} subject_id(s) map to multiple stay_ids. "
                  f"Check upstream filter_unique_subject.py.")
        else:
            print(f"[OK] Subject-level leakage check passed: "
                  f"{len(subj_stay_count)} unique subjects, each with 1 stay_id.")

    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=stay_labels[TARGET]
    )
    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels[TARGET]
    )
    return train_ids, val_ids, test_ids


# =============================================================================
# 2. Dataset  (uses pre-computed mask_* + bin_has_data like the Transformer)
# =============================================================================
class ExtubationSeqDataset(Dataset):
    def __init__(self, df, stay_ids, scaler, scale_cols, seq_time_bins):
        self.df = df[df["stay_id"].isin(stay_ids)].copy()
        self.seq_time_bins = list(seq_time_bins)
        self.stay_ids = sorted(list(set(stay_ids)))
        self.scaler = scaler
        self.scale_cols = list(scale_cols) if scale_cols is not None else []

        if self.scaler is not None and len(self.scale_cols) > 0 and len(self.df) > 0:
            self.df.loc[:, self.scale_cols] = self.scaler.transform(self.df[self.scale_cols])

        # Detect pre-computed masks (same logic as Transformer)
        self.has_precomputed_masks = (
            all(col in self.df.columns for col in MASK_COLS)
            and "bin_has_data" in self.df.columns
        )
        if self.has_precomputed_masks:
            print("[Dataset] Detected pre-computed missingness masks "
                  "(mask_* + bin_has_data) — reading from CSV.")
        else:
            print("[Dataset] Pre-computed mask columns not found — "
                  "computing masks on-the-fly from NaN.")

        # 預分組（O(n) 一次），避免每次 _build_one 做 O(n) 全表掃描
        self._df_grouped = {
            int(sid): grp.reset_index(drop=True)
            for sid, grp in self.df.groupby("stay_id")
        }

        self.samples = []
        for sid in self.stay_ids:
            s = self._build_one(int(sid))
            if s is not None:
                self.samples.append(s)

    def _build_one(self, sid: int):
        d = self._df_grouped.get(sid)
        if d is None or d.empty:
            return None
        y = int(d[TARGET].iloc[0])
        stat = d[STATIC_COLS].iloc[0].fillna(0).values.astype(np.float32)

        seq_values = []
        seq_step_present = []
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

            dyn_combined = np.concatenate([dyn_filled, mask], axis=0).astype(np.float32)
            seq_values.append(dyn_combined)
            seq_step_present.append(step_pres_val)

        seq_values = np.stack(seq_values, axis=0)                  # (T, 52)
        seq_step_present = np.array(seq_step_present, dtype=np.float32)
        return {
            "sid": int(sid),
            "x_dyn": seq_values,
            "x_stat": stat,
            "y": float(y),
            "step_present": seq_step_present,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["x_dyn"],        dtype=torch.float32),
            torch.tensor(s["x_stat"],       dtype=torch.float32),
            torch.tensor([s["y"]],          dtype=torch.float32),
            torch.tensor(s["step_present"], dtype=torch.float32),
            int(s["sid"]),
        )


# =============================================================================
# 3. Model
# =============================================================================
class ExtubationLSTM(nn.Module):
    def __init__(self, dyn_dim=52, stat_dim=4, hidden_dim=64, num_layers=2,
                 dropout=0.2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            dyn_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim + 16, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x_dyn, x_stat, step_present=None, prefix_cutoff=None):
        B, T, _ = x_dyn.shape
        output, _ = self.lstm(x_dyn)                               # (B, T, H)
        if prefix_cutoff is not None:
            idx = torch.full((B,), prefix_cutoff, dtype=torch.long, device=x_dyn.device)
        elif step_present is not None:
            idx = (step_present.sum(dim=1) - 1).clamp(min=0).long()
        else:
            idx = torch.full((B,), T - 1, dtype=torch.long, device=x_dyn.device)
        pooled = output[torch.arange(B), idx, :]                   # (B, H)
        s = self.stat_proj(x_stat)                                 # (B, 16)
        return self.classifier(torch.cat([pooled, s], dim=-1))     # (B, 1)


# =============================================================================
# 4. Evaluation helpers
# =============================================================================
def get_probs(model, loader, device):
    model.eval()
    probs, labels, sids = [], [], []
    with torch.no_grad():
        for x_dyn, x_stat, y, step_present, sid in loader:
            logit = model(x_dyn.to(device), x_stat.to(device), step_present.to(device))
            probs.extend(torch.sigmoid(logit).cpu().numpy().reshape(-1))
            labels.extend(y.numpy().reshape(-1))
            sids.extend([s.item() if hasattr(s, 'item') else int(s) for s in (sid if not isinstance(sid, int) else [sid])])
    return np.array(labels).astype(int), np.array(probs).astype(float), sids


def plot_roc_cm(y_true, y_prob, threshold, out_path, label="LSTM"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    print(f"\n{'='*30}")
    print(f"Test Set Performance (Threshold={threshold:.4f}):")
    print(f"AUROC       : {auroc:.4f}")
    print(f"AUPRC       : {auprc:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Sensitivity : {sens:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {sens:.4f}")
    print(f"F1 score    : {f1:.4f}")
    print("=" * 30)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(fpr, tpr, color="#1f77b4", lw=3, label=f"{label} (AUROC={auroc:.4f})")
    axes[0].fill_between(fpr, tpr, alpha=0.2, color="#1f77b4")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    axes[0].set_title("ROC Curve", fontweight="bold", fontsize=14)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower right")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_xticklabels(["Pred: Success (0)", "Pred: Failure (1)"])
    axes[1].set_yticklabels(["True: Success (0)", "True: Failure (1)"], va="center")
    axes[1].set_title(f"Confusion Matrix\n(threshold={threshold:.4f})",
                      fontweight="bold", fontsize=14)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": acc,
        "Sensitivity": sens, "Specificity": spec,
        "Precision": prec, "F1_score": f1, "Recall": sens,
    }


def plot_calibration_brier(y_true, y_prob, model_name, out_path, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mean_pred, frac_pos, "s-", color="#1f77b4", lw=2,
                 label=f"{model_name} (Brier={brier:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curve", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.6,
                 color="steelblue", label="True: Success (0)")
    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.6,
                 color="tomato", label="True: Failure (1)")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted Probability Distribution", fontweight="bold")
    axes[1].legend()

    plt.suptitle(f"{model_name} — Calibration Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Brier Score: {brier:.4f}  (saved to: {out_path})")
    return {"Brier_score": float(brier)}


def plot_trajectory_for_stay(model, dataset, target_sid, device,
                              threshold, save_path, window_hours=4):
    """
    Plot per-time-step predicted failure risk trajectory for a single stay_id.
    Uses prefix_cutoff to reveal the LSTM prediction at each sequential step.
    """
    model.eval()

    sample = next(
        (s for s in dataset.samples if int(s["sid"]) == int(target_sid)),
        None
    )
    if sample is None:
        print(f"[WARNING] stay_id={target_sid} not found in this dataset split.")
        return

    x_dyn       = torch.tensor(sample["x_dyn"],        dtype=torch.float32).unsqueeze(0).to(device)
    x_stat      = torch.tensor(sample["x_stat"],       dtype=torch.float32).unsqueeze(0).to(device)
    step_present = torch.tensor(sample["step_present"], dtype=torch.float32).unsqueeze(0).to(device)
    true_label  = int(sample["y"])
    T = x_dyn.shape[1]

    probs = []
    with torch.no_grad():
        for k in range(T):
            logit = model(x_dyn, x_stat, step_present=step_present, prefix_cutoff=k)
            probs.append(torch.sigmoid(logit).cpu().item())

    start_bins = list(getattr(dataset, "seq_time_bins", SEQ_TIME_BINS))
    x_mid = [tb + (window_hours / 2) for tb in start_bins]

    plt.figure(figsize=(12, 6))
    plt.plot(x_mid, probs, marker="o", linewidth=2.5, markersize=8,
             label="LSTM Predicted Risk")
    plt.axhline(y=threshold, color="crimson", linestyle="--", linewidth=2,
                label=f"Threshold={threshold:.3f}")
    plt.axvline(x=0, color="black", linestyle=":", linewidth=2,
                label="Extubation Time")

    xticks = list(range(-52, 5, 4))
    xlabels = [str(t) for t in xticks]
    xlabels[-1] = ""
    plt.xticks(xticks, xlabels)

    plt.ylim(-0.05, 1.05)
    plt.xlim(-56, 4)
    plt.grid(alpha=0.3, linestyle="--")
    plt.xlabel("Hours Before Extubation (Window Midpoint)")
    plt.ylabel("Predicted Failure Risk P(label=1)")
    plt.title(f"LSTM Stay {target_sid} Trajectory (True Label={true_label})",
              fontweight="bold", fontsize=14)
    plt.legend(loc="upper left")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved: {save_path}")


# =============================================================================
# 5. Argument parser
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="LSTM pre-extubation risk trajectory model"
    )
    # Data / output
    p.add_argument("--data_csv",    type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="results_lstm")
    p.add_argument("--seed",        type=int, default=42)

    # Training
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience",     type=int,   default=15)

    # LR schedule (like Transformer)
    p.add_argument("--lr_warmup_steps", type=int,   default=5)
    p.add_argument("--lr_decay",        type=float, default=0.99)

    # Model architecture
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--num_layers",    type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--bidirectional", type=int,   default=0, choices=[0, 1])

    # Time-weighted loss
    p.add_argument("--use_time_weights", type=int,   default=0, choices=[0, 1])
    p.add_argument("--tw_start",         type=float, default=0.1)
    p.add_argument("--tw_end",           type=float, default=1.0)

    # Threshold / trajectory
    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1"])
    p.add_argument("--focus_stay_id",  type=int, default=None)

    # Future experiments
    p.add_argument("--pe_factor",    type=float, default=1.0)
    p.add_argument("--train_noise",  type=float, default=0.0)

    # Test ID management (like Transformer)
    p.add_argument("--list_test_ids",    type=int, default=0, choices=[0, 1])
    p.add_argument("--save_test_ids_csv", type=int, default=0, choices=[0, 1])
    p.add_argument("--list_only",        type=int, default=0, choices=[0, 1])
    p.add_argument("--save_predictions", type=int, default=0, choices=[0, 1],
                   help="Save test_predictions.csv (y_true, y_prob) for multi-model curve comparison")

    return p.parse_args()


# =============================================================================
# 6. Main
# =============================================================================
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load & preprocess CSV
    # ------------------------------------------------------------------
    print(f"Loading data: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
    print(f"Rows after time_bin filter: {len(df):,}  |  "
          f"Unique stay_ids: {df['stay_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = split_by_stay_id(df, seed=args.seed)
    print(f"Split → train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ------------------------------------------------------------------
    # StandardScaler fit on train only
    # ------------------------------------------------------------------
    scale_cols = [
        c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
        if c not in ["sex", "Vasopressor_use", "Hemodialysis_use"]
    ]
    scaler = StandardScaler().fit(
        df[df["stay_id"].isin(train_ids)][scale_cols]
    )

    # ------------------------------------------------------------------
    # Datasets (with pre-computed mask support)
    # ------------------------------------------------------------------
    ds_train = ExtubationSeqDataset(df, train_ids, scaler, scale_cols, SEQ_TIME_BINS)
    ds_val   = ExtubationSeqDataset(df, val_ids,   scaler, scale_cols, SEQ_TIME_BINS)
    ds_test  = ExtubationSeqDataset(df, test_ids,  scaler, scale_cols, SEQ_TIME_BINS)
    print(f"Samples → train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    # ------------------------------------------------------------------
    # Optional: list / save test IDs
    # ------------------------------------------------------------------
    if args.list_test_ids == 1 or args.save_test_ids_csv == 1:
        rows = []
        for s in ds_test.samples:
            sid  = int(s["sid"])
            lbl  = int(s["y"])
            present_steps = int(np.sum(np.array(s["step_present"]) > 0))
            rows.append({"stay_id": sid, "label": lbl, "present_steps": present_steps})
        df_test_ids = (
            pd.DataFrame(rows)
            .sort_values(["label", "stay_id"])
            .reset_index(drop=True)
        )
        fail_ids = df_test_ids[df_test_ids["label"] == 1]["stay_id"].tolist()
        succ_ids = df_test_ids[df_test_ids["label"] == 0]["stay_id"].tolist()
        print(f"[TEST] Available stay_ids: {len(df_test_ids)} "
              f"(fail={len(fail_ids)}, success={len(succ_ids)})")
        if args.list_test_ids == 1:
            print(f"[TEST] failure examples: {fail_ids[:20]}")
            print(f"[TEST] success examples: {succ_ids[:20]}")
        if args.save_test_ids_csv == 1:
            out_csv = os.path.join(args.output_dir, "test_stay_ids.csv")
            df_test_ids.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"Saved: {out_csv}")

    if args.list_only == 1:
        print("[list_only=1] Test ID output complete — skipping training.")
        return

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=128, shuffle=False)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ExtubationLSTM(
        dyn_dim=52,
        stat_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Loss, optimizer, LR scheduler (like Transformer)
    # ------------------------------------------------------------------
    y_train = np.array([s["y"] for s in ds_train.samples])
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
        device=device
    )
    criterion_mean = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.lr_warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=args.lr_warmup_steps)
        decay_sched  = ExponentialLR(optimizer, gamma=args.lr_decay)
        scheduler    = SequentialLR(optimizer,
                                    schedulers=[warmup_sched, decay_sched],
                                    milestones=[args.lr_warmup_steps])
        print(f"[LR] Warmup {args.lr_warmup_steps} epochs -> "
              f"Exponential decay (gamma={args.lr_decay})")
    else:
        scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc  = -1.0
    bad       = 0
    best_path = os.path.join(args.output_dir, "best_lstm.pt")

    print(f"\nStarting training (device={device}, epochs={args.epochs}) ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for x_dyn, x_stat, y, step_present, _ in train_loader:
            x_dyn        = x_dyn.to(device)
            x_stat       = x_stat.to(device)
            y            = y.to(device)
            step_present = step_present.to(device)

            optimizer.zero_grad()

            if not args.use_time_weights:
                logit = model(x_dyn, x_stat, step_present)
                loss  = criterion_mean(logit, y)
            else:
                logits_list = []
                for k in range(SEQ_LEN):
                    lk = model(x_dyn, x_stat, step_present, prefix_cutoff=k)
                    logits_list.append(lk.unsqueeze(1))
                logits_seq = torch.cat(logits_list, dim=1)          # (B, T, 1)
                tw = torch.linspace(
                    args.tw_start, args.tw_end, SEQ_LEN, device=device
                ).view(1, SEQ_LEN, 1)
                loss = (
                    criterion_none(
                        logits_seq,
                        y.view(-1, 1, 1).repeat(1, SEQ_LEN, 1)
                    ) * tw
                ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches  += 1

        scheduler.step()

        yv, pv, _ = get_probs(model, val_loader, device)
        val_auc = roc_auc_score(yv, pv)
        lr_now  = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
                  f"val AUROC={val_auc:.4f} | lr={lr_now:.2e}")

        # Early abort if model is not learning
        if epoch >= 10 and val_auc < 0.55:
            print(f"[Early Abort] Epoch {epoch}: val AUROC={val_auc:.4f} < 0.55. "
                  f"Stopping early.")
            break

        # Early stopping with 5e-4 improvement threshold (like Transformer)
        if val_auc > best_auc + 5e-4:
            best_auc = float(val_auc)
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}. Best val AUROC={best_auc:.4f}")
                break

    print(f"\nTraining complete. Best val AUROC={best_auc:.4f}")

    # ------------------------------------------------------------------
    # Load best checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model: {best_path}")
    else:
        print("[WARNING] best_lstm.pt not found — using final epoch weights.")

    # ------------------------------------------------------------------
    # Threshold selection on validation set
    # ------------------------------------------------------------------
    yv, pv, _ = get_probs(model, val_loader, device)
    best_thr  = find_best_threshold(yv, pv, mode=args.threshold_mode)
    print(f"Best threshold ({args.threshold_mode}): {best_thr:.4f}")

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    yt, pt, sids = get_probs(model, test_loader, device)

    # ROC + confusion matrix
    roc_path = os.path.join(args.output_dir, "test_roc_cm_lstm.png")
    metrics  = plot_roc_cm(yt, pt, best_thr, roc_path, label="LSTM")

    # Calibration + Brier score
    cal_path     = os.path.join(args.output_dir, "calibration_curve.png")
    brier_metric = plot_calibration_brier(yt, pt, "LSTM", cal_path)
    metrics.update(brier_metric)

    # ------------------------------------------------------------------
    # Save performance_metrics.csv
    # ------------------------------------------------------------------
    ordered_keys = [
        "AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity",
        "Precision", "F1_score", "Recall", "Brier_score",
    ]
    row = {k: metrics.get(k, float("nan")) for k in ordered_keys}
    res_df = pd.DataFrame([row])
    metrics_path = os.path.join(args.output_dir, "performance_metrics.csv")
    res_df.to_csv(metrics_path, index=False)
    print(f"Performance metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Save test predictions（供多模型 ROC/PR 曲線比較用）
    # ------------------------------------------------------------------
    if getattr(args, "save_predictions", 0) == 1:
        pred_df = pd.DataFrame({"stay_id": sids, "y_true": yt.astype(int), "y_prob": pt})
        pred_path = os.path.join(args.output_dir, "test_predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Test predictions saved: {pred_path}")

    # ------------------------------------------------------------------
    # (Optional) Risk trajectory for a specific stay_id
    # ------------------------------------------------------------------
    if args.focus_stay_id is not None:
        traj_path = os.path.join(
            args.output_dir, f"lstm_trajectory_stay_{args.focus_stay_id}.png"
        )
        # Search test first, then val, then train
        for ds_name, ds_obj in [("test", ds_test), ("val", ds_val), ("train", ds_train)]:
            if any(int(s["sid"]) == int(args.focus_stay_id) for s in ds_obj.samples):
                print(f"stay_id {args.focus_stay_id} found in {ds_name} split.")
                plot_trajectory_for_stay(
                    model=model,
                    dataset=ds_obj,
                    target_sid=args.focus_stay_id,
                    device=device,
                    threshold=best_thr,
                    save_path=traj_path,
                    window_hours=WINDOW_HOURS,
                )
                break
        else:
            print(f"[WARNING] stay_id={args.focus_stay_id} not found in any split.")


if __name__ == "__main__":
    main(parse_args())
