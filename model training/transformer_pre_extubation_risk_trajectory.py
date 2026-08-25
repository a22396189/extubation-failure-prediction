#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transformer_pre_extubation_risk_trajectory.py

=============================================================
【目的】
  以 Transformer Encoder 預測 ICU 拔管失敗風險（label=1）。
  採用固定 12 步時間序列（time_bin ∈ {-52,-48,...,-8}，每步代表 [t, t+4h)）。

【輸入架構】
  - Dynamic stream：26 生理變數 + 26 二元缺失 mask = 52 維 / 每步
  - Static stream：age / sex / BMI / Charlson_Score（4 維），經 MLP late fusion

【設計特點（相較於 OPSUM，Klug et al. 2024）】
  ✅ 26 Missingness Masks：明確告知模型哪些值為缺失補值（OPSUM 無此設計）
  ✅ Late Fusion Static：靜態特徵在 Encoder output 後才融合（語意更清晰）
  ✅ Prefix Cumulative Training（--use_time_weights 1）：動態風險軌跡學習
  ✅ Threshold 多模式選擇（Youden / F1 / F2 / Sens=Spec）

【參考自 OPSUM 新增的設計】
  ✅ PE scaling factor（--pe_factor，預設 0.1）：縮放 sinusoidal PE 避免淹沒特徵
  ✅ Training noise（--train_noise）：訓練時注入 Gaussian 雜訊，提升正則化效果
  ✅ LR warmup + Exponential decay（--lr_warmup_steps / --lr_decay）
  ✅ Pooling 策略選擇（--pooling last/mean）
  ✅ Early abort safeguard：epoch ≥ 10 且 val AUROC < 0.55 時提前終止

【Key options】
  --use_time_weights 0/1
    0: many-to-one（stay-level BCE loss）
    1: prefix cumulative（每個前綴 k=0..11 各計算 logit，套用時間權重 BCE）

  --use_causal_mask 0/1
    0: Bidirectional Encoder（全序列注意力）
    1: Causal mask（只能看到過去步，與 prefix inference 更一致）

  --pooling last/mean
    last: 取最後一個有資料的 step 的 hidden state（預設）
    mean: 對所有有資料的 step 取加權平均

  --pe_factor FLOAT
    Positional Encoding 縮放倍率（OPSUM: 0.001~0.1；預設 0.1）

  --train_noise FLOAT
    訓練時對 x_dyn 注入 Gaussian noise 標準差（0=關閉）

【Outputs（在 --output_dir 下）】
  - best_transformer.pt
  - test_roc_cm.png
  - calibration_curve.png
  - performance_metrics.csv
  - trajectory_stay_<ID>.png（若提供 --focus_stay_id）

【Notes】
  - StandardScaler 僅在 TRAIN set fit，再 transform val/test（無 leakage）
  - 二元變數（sex / Vasopressor_use / Hemodialysis_use）不做 z-score 標準化
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, accuracy_score, f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )

warnings.filterwarnings("ignore")

# =========================
# 0. Feature config
# =========================
STATIC_COLS = ["age", "sex", "BMI", "Charlson_Score"]

DYNAMIC_COLS = [
    "heart_rate", "resp_rate", "spo2", "mbp", "temperature", "GCS",
    "FiO2", "MAP", "PEEP", "TV_per_kg", "MV_day", "pH", "PaO2",
    "PaCO2", "BE", "OI", "Cr", "WBC", "Hb", "PLT", "AnionGap",
    "Lactate", "Glucose", "io_balance", "Vasopressor_use", "Hemodialysis_use"
]

# Pre-computed missingness mask 欄位名稱（由 impute 腳本在填補前建立）
# mask_{col} = 1：該特徵在此 bin 有原始量測值（not NaN）
#            = 0：該特徵在此 bin 原本缺失（已被填補）
# bin_has_data = 1：該 time_bin 至少有一個 dynamic feature 有量測值
MASK_COLS = [f"mask_{col}" for col in DYNAMIC_COLS]

TARGET = "Extubation_failure"

# Each time_bin represents [t, t+4h); using up to -8 makes last window [-8,-4) (gap=4h leakage-safe)
SEQ_TIME_BINS = list(range(-52, -4, 4))  # -52, -48, ..., -8
SEQ_LEN = len(SEQ_TIME_BINS)
WINDOW_HOURS = 4


# =========================
# 1. CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_csv", type=str, required=True,
                   help="Input CSV containing stay_id, time_bin, features, and Extubation_failure.")
    p.add_argument("--output_dir", type=str, default="results_transformer",
                   help="Directory to write outputs.")
    p.add_argument("--seed", type=int, default=42)

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8)

    # model architecture
    p.add_argument("--d_model", type=int, default=64,
                   help="Transformer hidden dim. OPSUM uses 128; smaller may be better for N~6k.")
    p.add_argument("--nhead", type=int, default=4,
                   help="Attention heads (must divide d_model). OPSUM uses 8.")
    p.add_argument("--num_layers", type=int, default=3,
                   help="Encoder layers. OPSUM uses 6.")
    p.add_argument("--dim_ff", type=int, default=128,
                   help="Feedforward dim. OPSUM uses 256.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_causal_mask", type=int, default=0, choices=[0, 1],
                   help="1=causal self-attention (past only), 0=bidirectional.")

    # PE scaling (OPSUM-inspired)
    p.add_argument("--pe_factor", type=float, default=0.1,
                   help="Positional Encoding scaling factor (OPSUM explores 1e-3~1.0). "
                        "Smaller values prevent PE from overwhelming features in short sequences.")

    # training noise (OPSUM-inspired regularisation)
    p.add_argument("--train_noise", type=float, default=0.0,
                   help="Std of Gaussian noise injected into x_dyn during training (0=off). "
                        "OPSUM uses small values for regularisation.")

    # LR schedule (OPSUM-inspired)
    p.add_argument("--lr_warmup_steps", type=int, default=5,
                   help="Linear LR warmup epochs (OPSUM style). 0=no warmup.")
    p.add_argument("--lr_decay", type=float, default=0.99,
                   help="Exponential LR decay factor per epoch (OPSUM uses 0.99).")

    # pooling strategy (OPSUM-inspired, keep 'last' as default to preserve original design)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"],
                   help="'last': last present token (original design); "
                        "'mean': masked mean over all present tokens.")

    # time-weights switch
    p.add_argument("--use_time_weights", type=int, default=0, choices=[0, 1],
                   help="0=stay-level loss (many-to-one), 1=prefix loss + time weights (cumulative).")
    p.add_argument("--tw_start", type=float, default=0.1)
    p.add_argument("--tw_end", type=float, default=1.0)

    # threshold
    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1", "f2", "sens_spec"],
                   help="Threshold selection on VAL: youden, f1, f2, sens_spec (sens≈spec intersection).")

    # optional trajectory
    p.add_argument("--focus_stay_id", type=int, default=None,
                   help="If provided, plot cumulative trajectory for this stay_id (must exist in TEST).")

    # print test IDs
    p.add_argument("--print_test_ids", type=int, default=0, choices=[0, 1],
                   help="If 1, print some test stay_ids grouped by label.")
    
    # print list of test set stay_id 
    p.add_argument("--list_test_ids", type=int, default=0, choices=[0,1],
               help="If 1, print all test stay_ids (available for trajectory) and exit.")
    p.add_argument("--max_list", type=int, default=100,
                help="Max number of IDs to print per group when listing.")
    p.add_argument("--save_test_ids_csv", type=int, default=0, choices=[0,1],
                help="If 1, save test stay_id list (available for trajectory) to CSV.")
    p.add_argument("--save_predictions", type=int, default=0, choices=[0, 1],
                   help="If 1, save test set predictions (y_true, y_prob) to "
                        "test_predictions.csv for multi-model calibration comparison.")
    p.add_argument("--list_only", type=int, default=0, choices=[0, 1],
                   help="If 1, build dataset, save test_stay_ids.csv, then exit WITHOUT training. "
                        "Useful for quickly finding valid --focus_stay_id values.")

    return p.parse_args()


# =========================
# 2. Reproducibility
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. Threshold selection
# =========================
def find_best_threshold(y_true, y_prob, mode="youden"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        return float(thr[np.argmax(j)])

    if mode == "sens_spec":
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        sens = tpr
        spec = 1.0 - fpr
        return float(thr[np.argmin(np.abs(sens - spec))])

    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]

    if mode == "f1":
        beta = 1.0
    elif mode == "f2":
        beta = 2.0
    else:
        raise ValueError("mode must be one of: youden, f1, f2, sens_spec")

    denom = (beta**2 * precision + recall)
    denom = np.where(denom == 0, 1e-12, denom)
    fbeta = (1 + beta**2) * precision * recall / denom

    if len(thr) == 0:
        return 0.5
    return float(thr[np.argmax(fbeta)])


# =========================
# 4. Split by stay_id (no leakage)
# =========================
def split_by_stay_id(df, train_ratio=0.7, seed=42):
    """
    以 stay_id 為單位做 stratified split（70 / 15 / 15）。

    Subject-level leakage 說明：
    pipeline 第一階段 filter_unique_subject.py 已確保每個 subject_id
    只對應唯一一個 stay_id（保留 endtime 最早的那筆），故以 stay_id
    切分即等同以 subject_id 切分，不存在 subject-level leakage 風險。

    此函數在切分前以輕量驗證確認該假設成立（若上游出錯會發出警告）。
    """
    # 輕量驗證：確認上游 filter_unique_subject.py 的假設成立
    if "subject_id" in df.columns:
        subj_stay_count = df.groupby("subject_id")["stay_id"].nunique()
        multi_stay_count = int((subj_stay_count > 1).sum())
        if multi_stay_count > 0:
            print(f"[WARNING] 發現 {multi_stay_count} 個 subject_id 對應多個 stay_id！"
                  f"請檢查上游 filter_unique_subject.py 是否正確執行。")
        else:
            print(f"[OK] Subject-level leakage 驗證通過：每個 subject_id 僅對應 1 個 stay_id"
                  f"（共 {len(subj_stay_count)} 位病人）。")

    stay_labels = df.groupby("stay_id")[TARGET].first().reset_index()
    y = stay_labels[TARGET].values

    train_ids, temp_ids = train_test_split(
        stay_labels["stay_id"].values,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=y
    )

    temp_labels = stay_labels[stay_labels["stay_id"].isin(temp_ids)]
    y_temp = temp_labels[TARGET].values

    val_ids, test_ids = train_test_split(
        temp_labels["stay_id"].values,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )
    return train_ids, val_ids, test_ids


# =========================
# 5. Dataset (fixed bins + mask)
# =========================
class ExtubationSeqDataset(Dataset):
    def __init__(self, df, stay_ids, scaler, scale_cols, seq_time_bins):
        self.df = df[df["stay_id"].isin(stay_ids)].copy()
        self.seq_time_bins = list(seq_time_bins)
        self.stay_ids = sorted(list(set(stay_ids)))

        self.scaler = scaler
        self.scale_cols = list(scale_cols) if scale_cols is not None else []

        if self.scaler is not None and len(self.scale_cols) > 0 and len(self.df) > 0:
            self.df.loc[:, self.scale_cols] = self.scaler.transform(self.df[self.scale_cols])

        # 偵測是否有 impute 腳本預先計算的 missingness mask 欄位
        # True  → 從 CSV mask 欄位讀取（impute 後執行，正確）
        # False → 從 NaN 即時計算（僅在原始未填補資料上正確，向後相容）
        self.has_precomputed_masks = (
            all(col in self.df.columns for col in MASK_COLS)
            and "bin_has_data" in self.df.columns
        )
        if self.has_precomputed_masks:
            print("[Dataset] ✅ 偵測到預計算 missingness mask（mask_* + bin_has_data），"
                  "從 CSV 讀取（正確反映填補前的缺失位置）")
        else:
            print("[Dataset] ⚠️  未偵測到預計算 mask 欄位，改以即時 NaN 計算（"
                  "若資料已填補，mask 將全為 1，請確認是否使用未填補的原始資料）")

        self.samples = []
        for sid in self.stay_ids:
            s = self._build_one(int(sid))
            if s is not None:
                self.samples.append(s)

    def _build_one(self, sid: int):
        d = self.df[self.df["stay_id"] == sid].copy()
        if d.empty:
            return None

        y = int(d[TARGET].iloc[0])
        stat = d[STATIC_COLS].iloc[0].copy()

        seq_values = []
        seq_step_present = []

        for tb in self.seq_time_bins:
            row = d[d["time_bin"] == tb]

            if row.empty:
                # 整個 bin 完全不存在（特徵提取階段就沒有資料）
                dyn_filled    = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)
                mask          = np.zeros(len(DYNAMIC_COLS), dtype=np.float32)  # 全缺失
                step_pres_val = 0.0

            elif self.has_precomputed_masks:
                # ── 正確路徑：從 impute 腳本預計算的 mask 欄位讀取 ──────────────
                # 此路徑確保 mask 反映「填補前的原始缺失位置」，
                # 而非填補後（NaN 已不存在）的錯誤計算結果。
                dyn_filled    = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                mask          = row[MASK_COLS].iloc[0].values.astype(np.float32)
                # bin_has_data：至少一個 feature 有原始量測值 → step 有效
                step_pres_val = float(row["bin_has_data"].iloc[0])

            else:
                # ── Fallback：即時從 NaN 計算（僅在原始未填補資料上正確）──────
                dyn_raw       = row[DYNAMIC_COLS].iloc[0].values.astype(np.float32)
                mask          = (~np.isnan(dyn_raw)).astype(np.float32)
                dyn_filled    = np.nan_to_num(dyn_raw, nan=0.0).astype(np.float32)
                step_pres_val = 1.0 if mask.sum() > 0 else 0.0

            # x_dyn = [26 feature values | 26 presence masks]，共 52 維
            dyn_combined = np.concatenate([dyn_filled, mask], axis=0).astype(np.float32)
            seq_values.append(dyn_combined)
            seq_step_present.append(step_pres_val)

        seq_values        = np.stack(seq_values, axis=0)           # (T, 52)
        seq_step_present  = np.array(seq_step_present, dtype=np.float32)  # (T,)

        stat = stat.fillna(0).values.astype(np.float32)

        return {
            "sid":          int(sid),
            "x_dyn":        seq_values,
            "x_stat":       stat,
            "y":            float(y),
            "step_present": seq_step_present
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_dyn = torch.tensor(s["x_dyn"], dtype=torch.float32)
        x_stat = torch.tensor(s["x_stat"], dtype=torch.float32)
        y = torch.tensor([s["y"]], dtype=torch.float32)
        step_present = torch.tensor(s["step_present"], dtype=torch.float32)
        sid = int(s["sid"])
        return x_dyn, x_stat, y, step_present, sid


# =========================
# 6. Model
# =========================
class SinusoidalPositionalEncoding(nn.Module):
    """
    標準 sinusoidal PE，加入 factor 縮放倍率（OPSUM-inspired）。

    動機：本模型序列長度僅 12 bins，位置資訊已相對明確；
    過大的 PE 值可能壓過特徵本身的數值尺度。
    OPSUM 探索範圍 1e-3 ～ 1.0，預設 0.1。
    """
    def __init__(self, d_model, max_len=512, factor=0.1):
        super().__init__()
        self.factor = factor
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.factor * self.pe[:, :T, :]


class ExtubationTransformer(nn.Module):
    """
    Transformer Encoder for extubation failure prediction.

    Design principles:
    - Dynamic stream：26 vars + 26 missingness masks = 52 dim（26 masks 保留缺失資訊）
    - Static stream：Late fusion via MLP projector，在 Encoder output 後才融合
    - PE scaling：factor 縮放 sinusoidal PE，防止 PE 壓過特徵（OPSUM-inspired）
    - Training noise：訓練時注入 Gaussian noise，強化正則化（OPSUM-inspired）
    - Pooling：last-present-token（原設計）或 masked mean（新增選項）
    """
    def __init__(
        self,
        dyn_dim=52,
        stat_dim=4,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_ff=128,
        dropout=0.1,
        use_causal_mask=False,
        pe_factor=0.1,
        pooling="last",
        train_noise=0.0,
    ):
        super().__init__()
        self.use_causal_mask = bool(use_causal_mask)
        self.pooling = pooling
        self.train_noise = float(train_noise)

        self.dyn_proj = nn.Linear(dyn_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=256, factor=pe_factor)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Static MLP projector（late fusion，保留原設計）
        self.stat_proj = nn.Sequential(
            nn.Linear(stat_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x_dyn, x_stat, step_present=None, prefix_cutoff=None,
                return_embedding=False):
        """
        Parameters
        ----------
        x_dyn          : (B, T, 52) 動態特徵（26 vars + 26 masks）
        x_stat         : (B, 4)     靜態特徵（age/sex/BMI/CCI）
        step_present   : (B, T)     各 step 是否有資料（>0=有）
        prefix_cutoff  : int        Prefix cumulative inference 的截止 step 索引
        return_embedding: bool      True → 回傳 pooled encoder output（B, d_model），
                                    供 clustering 使用；False → 回傳分類 logit（B, 1）
        """
        B, T, _ = x_dyn.shape

        # Training noise injection（OPSUM-inspired 正則化）
        if self.training and self.train_noise > 0.0:
            x_dyn = x_dyn + torch.randn_like(x_dyn) * self.train_noise

        h = self.dyn_proj(x_dyn)
        h = self.pos(h)  # PE scaling 已在 SinusoidalPositionalEncoding 內套用

        # Key padding mask：step 完全無資料（全 NaN bin）→ 不參與 attention
        key_padding_mask = None
        if step_present is not None:
            key_padding_mask = (step_present <= 0.0)

        # Prefix cutoff：用於 prefix cumulative inference，遮蔽未來 steps
        if prefix_cutoff is not None:
            future_mask = torch.arange(T, device=x_dyn.device).unsqueeze(0) > int(prefix_cutoff)
            key_padding_mask = future_mask if key_padding_mask is None else (key_padding_mask | future_mask)

        # Causal mask（OPSUM 與本設計均支援，預設關閉）
        attn_mask = None
        if self.use_causal_mask:
            attn_mask = torch.triu(torch.ones(T, T, device=x_dyn.device) * float("-inf"), diagonal=1)

        z = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)

        # ── Pooling ──────────────────────────────────────────────────────────
        if self.pooling == "mean":
            # Masked mean：對所有「有資料」的 step 取平均（OPSUM-inspired）
            if step_present is None:
                present_mask = torch.ones(B, T, device=x_dyn.device)
            else:
                present_mask = (step_present > 0.0).float()
                if prefix_cutoff is not None:
                    cutoff_mask = (torch.arange(T, device=x_dyn.device).unsqueeze(0)
                                   <= int(prefix_cutoff)).float()
                    present_mask = present_mask * cutoff_mask
            present_mask_3d = present_mask.unsqueeze(-1)               # (B, T, 1)
            pooled = (z * present_mask_3d).sum(dim=1) / present_mask_3d.sum(dim=1).clamp(min=1e-6)

        else:
            # Last-present-token（原設計，預設）
            if step_present is None:
                last_idx = torch.full((B,), T - 1, dtype=torch.long, device=x_dyn.device)
            else:
                present = (step_present > 0.0)
                if prefix_cutoff is not None:
                    present = present & (torch.arange(T, device=x_dyn.device).unsqueeze(0) <= int(prefix_cutoff))
                idx = present.float() * torch.arange(T, device=x_dyn.device).unsqueeze(0)
                last_idx = idx.max(dim=1).values.long()
            pooled = z[torch.arange(B, device=x_dyn.device), last_idx, :]
        # ─────────────────────────────────────────────────────────────────────

        # ── Return embedding for clustering ──────────────────────────────────
        # return_embedding=True：回傳純動態軌跡 embedding（B, d_model）
        #   - 供 extubation_failure_phenotyping.py 做 clustering
        #   - 靜態特徵是否納入 clustering 由呼叫端決定（concat 至此向量，或保持純動態）
        #   - 不經過 stat_proj / classifier，保留最大的語意完整性
        if return_embedding:
            return pooled  # (B, d_model)

        # Late fusion：static MLP 輸出與 pooled dynamic 拼接
        s = self.stat_proj(x_stat)
        out = self.classifier(torch.cat([pooled, s], dim=1))
        return out


# =========================
# 7. Helpers
# =========================
def get_probs(model, loader, device):
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
    return np.array(labels).astype(int), np.array(probs).astype(float), sids


def plot_roc_cm(y_true, y_prob, threshold, out_path):
    """
    產生包含完整指標的 ROC 曲線與混淆矩陣
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 指標計算
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "AUROC": auroc, "AUPRC": auprc, "Accuracy": accuracy,
        "Sensitivity": sensitivity, "Specificity": specificity,
        "Precision": precision, "Recall": recall, "F1 score": f1
    }

    print("\n" + "="*30)
    print(f"測試集性能指標 (Threshold={threshold:.4f}):")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")
    print("="*30)

    # 繪圖
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # 左圖：ROC Curve (參考樣式)
    ax[0].plot(fpr, tpr, color='#1f77b4', lw=3, label=f"Transformer (AUROC={auroc:.4f})")
    ax[0].fill_between(fpr, tpr, alpha=0.2, color='#1f77b4')
    ax[0].plot([0, 1], [0, 1], color='black', lw=1.5, linestyle="--", label="Random")
    ax[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax[0].set_title("ROC Curve", fontweight="bold", fontsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.4)
    ax[0].legend(loc="lower right")

    # 右圖：Confusion Matrix (參考樣式)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1], cbar_kws={'label': 'Count'})
    ax[1].set_xticklabels(['Pred: Success (0)', 'Pred: Failure (1)'])
    ax[1].set_yticklabels(['True: Success (0)', 'True: Failure (1)'], va='center')
    ax[1].set_title(f"Confusion Matrix\n(threshold={threshold:.3f})", fontweight="bold", fontsize=14)
    ax[1].set_xlabel("Predicted Label")
    ax[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return metrics


def plot_calibration_brier(y_true, y_prob, model_name, out_path,
                           n_bins=10, n_bootstrap=1000, seed=42):
    """
    Publication-quality Calibration Curve（Reliability Diagram）with Bootstrap 95% CI.
    Layout:
      - Upper panel (3/4): calibration curve + CI band + perfect calibration line
      - Lower panel (1/4): predicted probability histogram (success vs failure)
    - Brier Score：越小越好（0=完美，0.25=隨機；隨機猜測約 0.25）
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = brier_score_loss(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # ── Bootstrap 95% CI ──────────────────────────────────────────
    boot_frac = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            fp, mp = calibration_curve(y_true[idx], y_prob[idx],
                                       n_bins=n_bins, strategy="uniform")
            fp_interp = np.interp(mean_pred, mp, fp)
            boot_frac.append(fp_interp)
        except Exception:
            continue
    boot_frac = np.array(boot_frac)
    ci_lo = np.percentile(boot_frac, 2.5,  axis=0)
    ci_hi = np.percentile(boot_frac, 97.5, axis=0)

    # ── Figure layout: single panel (calibration curve only) ──
    fig, ax_cal = plt.subplots(figsize=(6, 5))

    # ── Calibration curve ─────────────────────────────────────────
    ax_cal.plot(mean_pred, frac_pos, "s-", color="#1f77b4", lw=2, ms=7, zorder=3,
                label=f"{model_name}  (Brier = {brier:.4f})")
    ax_cal.fill_between(mean_pred, ci_lo, ci_hi,
                        alpha=0.18, color="#1f77b4", label="95% Bootstrap CI")
    ax_cal.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax_cal.set_xlabel("Mean Predicted Probability", fontsize=13)
    ax_cal.set_ylabel("Fraction of Positives", fontsize=13)
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    ax_cal.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax_cal.grid(alpha=0.3, linestyle="--")
    ax_cal.set_title("Calibration Curve",
                     fontsize=14, fontweight="bold", pad=8)
    for sp in ["top", "right"]:
        ax_cal.spines[sp].set_visible(False)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Brier Score: {brier:.4f}  (儲存至: {out_path})")
    return {"brier_score": float(brier)}


def plot_trajectory_for_stay(model, dataset, target_sid, device, threshold, save_path, window_hours=4):
    model.eval()

    sample = next((s for s in dataset.samples if int(s["sid"]) == int(target_sid)), None)
    if sample is None:
        raise ValueError(f"stay_id={target_sid} not found in this dataset split.")

    x_dyn = torch.tensor(sample["x_dyn"], dtype=torch.float32).unsqueeze(0).to(device)
    x_stat = torch.tensor(sample["x_stat"], dtype=torch.float32).unsqueeze(0).to(device)
    step_present = torch.tensor(sample["step_present"], dtype=torch.float32).unsqueeze(0).to(device)
    true_label = int(sample["y"])
    T = x_dyn.shape[1]

    probs = []
    with torch.no_grad():
        for k in range(T):
            logit = model(x_dyn, x_stat, step_present=step_present, prefix_cutoff=k)
            probs.append(float(torch.sigmoid(logit).cpu().numpy().reshape(-1)[0]))

    start_bins = list(getattr(dataset, "seq_time_bins", SEQ_TIME_BINS))
    x_mid = [tb + (window_hours / 2) for tb in start_bins]  # -50,-46,...,-6

    plt.figure(figsize=(12, 6))
    plt.plot(x_mid, probs, marker="o", linewidth=2.5, label="Predicted failure risk")
    plt.axhline(y=threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold={threshold:.3f}")
    # plt.axvline(x=-4, color="teal", linestyle="--", linewidth=2, label="Last available data boundary (t=-4h)")
    plt.axvline(x=0, color="black", linestyle=":", linewidth=2, label="Extubation time")

    xticks = list(range(-52, 5, 4))  # includes 4
    xlabels = [str(t) for t in xticks]
    xlabels[-1] = ""  # hide +4 label
    plt.xticks(xticks, xlabels)

    plt.ylim(-0.05, 1.05)
    plt.xlim(-56, 4)
    plt.grid(alpha=0.3, linestyle="--")

    plt.xlabel("Window midpoint (hours before extubation); each point represents a 4h window")
    plt.ylabel("Predicted failure risk P(label=1)")
    plt.title(f"Stay {target_sid} Trajectory (True label={true_label})", fontweight="bold")
    plt.legend(loc="best")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Trajectory saved: {save_path}")


# =========================
# 8. Main
# =========================
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    df = pd.read_csv(args.data_csv)

    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)

    df[TARGET] = df[TARGET].astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
    if df.empty:
        raise ValueError("After filtering by SEQ_TIME_BINS, dataframe is empty. Check time_bin values.")

    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.7, seed=args.seed)
    train_df = df[df["stay_id"].isin(train_ids)].copy()
    val_df   = df[df["stay_id"].isin(val_ids)].copy()
    test_df  = df[df["stay_id"].isin(test_ids)].copy()

    binary_like = ["sex", "Vasopressor_use", "Hemodialysis_use"]
    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"]) if c not in binary_like]

    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])

    ds_train = ExtubationSeqDataset(train_df, train_ids, scaler, scale_cols, SEQ_TIME_BINS)
    ds_val   = ExtubationSeqDataset(val_df,   val_ids,   scaler, scale_cols, SEQ_TIME_BINS)
    ds_test  = ExtubationSeqDataset(test_df,  test_ids,  scaler, scale_cols, SEQ_TIME_BINS)

    print(f"[SEQ] time bins: {SEQ_TIME_BINS} (len={SEQ_LEN})")
    print(f"Train stays: {len(ds_train)} | Val stays: {len(ds_val)} | Test stays: {len(ds_test)}")
    print(f"Failure rate train/val/test: "
          f"{np.mean([s['y'] for s in ds_train.samples]):.3f} / "
          f"{np.mean([s['y'] for s in ds_val.samples]):.3f} / "
          f"{np.mean([s['y'] for s in ds_test.samples]):.3f}")

    # ===== list test stay_ids available for trajectory (from ds_test.samples) =====
    if args.list_test_ids == 1:
        rows = []
        for s in ds_test.samples:
            sid = int(s["sid"])
            y = int(s["y"])  # Extubation_failure: 1=fail
            present_steps = int(np.sum(np.array(s["step_present"]) > 0))
            rows.append({"stay_id": sid, "label": y, "present_steps": present_steps})

        df_test_ids = pd.DataFrame(rows).sort_values(["label", "stay_id"]).reset_index(drop=True)

        fail_ids = df_test_ids[df_test_ids["label"] == 1]["stay_id"].tolist()
        succ_ids = df_test_ids[df_test_ids["label"] == 0]["stay_id"].tolist()

        print(f"[TEST] Available trajectory stay_ids: {len(df_test_ids)} "
            f"(fail={len(fail_ids)}, success={len(succ_ids)})")

        m = int(args.max_list)
        print(f"[TEST] failure examples (up to {m}): {fail_ids[:m]}")
        print(f"[TEST] success examples (up to {m}): {succ_ids[:m]}")

        if args.save_test_ids_csv == 1:
            out_csv = os.path.join(args.output_dir, "test_stay_ids.csv")
            df_test_ids.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"✓ Saved: {out_csv}")

    # --list_only=1：僅輸出 test ID，不進行訓練
    if getattr(args, "list_only", 0) == 1:
        print("\n[list_only=1] Test ID 已輸出，跳過訓練。程式結束。")
        return

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(ds_test,  batch_size=256, shuffle=False)

    model = ExtubationTransformer(
        dyn_dim=52, stat_dim=len(STATIC_COLS),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_ff=args.dim_ff, dropout=args.dropout,
        use_causal_mask=bool(args.use_causal_mask),
        pe_factor=args.pe_factor,
        pooling=args.pooling,
        train_noise=args.train_noise,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] d_model={args.d_model} | nhead={args.nhead} | layers={args.num_layers} "
          f"| dim_ff={args.dim_ff} | dropout={args.dropout}")
    print(f"[Model] pe_factor={args.pe_factor} | pooling={args.pooling} "
          f"| train_noise={args.train_noise} | causal={bool(args.use_causal_mask)}")
    print(f"[Model] Trainable parameters: {total_params:,}")

    y_train = np.array([s["y"] for s in ds_train.samples]).astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    criterion_mean = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_none = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    use_time_weights = bool(args.use_time_weights)
    time_weights = torch.linspace(args.tw_start, args.tw_end, steps=SEQ_LEN, device=device).view(1, SEQ_LEN, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR Schedule：Linear warmup + Exponential decay（OPSUM-inspired）
    from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
    if args.lr_warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=args.lr_warmup_steps)
        decay_sched  = ExponentialLR(optimizer, gamma=args.lr_decay)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, decay_sched],
                                 milestones=[args.lr_warmup_steps])
        print(f"[LR] Warmup {args.lr_warmup_steps} epochs → Exponential decay (γ={args.lr_decay})")
    else:
        scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
        print(f"[LR] Exponential decay only (γ={args.lr_decay})")

    best_auc = -1.0
    best_path = os.path.join(args.output_dir, "best_transformer.pt")
    bad = 0

    print(f"[Train] use_time_weights={int(use_time_weights)} | use_causal_mask={int(bool(args.use_causal_mask))} "
          f"| pooling={args.pooling} | train_noise={args.train_noise}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x_dyn, x_stat, y, step_present, _sid in train_loader:
            x_dyn = x_dyn.to(device)
            x_stat = x_stat.to(device)
            y = y.to(device)
            step_present = step_present.to(device)

            optimizer.zero_grad()

            if not use_time_weights:
                logit = model(x_dyn, x_stat, step_present=step_present)
                loss = criterion_mean(logit, y)
            else:
                logits_list = []
                for k in range(SEQ_LEN):
                    lk = model(x_dyn, x_stat, step_present=step_present, prefix_cutoff=k)
                    logits_list.append(lk.unsqueeze(1))
                logits_seq = torch.cat(logits_list, dim=1)  # (B,T,1)

                y_seq = y.view(-1, 1, 1).repeat(1, SEQ_LEN, 1)
                loss_mat = criterion_none(logits_seq, y_seq)
                loss = (loss_mat * time_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()  # LR decay（OPSUM-inspired）

        yv, pv, _ = get_probs(model, val_loader, device)
        val_auc = roc_auc_score(yv, pv)
        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | train loss={total_loss/len(train_loader):.4f} "
                  f"| val AUROC={val_auc:.4f} | lr={current_lr:.2e}")

        # Early abort safeguard（OPSUM-inspired）：
        # epoch ≥ 10 時若 val AUROC 仍低於 0.55，代表模型未有效學習，提前終止
        if epoch >= 10 and val_auc < 0.55:
            print(f"[Early Abort] Epoch {epoch}: val AUROC={val_auc:.4f} < 0.55. 模型未有效收斂，終止訓練。")
            break

        if val_auc > best_auc + 5e-4:
            best_auc = float(val_auc)
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}. Best val AUROC={best_auc:.4f}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"✓ Loaded best model: {best_path} (best val AUROC={best_auc:.4f})")

    yv, pv, _ = get_probs(model, val_loader, device)
    best_thr = find_best_threshold(yv, pv, mode=args.threshold_mode)
    print(f"[VAL] threshold_mode={args.threshold_mode} -> best_thr={best_thr:.4f}")

    # --- main 函數評估段落 ---
    yt, pt, _ = get_probs(model, test_loader, device)
    
    # 呼叫更新後的繪圖與指標計算函數
    test_metrics = plot_roc_cm(yt, pt, best_thr, os.path.join(args.output_dir, "test_roc_cm.png"))

    # Calibration curve + Brier Score
    cal_metrics = plot_calibration_brier(
        yt, pt, model_name="Transformer",
        out_path=os.path.join(args.output_dir, "calibration_curve.png")
    )
    test_metrics.update(cal_metrics)

    # 加入 best_val_AUROC，供 hyperparam search 以 val 指標排名（避免 test set leakage）
    test_metrics["best_val_AUROC"] = float(best_auc)

    # 儲存結果為 CSV
    res_df = pd.DataFrame([test_metrics])
    csv_out = os.path.join(args.output_dir, "performance_metrics.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"✓ 完整指標（含 Brier Score 與 best_val_AUROC）已儲存至: {csv_out}")

    # 儲存 test set 預測值（供多模型校準曲線比較用）
    if getattr(args, "save_predictions", 0) == 1:
        pred_df = pd.DataFrame({"y_true": yt.astype(int), "y_prob": pt})
        pred_csv = os.path.join(args.output_dir, "test_predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"✓ Test predictions 已儲存至: {pred_csv}")

    if args.focus_stay_id is not None:
        traj_path = os.path.join(args.output_dir, f"trajectory_stay_{int(args.focus_stay_id)}.png")
        plot_trajectory_for_stay(model, ds_test, int(args.focus_stay_id), device, best_thr, traj_path, window_hours=WINDOW_HOURS)


if __name__ == "__main__":
    """
    ─── 執行範例 ────────────────────────────────────────────────────────────────

    # [基本設定]（保留原設計 + OPSUM 新增預設值）
    python "%EXTUBATION_PROJECT_ROOT%/model training/transformer_pre_extubation_risk_trajectory.py" \
    --data_csv "%EXTUBATION_PROJECT_ROOT%/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
    --output_dir "%EXTUBATION_PROJECT_ROOT%/results/transformer_0622" \
    --d_model 64 \
    --nhead 4 \
    --num_layers 3 \
    --dim_ff 128 \
    --dropout 0.2 \
    --lr 5e-5 \
    --pe_factor 1.0 \
    --lr_warmup_steps 5 \
    --lr_decay 0.99 \
    --pooling last \
    --use_time_weights 0 \
    --use_causal_mask 0 \
    --threshold_mode youden \
    --focus_stay_id 30015288

    # [OPSUM 風格較大模型]（供 ablation 比較）
    python "C:/Users/your-username/Desktop/extubation_project_revised/model training/transformer_pre_extubation_risk_trajectory.py" \
      --data_csv "C:/Users/your-username/Desktop/extubation_project_revised/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
      --output_dir "C:/Users/your-username/Desktop/extubation_project_revised/results/transformer_large" \
      --d_model 128 --nhead 8 --num_layers 4 --dim_ff 256 \
      --dropout 0.2 --lr 1e-4 \
      --pe_factor 0.01 \
      --train_noise 0.02 \
      --lr_warmup_steps 5 --lr_decay 0.99 \
      --pooling mean \
      --use_time_weights 1 --use_causal_mask 0 \
      --threshold_mode youden

    # [Prefix cumulative training + causal mask]（動態風險軌跡學習）
    python "%EXTUBATION_PROJECT_ROOT%/model training/transformer_pre_extubation_risk_trajectory.py" \
      --data_csv "%EXTUBATION_PROJECT_ROOT%/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv" \
      --output_dir "%EXTUBATION_PROJECT_ROOT%/results/transformer_0528" \
      --d_model 128 \
      --nhead 4 \
      --num_layers 2 \
      --dim_ff 256 \
      --dropout 0.323593564991412 \
      --weight_decay 0.00198653514661317 \
      --train_noise 0.0905167446200122 \
      --lr 0.000375455940828661 \
      --pe_factor 0.99210135508788 \
      --lr_warmup_steps 4 \
      --lr_decay 0.977256272747636 \
      --pooling last \
      --use_time_weights 1 \
      --use_causal_mask 1 \
      --threshold_mode youden \
      --focus_stay_id 30015288

    ─────────────────────────────────────────────────────────────────────────────
    """
        
    args = parse_args()
    main(args)
