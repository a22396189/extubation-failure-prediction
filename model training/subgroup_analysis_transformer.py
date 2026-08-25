# -*- coding: utf-8 -*-
"""
subgroup_analysis_transformer.py

對已訓練好的 Transformer 模型（transformer_pre_extubation_risk_trajectory.py
產生的 best_transformer.pt）在 test set 上做 Subgroup Analysis。

【不重新訓練模型，只做推論】— 與 plot_trajectory_inference_only.py /
plot_phenotype_risk_trajectories.py 相同模式：
  1. 以相同 seed 重建 train/val/test split（stay_id-level, stratified）
  2. 以 train set fit StandardScaler（與訓練腳本完全一致，無 leakage）
  3. 載入模型權重，在 val set 上重新尋找 Youden threshold
     （訓練腳本未把 threshold 存進 performance_metrics.csv，故於此重算）
  4. 在 test set 上取得每位病人的預測機率
  5. 依臨床分組變數（年齡 / 性別 / BMI / Charlson 共病指數 /
     升壓藥使用 / 腎替代治療使用 / 主診斷大類 DxGroup_Major）
     計算各子群的 AUROC / AUPRC / Sensitivity / Specificity / F1 / Brier
     （含 bootstrap 95% CI），輸出成 CSV 與 Forest Plot。

【輸入】
  --data_csv   訓練所用的 imputed feature csv
               （例如 data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv）
  --model_pt   訓練產生的 best_transformer.pt
  --dx_csv     （可選）extubation_features_final_categories.csv，
               用於合併 DxGroup_Major 主診斷大類子群
               （由 build_extubation_features_dx_pipeline.py 產生）
  --output_dir 輸出目錄

【模型架構參數】
  需與訓練該 checkpoint 時使用的參數一致，預設值取自
  transformer_pre_extubation_risk_trajectory.py 的基本訓練設定
  （d_model=64, nhead=4, num_layers=3, dim_ff=128, dropout=0.2, pe_factor=1.0, pooling=last）。
  若你用其他超參數訓練了模型，請在執行時覆寫對應參數。

【輸出（於 --output_dir 下）】
  - subgroup_predictions.csv   test set 病人層級：y_true / y_prob / 各子群標籤
  - subgroup_metrics.csv       每個子群的完整效能指標（含 bootstrap 95% CI）
  - forest_plot_auroc.png/pdf  各子群 AUROC 森林圖

執行範例（<EXTUBATION_PROJECT_ROOT> 為佔位符，請先設定好環境變數，見 .env.example）：
  python "subgroup_analysis_transformer.py"
    --data_csv "<EXTUBATION_PROJECT_ROOT>/data/outputs/gap4_52to4/extubation_features_imputed_gap4_52to4.csv"
    --model_pt "<EXTUBATION_PROJECT_ROOT>/results/transformer/best_transformer.pt"
    --dx_csv   "<EXTUBATION_PROJECT_ROOT>/data/outputs/extubation_features_final_categories.csv"
    --output_dir "<EXTUBATION_PROJECT_ROOT>/results/subgroup_analysis"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


# 強制標準輸出使用 UTF-8，避免 Windows cp950 主控台無法顯示 ✓ 等符號而報錯
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score,
    brier_score_loss
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# =========================================================================
# 0. 與訓練腳本完全一致的常數 / class（確保 split / scaler / state_dict 相容）
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
    """與訓練腳本完全一致：70/15/15 stratified split by stay_id。"""
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
    """與訓練腳本完全一致的 threshold 選擇邏輯。"""
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
    beta = 1.0 if mode == "f1" else 2.0
    denom = (beta**2 * precision + recall)
    denom = np.where(denom == 0, 1e-12, denom)
    fbeta = (1 + beta**2) * precision * recall / denom
    if len(thr) == 0:
        return 0.5
    return float(thr[np.argmax(fbeta)])


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
    """與 transformer_pre_extubation_risk_trajectory.py 完全一致（確保 state_dict 相容）。"""
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
    """與訓練腳本完全一致（含預計算 missingness mask 偵測）。"""
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
    return np.array(labels).astype(int), np.array(probs).astype(float), np.array(sids).astype(int)


# =========================================================================
# 1. Subgroup 標籤定義
# =========================================================================
def label_age(a):
    if pd.isna(a):
        return None
    if a < 65:
        return "<65"
    if a < 80:
        return "65-79"
    return "≥80"  # ≥80


def label_bmi(b):
    if pd.isna(b):
        return None
    if b < 18.5:
        return "Underweight (<18.5)"
    if b < 25:
        return "Normal (18.5-24.9)"
    if b < 30:
        return "Overweight (25-29.9)"
    return "Obese (≥30)"  # ≥30


def label_charlson(c):
    if pd.isna(c):
        return None
    if c <= 0:
        return "0"
    if c <= 2:
        return "1-2"
    return "≥3"  # ≥3


def label_sex(s):
    if pd.isna(s):
        return None
    return "Male" if int(s) == 1 else "Female"


def label_binary(v, pos_label, neg_label):
    if pd.isna(v):
        return None
    return pos_label if int(v) == 1 else neg_label


# 每個子群變數：(顯示名稱, 標籤產生函式, 來源欄位, level 顯示順序（None=依出現次數排序）)
SUBGROUP_SPECS = [
    ("Sex", "sex", lambda s: label_sex(s), ["Male", "Female"]),
    ("Age group", "age", lambda a: label_age(a), ["<65", "65-79", "≥80"]),
    ("BMI group", "BMI", lambda b: label_bmi(b),
     ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)", "Obese (≥30)"]),
    ("Charlson comorbidity index", "Charlson_Score", lambda c: label_charlson(c), ["0", "1-2", "≥3"]),
    ("Vasopressor use (ever, pre-extubation)", "Vasopressor_use_ever",
     lambda v: label_binary(v, "Vasopressor used", "No vasopressor"), ["Vasopressor used", "No vasopressor"]),
    ("Renal replacement therapy (ever, pre-extubation)", "Hemodialysis_use_ever",
     lambda v: label_binary(v, "RRT used", "No RRT"), ["RRT used", "No RRT"]),
]


def build_stay_level_covariates(df, dx_csv_path=None):
    """
    以 stay_id 彙整靜態子群變數：
      - age / sex / BMI / Charlson_Score：每個 stay 皆相同 → 取第一筆
      - Vasopressor_use / Hemodialysis_use：時變欄位 → 取 pre-extubation 觀察窗內
        是否「曾經」使用（max），作為 ever-used 子群
      - DxGroup_Major：（可選）從 dx_csv 依 stay_id 合併
    """
    agg = df.groupby("stay_id").agg(
        age=("age", "first"),
        sex=("sex", "first"),
        BMI=("BMI", "first"),
        Charlson_Score=("Charlson_Score", "first"),
        Vasopressor_use_ever=("Vasopressor_use", "max"),
        Hemodialysis_use_ever=("Hemodialysis_use", "max"),
    ).reset_index()

    if dx_csv_path is not None and os.path.exists(dx_csv_path):
        dx = pd.read_csv(dx_csv_path, usecols=["stay_id", "DxGroup_Major"])
        agg = agg.merge(dx, on="stay_id", how="left")
        n_missing = agg["DxGroup_Major"].isna().sum()
        if n_missing > 0:
            print(f"[WARN] {n_missing} stay_id 在 dx_csv 中找不到 DxGroup_Major，將標記為 'Others/Unknown'。")
            agg["DxGroup_Major"] = agg["DxGroup_Major"].fillna("Others/Unknown")
    else:
        agg["DxGroup_Major"] = None
        if dx_csv_path is not None:
            print(f"[WARN] 找不到 dx_csv: {dx_csv_path}，跳過 DxGroup_Major 子群。")

    # 產生每個子群變數的顯示標籤欄位
    for disp_name, src_col, fn, _order in SUBGROUP_SPECS:
        agg[disp_name] = agg[src_col].apply(fn)

    if agg["DxGroup_Major"].notna().any():
        agg["Primary diagnosis group"] = agg["DxGroup_Major"]

    return agg


# =========================================================================
# 2. Bootstrap 效能指標
# =========================================================================
def compute_metrics_with_ci(y_true, y_prob, threshold, n_boot=2000, seed=42, min_n=10):
    """
    計算單一（子）群的效能指標與 bootstrap 95% CI。
    若樣本數過小或只有單一類別，AUROC/AUPRC 回傳 NaN 但仍回傳其餘資訊。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    result = {
        "N": n, "n_event": n_pos, "event_rate": (n_pos / n) if n > 0 else np.nan,
        "AUROC": np.nan, "AUROC_lo": np.nan, "AUROC_hi": np.nan,
        "AUPRC": np.nan, "AUPRC_lo": np.nan, "AUPRC_hi": np.nan,
        "Sensitivity": np.nan, "Sensitivity_lo": np.nan, "Sensitivity_hi": np.nan,
        "Specificity": np.nan, "Specificity_lo": np.nan, "Specificity_hi": np.nan,
        "Precision": np.nan, "F1": np.nan, "Brier": np.nan,
        "insufficient_data": False,
    }

    if n < min_n or n_pos == 0 or n_neg == 0:
        result["insufficient_data"] = True
        return result

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result["AUROC"] = roc_auc_score(y_true, y_prob)
    result["AUPRC"] = average_precision_score(y_true, y_prob)
    result["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    result["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    result["Precision"] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    result["F1"] = f1_score(y_true, y_pred, zero_division=0)
    result["Brier"] = brier_score_loss(y_true, y_prob)

    # ── Bootstrap CI（重抽樣，重算 AUROC/AUPRC/Sens/Spec）───────────────
    rng = np.random.RandomState(seed)
    boot_auroc, boot_auprc, boot_sens, boot_spec = [], [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt_b, yp_b = y_true[idx], y_prob[idx]
        if yt_b.sum() == 0 or yt_b.sum() == n:
            continue
        boot_auroc.append(roc_auc_score(yt_b, yp_b))
        boot_auprc.append(average_precision_score(yt_b, yp_b))
        pred_b = (yp_b >= threshold).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt_b, pred_b, labels=[0, 1]).ravel()
        boot_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else np.nan)
        boot_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else np.nan)

    def ci(vals):
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) < 10:
            return np.nan, np.nan
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    result["AUROC_lo"], result["AUROC_hi"] = ci(boot_auroc)
    result["AUPRC_lo"], result["AUPRC_hi"] = ci(boot_auprc)
    result["Sensitivity_lo"], result["Sensitivity_hi"] = ci(boot_sens)
    result["Specificity_lo"], result["Specificity_hi"] = ci(boot_spec)
    return result


# =========================================================================
# 3. Forest plot
# =========================================================================
def plot_forest(rows, overall_auroc, out_png, out_pdf):
    """
    rows: list of dict，每筆含 {"group": str, "level": str, "N", "n_event",
                              "AUROC", "AUROC_lo", "AUROC_hi", "insufficient_data"}
          依 group 分區塊，區塊內依 rows 原始順序畫。
    """
    plot_rows = [r for r in rows if not r["insufficient_data"]]
    if len(plot_rows) == 0:
        print("[WARN] 無足夠資料可畫 forest plot，略過。")
        return

    fig_h = max(4, 0.42 * (len(plot_rows) + len(set(r["group"] for r in plot_rows)) + 2))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y = 0
    yticks, yticklabels = [], []
    current_group = None
    for r in plot_rows:
        if r["group"] != current_group:
            current_group = r["group"]
            y -= 1
            yticks.append(y)
            yticklabels.append(f"$\\bf{{{current_group}}}$".replace(" ", "\\ "))
            ax.text(-0.02, y, "", transform=ax.get_yaxis_transform())  # placeholder row for header
            y -= 1

        auroc, lo, hi = r["AUROC"], r["AUROC_lo"], r["AUROC_hi"]
        label = f"{r['level']}  (n={r['N']}, event={r['n_event']})"
        yticks.append(y)
        yticklabels.append(label)

        color = "#D62728" if r["level"].lower().startswith("overall") else "#1F77B4"
        if not (np.isnan(lo) or np.isnan(hi)):
            ax.errorbar(auroc, y, xerr=[[auroc - lo], [hi - auroc]],
                        fmt="o", color=color, ecolor=color, elinewidth=1.6,
                        capsize=3, markersize=6, zorder=3)
        else:
            ax.plot(auroc, y, "o", color=color, markersize=6, zorder=3)
        y -= 1

    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1.2, label="Chance (AUROC=0.5)")
    if overall_auroc is not None and not np.isnan(overall_auroc):
        ax.axvline(x=overall_auroc, color="#D62728", linestyle="--", linewidth=1.3,
                   label=f"Overall AUROC = {overall_auroc:.3f}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC (95% Bootstrap CI)", fontsize=11)
    ax.set_title("Subgroup Analysis — Transformer Model Discrimination (Test Set)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_png}")
    print(f"✓ Saved: {out_pdf}")


# =========================================================================
# 4. CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Subgroup analysis for the trained Transformer model (inference only).")
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--model_pt", type=str, required=True)
    p.add_argument("--dx_csv", type=str, default=None,
                   help="extubation_features_final_categories.csv（含 DxGroup_Major，可選）")
    p.add_argument("--output_dir", type=str, default="results/subgroup_analysis")
    p.add_argument("--seed", type=int, default=42, help="需與訓練時的 --seed 一致，才能重建相同 split。")

    # 模型架構（需與訓練該 checkpoint 時一致；預設值取自基本訓練設定）
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dim_ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--pe_factor", type=float, default=1.0)
    p.add_argument("--pooling", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--use_causal_mask", type=int, default=0, choices=[0, 1])

    # Threshold
    p.add_argument("--threshold", type=float, default=None,
                   help="若指定，直接使用此 threshold（略過 val set 重新尋找）。")
    p.add_argument("--threshold_mode", type=str, default="youden",
                   choices=["youden", "f1", "f2", "sens_spec"])

    # Bootstrap
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--min_n", type=int, default=10,
                   help="子群樣本數（或單一 outcome 類別）低於此值則標記為資料不足，不計算 CI。")

    return p.parse_args()


# =========================================================================
# 5. Main
# =========================================================================
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── 1. 載入資料 ─────────────────────────────────────────────────────
    print("Loading data ...")
    df = pd.read_csv(args.data_csv)
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df = df[df["time_bin"].isin(SEQ_TIME_BINS)].copy()
    if df.empty:
        raise ValueError("After filtering by SEQ_TIME_BINS, dataframe is empty. Check time_bin values.")

    # ── 2. 重建 split + scaler（與訓練腳本完全一致，不訓練）──────────────
    print(f"Rebuilding train/val/test split (seed={args.seed}) ...")
    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.7, seed=args.seed)
    train_df = df[df["stay_id"].isin(train_ids)].copy()
    val_df = df[df["stay_id"].isin(val_ids)].copy()
    test_df = df[df["stay_id"].isin(test_ids)].copy()

    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"]) if c not in BINARY_LIKE]
    scaler = StandardScaler()
    scaler.fit(train_df[scale_cols])

    ds_val = ExtubationSeqDataset(val_df, val_ids, scaler, scale_cols, SEQ_TIME_BINS)
    ds_test = ExtubationSeqDataset(test_df, test_ids, scaler, scale_cols, SEQ_TIME_BINS)
    val_loader = DataLoader(ds_val, batch_size=256, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=256, shuffle=False)
    print(f"Val stays: {len(ds_val)} | Test stays: {len(ds_test)}")

    # ── 3. 載入模型 ─────────────────────────────────────────────────────
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

    # ── 4. 決定 threshold ──────────────────────────────────────────────
    if args.threshold is not None:
        best_thr = float(args.threshold)
        print(f"[Threshold] 使用指定值：{best_thr:.4f}")
    else:
        yv, pv, _ = get_probs(model, val_loader, device)
        best_thr = find_best_threshold(yv, pv, mode=args.threshold_mode)
        print(f"[Threshold] Val set 重新尋找（mode={args.threshold_mode}）：{best_thr:.4f}")

    # ── 5. Test set 推論 ────────────────────────────────────────────────
    print("Running inference on test set ...")
    yt, pt, sids_t = get_probs(model, test_loader, device)
    pred_df = pd.DataFrame({"stay_id": sids_t, "y_true": yt, "y_prob": pt})

    # ── 6. 合併子群標籤 ─────────────────────────────────────────────────
    print("Building subgroup covariates ...")
    stay_cov = build_stay_level_covariates(df, dx_csv_path=args.dx_csv)
    pred_df = pred_df.merge(stay_cov, on="stay_id", how="left")

    pred_csv = os.path.join(args.output_dir, "subgroup_predictions.csv")
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {pred_csv}")

    # ── 7. 整體（Overall）指標 ─────────────────────────────────────────
    print(f"Computing metrics (n_boot={args.n_boot}) ...")
    rows = []
    overall = compute_metrics_with_ci(pred_df["y_true"], pred_df["y_prob"], best_thr,
                                       n_boot=args.n_boot, seed=args.seed, min_n=args.min_n)
    overall.update({"group": "Overall", "level": "Overall (All test patients)"})
    rows.append(overall)
    print(f"  Overall: AUROC={overall['AUROC']:.4f} "
          f"({overall['AUROC_lo']:.4f}-{overall['AUROC_hi']:.4f})  N={overall['N']}")

    # ── 8. 各子群變數 × 各 level 指標 ──────────────────────────────────
    for disp_name, _src_col, _fn, level_order in SUBGROUP_SPECS:
        if disp_name not in pred_df.columns:
            continue
        levels_present = [lv for lv in pred_df[disp_name].dropna().unique()]
        if level_order is not None:
            ordered_levels = [lv for lv in level_order if lv in levels_present] + \
                              [lv for lv in levels_present if lv not in level_order]
        else:
            ordered_levels = sorted(levels_present)

        for lv in ordered_levels:
            sub = pred_df[pred_df[disp_name] == lv]
            m = compute_metrics_with_ci(sub["y_true"], sub["y_prob"], best_thr,
                                         n_boot=args.n_boot, seed=args.seed, min_n=args.min_n)
            m.update({"group": disp_name, "level": str(lv)})
            rows.append(m)
            flag = " [insufficient data]" if m["insufficient_data"] else ""
            auroc_str = f"{m['AUROC']:.4f}" if not np.isnan(m["AUROC"]) else "NA"
            print(f"  {disp_name} = {lv}: AUROC={auroc_str}  N={m['N']}  event={m['n_event']}{flag}")

    # DxGroup_Major（若有提供 dx_csv）
    if "Primary diagnosis group" in pred_df.columns:
        levels_present = sorted(pred_df["Primary diagnosis group"].dropna().unique())
        for lv in levels_present:
            sub = pred_df[pred_df["Primary diagnosis group"] == lv]
            m = compute_metrics_with_ci(sub["y_true"], sub["y_prob"], best_thr,
                                         n_boot=args.n_boot, seed=args.seed, min_n=args.min_n)
            m.update({"group": "Primary diagnosis group (DxGroup_Major)", "level": str(lv)})
            rows.append(m)
            flag = " [insufficient data]" if m["insufficient_data"] else ""
            auroc_str = f"{m['AUROC']:.4f}" if not np.isnan(m["AUROC"]) else "NA"
            print(f"  DxGroup_Major = {lv}: AUROC={auroc_str}  N={m['N']}  event={m['n_event']}{flag}")

    # ── 9. 儲存表格 ─────────────────────────────────────────────────────
    res_df = pd.DataFrame(rows)
    col_order = ["group", "level", "N", "n_event", "event_rate",
                 "AUROC", "AUROC_lo", "AUROC_hi",
                 "AUPRC", "AUPRC_lo", "AUPRC_hi",
                 "Sensitivity", "Sensitivity_lo", "Sensitivity_hi",
                 "Specificity", "Specificity_lo", "Specificity_hi",
                 "Precision", "F1", "Brier", "insufficient_data"]
    res_df = res_df[[c for c in col_order if c in res_df.columns]]
    metrics_csv = os.path.join(args.output_dir, "subgroup_metrics.csv")
    res_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"✓ Saved: {metrics_csv}")

    # ── 10. Forest plot ─────────────────────────────────────────────────
    plot_forest(
        rows, overall_auroc=overall["AUROC"],
        out_png=os.path.join(args.output_dir, "forest_plot_auroc.png"),
        out_pdf=os.path.join(args.output_dir, "forest_plot_auroc.pdf"),
    )

    print("\n完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
