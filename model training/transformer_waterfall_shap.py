#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transformer_waterfall_shap.py

Committee comment 8: individual-patient interpretability (waterfall plots)
for the PRIMARY Transformer EF-prediction model.

── Why KernelExplainer, not TreeSHAP ────────────────────────────────────────
TreeSHAP (used elsewhere in this project for the XGBoost early-phenotype
classifier) only applies to tree ensembles. The Transformer is the thesis's
main EF-risk model, so per-patient explanations here use shap.KernelExplainer,
a model-agnostic method that treats the model as a black-box function. This
is only tractable because we explain a handful (4) of individual patients,
not the full test set.

── What is explained ────────────────────────────────────────────────────────
The "explainable" feature space is the 26 clinical dynamic variables across
all 12 time bins (312 features) plus the 4 static features (age, sex, BMI,
Charlson) = 316 features, expressed in raw clinical units so the waterfall
plot is clinically readable. The 26 missingness-mask channels per bin are a
technical/nuisance input (not a clinical feature) and are held fixed at each
patient's true observed/missing pattern during perturbation, exactly like
step_present (bin presence) is held fixed. No retraining is performed; the
frozen best_transformer.pt checkpoint is reused.

── Case selection ───────────────────────────────────────────────────────────
Four representative cases (TP, TN, FP, FN) are selected from the primary
Transformer's own test-set confusion matrix, using the same operating
threshold reported in the thesis (0.456; reproduced here from
results/transformer/test_predictions.csv rather than hard-coded, see
`recover_operating_threshold`). Within each quadrant, the patient whose
predicted probability is closest to that quadrant's median is chosen, so
each case is representative rather than an extreme outlier.

Outputs (in OUTPUT_DIR):
  - selected_cases.csv                        (the 4 chosen patients + info)
  - waterfall_{quadrant}_stay{stay_id}.png     (one plot per case)
  - figure16_waterfall_grid.png                (2x2 composite, Figure 16)
  - shap_values_{quadrant}_stay{stay_id}.csv   (full per-feature SHAP table)
"""

import os
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_pre_extubation_risk_trajectory import (
    ExtubationTransformer, STATIC_COLS, DYNAMIC_COLS, TARGET,
    SEQ_TIME_BINS, split_by_stay_id
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import shap
import matplotlib.pyplot as plt
from PIL import Image

# =========================
# Paths / constants
# =========================
# 以下路徑由環境變數 EXTUBATION_PROJECT_ROOT / MIMIC_DATA_DIR 提供，請參考 repo 根目錄的 .env.example 設定
BASE       = rf"{EXTUBATION_ROOT}"
DATA_CSV   = rf"{BASE}\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
MODEL_PATH = rf"{BASE}\results\transformer\best_transformer.pt"
PRED_CSV   = rf"{BASE}\results\transformer\test_predictions.csv"
OUTPUT_DIR = rf"{BASE}\results\transformer\waterfall_shap"
os.makedirs(OUTPUT_DIR, exist_ok=True)

D_MODEL, NHEAD, NUM_LAYERS, DIM_FF = 64, 4, 3, 128
DROPOUT, PE_FACTOR = 0.2, 1.0
SEED = 42

MASK_COLS = [f"mask_{c}" for c in DYNAMIC_COLS]
NO_SCALE_DYNAMIC = {"Vasopressor_use", "Hemodialysis_use"}   # binary flags, kept raw
NO_SCALE_STATIC  = {"sex"}                                     # binary flag, kept raw

FEATURE_DISPLAY_NAME = {
    "age": "Age", "sex": "Sex (Male)", "BMI": "BMI", "Charlson_Score": "Charlson",
    "heart_rate": "HR", "resp_rate": "RR", "spo2": "SpO2", "mbp": "MBP",
    "temperature": "Temp", "GCS": "GCS", "FiO2": "FiO2", "MAP": "Paw_mean",
    "PEEP": "PEEP", "TV_per_kg": "TV/kg", "MV_day": "MV Day",
    "pH": "pH", "PaO2": "PaO2", "PaCO2": "PaCO2", "BE": "BE", "OI": "OI",
    "Cr": "Cr", "WBC": "WBC", "Hb": "Hb", "PLT": "PLT",
    "AnionGap": "AG", "Lactate": "Lactate", "Glucose": "Glucose",
    "io_balance": "IO Balance", "Vasopressor_use": "Vasopressor",
    "Hemodialysis_use": "Hemodialysis",
}

QUADRANT_TITLES = {
    "TP": "True Positive (correctly predicted EF)",
    "TN": "True Negative (correctly predicted no EF)",
    "FP": "False Positive (predicted EF, actually no EF)",
    "FN": "False Negative (predicted no EF, actually EF)",
}


# =========================
# Helpers
# =========================
def recover_operating_threshold(pred_csv):
    """
    Recover the Youden threshold selected on the validation set (reported in
    the thesis as 0.456) by scanning candidate thresholds against the
    Accuracy/Sensitivity/Specificity already reported in
    results/transformer/performance_metrics.csv, so the number is not
    hard-coded independently of the saved artifacts.
    """
    perf_path = os.path.join(os.path.dirname(pred_csv), "performance_metrics.csv")
    perf = pd.read_csv(perf_path).set_index(perf_path and "Metric" if False else "Metric")
    # performance_metrics.csv has no header key "Metric" in some runs; handle both shapes
    return perf


def find_threshold_matching_metrics(df_pred, target_sens, target_spec, tol=1e-3):
    y = df_pred["y_true"].values
    p = df_pred["y_prob"].values
    for thr in np.linspace(0.01, 0.99, 981):
        pred = (p >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        if abs(sens - target_sens) < tol and abs(spec - target_spec) < tol:
            return float(thr)
    raise RuntimeError("Could not recover operating threshold from performance_metrics.csv")


def build_patient_raw(df_indexed, sid, seq_time_bins):
    """
    Mirror ExtubationSeqDataset._build_one(), but return UNSCALED clinical
    values (for SHAP display) plus the mask / step_present arrays.
    Returns dyn_raw (12,26), mask (12,26), step_present (12,), stat_raw (4,)
    """
    d = df_indexed[df_indexed["stay_id"] == sid]
    dyn_raw, mask, step_present = [], [], []
    for tb in seq_time_bins:
        row = d[d["time_bin"] == tb]
        if row.empty:
            dyn_raw.append(np.zeros(len(DYNAMIC_COLS), dtype=np.float32))
            mask.append(np.zeros(len(DYNAMIC_COLS), dtype=np.float32))
            step_present.append(0.0)
        else:
            dyn_raw.append(row[DYNAMIC_COLS].iloc[0].values.astype(np.float32))
            mask.append(row[MASK_COLS].iloc[0].values.astype(np.float32))
            step_present.append(float(row["bin_has_data"].iloc[0]))
    stat_raw = d[STATIC_COLS].iloc[0].fillna(0).values.astype(np.float32)
    return (np.stack(dyn_raw), np.stack(mask),
            np.array(step_present, dtype=np.float32), stat_raw)


def make_scaling_fns(scaler, scale_cols):
    """
    Returns callables that map RAW dynamic (12,26) / static (4,) arrays to
    the SAME standardized space the Transformer was trained on, using the
    already-fitted scaler (scale_cols order == scaler.mean_/scale_ order).
    """
    mean_map = dict(zip(scale_cols, scaler.mean_))
    scale_map = dict(zip(scale_cols, scaler.scale_))

    dyn_mean = np.array([mean_map.get(c, 0.0) for c in DYNAMIC_COLS])
    dyn_scale = np.array([scale_map.get(c, 1.0) for c in DYNAMIC_COLS])
    dyn_scaled_flag = np.array([c not in NO_SCALE_DYNAMIC and c in scale_cols
                                 for c in DYNAMIC_COLS])

    stat_mean = np.array([mean_map.get(c, 0.0) for c in STATIC_COLS])
    stat_scale = np.array([scale_map.get(c, 1.0) for c in STATIC_COLS])
    stat_scaled_flag = np.array([c not in NO_SCALE_STATIC and c in scale_cols
                                  for c in STATIC_COLS])

    def scale_dyn(raw_12x26):
        out = raw_12x26.copy()
        out[:, dyn_scaled_flag] = (out[:, dyn_scaled_flag] - dyn_mean[dyn_scaled_flag]) / dyn_scale[dyn_scaled_flag]
        return out

    def scale_stat(raw_4):
        out = raw_4.copy()
        out[stat_scaled_flag] = (out[stat_scaled_flag] - stat_mean[stat_scaled_flag]) / stat_scale[stat_scaled_flag]
        return out

    return scale_dyn, scale_stat


def make_predict_fn(model, device, scale_dyn, scale_stat, mask_fixed, step_present_fixed, n_dyn_flat):
    """
    Build the black-box function KernelExplainer perturbs.
    X: (n_samples, 316) raw values = [dyn_flat(312, bin-major) | static(4)]
    Returns: (n_samples,) predicted probability of Extubation_failure.
    """
    T, C = mask_fixed.shape  # (12, 26)

    def predict_fn(X):
        X = np.atleast_2d(X)
        n = X.shape[0]
        dyn_flat = X[:, :n_dyn_flat].reshape(n, T, C).astype(np.float32)
        stat_raw = X[:, n_dyn_flat:].astype(np.float32)

        dyn_scaled = np.stack([scale_dyn(dyn_flat[i]) for i in range(n)], axis=0)  # (n,T,C)
        stat_scaled = np.stack([scale_stat(stat_raw[i]) for i in range(n)], axis=0)  # (n,4)

        mask_rep = np.repeat(mask_fixed[None, :, :], n, axis=0)          # (n,T,C)
        step_rep = np.repeat(step_present_fixed[None, :], n, axis=0)     # (n,T)

        x_dyn = np.concatenate([dyn_scaled, mask_rep], axis=-1)          # (n,T,52)

        with torch.no_grad():
            x_dyn_t = torch.tensor(x_dyn, dtype=torch.float32, device=device)
            x_stat_t = torch.tensor(stat_scaled, dtype=torch.float32, device=device)
            step_t = torch.tensor(step_rep, dtype=torch.float32, device=device)
            logit = model(x_dyn_t, x_stat_t, step_present=step_t, return_embedding=False)
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
        return prob

    return predict_fn


def flat_feature_names():
    names = []
    for tb in SEQ_TIME_BINS:
        for c in DYNAMIC_COLS:
            disp = FEATURE_DISPLAY_NAME.get(c, c)
            names.append(f"{disp} (t={tb}h)")
    for c in STATIC_COLS:
        names.append(FEATURE_DISPLAY_NAME.get(c, c))
    return names


# =========================
# Main
# =========================
def main():
    device = torch.device("cpu")   # see sensitivity_leave_first_bin_out.py for why CPU is forced
    print(f"[Device] {device}")

    df = pd.read_csv(DATA_CSV)
    if "sex" in df.columns and df["sex"].dtype == object:
        df["sex"] = (df["sex"].astype(str).str.lower() == "male").astype(int)

    train_ids, val_ids, test_ids = split_by_stay_id(df, train_ratio=0.70, seed=SEED)
    print(f"[Split] train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    scale_cols = [c for c in (DYNAMIC_COLS + ["age", "BMI", "Charlson_Score"])
                  if c in df.columns and c not in {"sex", "Vasopressor_use", "Hemodialysis_use"}]
    scaler = StandardScaler().fit(df[df["stay_id"].isin(train_ids)][scale_cols])
    scale_dyn, scale_stat = make_scaling_fns(scaler, scale_cols)

    model = ExtubationTransformer(
        dyn_dim=52, stat_dim=len(STATIC_COLS),
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF,
        dropout=DROPOUT, pe_factor=PE_FACTOR,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    print(f"✓ Model loaded: {MODEL_PATH}")

    # ── Recover operating threshold from saved metrics (thesis reports 0.456) ──
    df_pred = pd.read_csv(PRED_CSV)
    perf = pd.read_csv(os.path.join(BASE, "results", "transformer", "performance_metrics.csv"))
    perf_d = dict(zip(perf.iloc[:, 0], perf.iloc[:, 1]))
    threshold = find_threshold_matching_metrics(
        df_pred, target_sens=float(perf_d["Sensitivity"]), target_spec=float(perf_d["Specificity"]))
    print(f"[Threshold] recovered operating threshold = {threshold:.4f}")

    df_pred["pred"] = (df_pred["y_prob"] >= threshold).astype(int)
    df_pred["quadrant"] = np.select(
        [(df_pred["y_true"] == 1) & (df_pred["pred"] == 1),
         (df_pred["y_true"] == 0) & (df_pred["pred"] == 0),
         (df_pred["y_true"] == 0) & (df_pred["pred"] == 1),
         (df_pred["y_true"] == 1) & (df_pred["pred"] == 0)],
        ["TP", "TN", "FP", "FN"])

    # ── Select one representative (median-probability) patient per quadrant ──
    selected = []
    for q in ["TP", "TN", "FP", "FN"]:
        sub = df_pred[df_pred["quadrant"] == q].copy()
        med = sub["y_prob"].median()
        sub["dist_to_median"] = (sub["y_prob"] - med).abs()
        sub = sub.sort_values("dist_to_median")
        chosen = sub.iloc[0]
        selected.append(chosen)
        print(f"  {q}: n={len(sub)}  chosen stay_id={int(chosen['stay_id'])}  "
              f"y_true={int(chosen['y_true'])}  y_prob={chosen['y_prob']:.4f}")

    df_sel = pd.DataFrame(selected)[["stay_id", "y_true", "y_prob", "quadrant"]]
    df_sel.to_csv(os.path.join(OUTPUT_DIR, "selected_cases.csv"), index=False, encoding="utf-8-sig")

    # ── Background: k-means summary of TRAIN patients' raw feature vectors ──
    print("\n[Background] Building raw feature vectors for a training-set sample...")
    rng = np.random.default_rng(SEED)
    bg_ids = rng.choice(train_ids, size=min(300, len(train_ids)), replace=False)
    bg_rows = []
    for sid in bg_ids:
        dyn_raw, _, _, stat_raw = build_patient_raw(df, int(sid), SEQ_TIME_BINS)
        bg_rows.append(np.concatenate([dyn_raw.reshape(-1), stat_raw]))
    bg_matrix = np.stack(bg_rows, axis=0)
    n_dyn_flat = len(SEQ_TIME_BINS) * len(DYNAMIC_COLS)
    print(f"  Background sample: {bg_matrix.shape[0]} patients x {bg_matrix.shape[1]} features")

    background_summary = shap.kmeans(bg_matrix, 25)
    feature_names = flat_feature_names()

    # ── Explain each of the 4 selected patients ──
    n_display_panels = []
    for _, row in df_sel.iterrows():
        sid = int(row["stay_id"])
        quadrant = row["quadrant"]
        dyn_raw, mask_fixed, step_fixed, stat_raw = build_patient_raw(df, sid, SEQ_TIME_BINS)
        x_instance = np.concatenate([dyn_raw.reshape(-1), stat_raw]).reshape(1, -1)

        predict_fn = make_predict_fn(model, device, scale_dyn, scale_stat,
                                      mask_fixed, step_fixed, n_dyn_flat)

        # Sanity check: predict_fn on this patient's true input should match test_predictions.csv
        prob_check = predict_fn(x_instance)[0]
        print(f"\n[{quadrant}] stay_id={sid}  recorded y_prob={row['y_prob']:.4f}  "
              f"recomputed y_prob={prob_check:.4f}")

        explainer = shap.KernelExplainer(predict_fn, background_summary, link="identity")
        shap_vals = explainer.shap_values(x_instance, nsamples=500, silent=True)
        shap_vals = np.array(shap_vals).reshape(-1)
        base_value = float(np.array(explainer.expected_value).reshape(-1)[0])

        # Save full per-feature SHAP table
        df_shap = pd.DataFrame({
            "feature": feature_names,
            "value": x_instance.reshape(-1),
            "shap_value": shap_vals,
        }).sort_values("shap_value", key=np.abs, ascending=False)
        df_shap.to_csv(os.path.join(OUTPUT_DIR, f"shap_values_{quadrant}_stay{sid}.csv"),
                        index=False, encoding="utf-8-sig")

        expl = shap.Explanation(
            values=shap_vals, base_values=base_value,
            data=x_instance.reshape(-1), feature_names=feature_names)

        plt.figure()
        shap.plots.waterfall(expl, max_display=12, show=False)
        plt.title(f"{QUADRANT_TITLES[quadrant]}\n"
                  f"stay_id={sid}  |  true label={'EF' if row['y_true'] == 1 else 'No EF'}  "
                  f"|  predicted P(EF)={row['y_prob']:.3f}  (threshold={threshold:.3f})",
                  fontsize=10, fontweight="bold")
        panel_path = os.path.join(OUTPUT_DIR, f"waterfall_{quadrant}_stay{sid}.png")
        plt.tight_layout()
        plt.savefig(panel_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved: {panel_path}")
        n_display_panels.append(panel_path)

    # ── Combine into one 2x2 grid: Figure 16 ──
    imgs = [Image.open(p) for p in n_display_panels]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    grid = Image.new("RGB", (w * 2, h * 2), "white")
    for i, im in enumerate(imgs):
        im_resized = im.resize((w, h))
        x_off = (i % 2) * w
        y_off = (i // 2) * h
        grid.paste(im_resized, (x_off, y_off))
    grid_path = os.path.join(OUTPUT_DIR, "figure16_waterfall_grid.png")
    grid.save(grid_path)
    print(f"\n✓ Figure 16 (2x2 composite) saved: {grid_path}")

    print(f"\n✅ Done. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Run with the conda env properly ACTIVATED (not invoked by bare path —
    # see sensitivity_leave_first_bin_out.py note on the sklearn/MKL DLL crash):
    #
    #   conda activate extubation_env
    #   python "model training/transformer_waterfall_shap.py"
    main()
