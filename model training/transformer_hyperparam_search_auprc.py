#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transformer_hyperparam_search_auprc.py

使用 Optuna（TPE Bayesian Optimization）對 transformer_pre_extubation_risk_trajectory.py
進行超參數搜尋。

【與 transformer_hyperparam_search.py 的差異】
  - 排序指標：best_val_AUPRC（而非 best_val_AUROC）
  - 輸出目錄：results/transformer_hparam_search_auprc（不覆蓋舊結果）
  - Optuna study：transformer_hparam_auprc（獨立 DB）

【使用 AUPRC 作為主要指標的臨床理由】
  拔管失敗（Label=1）在 ICU 族群中屬於少數事件（約 40%），
  AUPRC（Precision-Recall AUC）對不平衡資料集更敏感，
  能更直接反映模型在辨識失敗病患上的實際能力。

【固定設計決策（不參與搜尋）】
  use_time_weights = 1  Prefix Cumulative Training：動態風險軌跡（論文核心設計）
  use_causal_mask  = 1  Causal Self-Attention：訓練與 prefix inference 行為一致

【搜尋空間】
  ┌─ 模型架構（耦合：nhead 必須整除 d_model）
  │    (d_model, nhead, dim_ff): 8 種 categorical 組合
  │    num_layers: int 2–4
  ├─ 正則化
  │    dropout:      float 0.0–0.5（線性連續）
  │    weight_decay: float 1e-5–1e-2（對數連續）
  │    train_noise:  float 0.0–0.1（線性連續）
  ├─ 學習率排程
  │    lr:              float 1e-5–5e-4（對數連續）
  │    lr_warmup_steps: int   2–10
  │    lr_decay:        float 0.90–1.0（線性連續）
  ├─ Positional Encoding
  │    pe_factor: float 0.001–1.0（對數連續）
  └─ 架構選項
       pooling: categorical [last, mean]

【執行方式】
  # 預設 100 次試驗，每次最多 50 epochs
  python transformer_hyperparam_search_auprc.py

  # 自訂試驗次數與每次 epoch 上限
  python transformer_hyperparam_search_auprc.py --max_trials 150 --epochs_per_trial 30

  # 中斷後繼續（自動從 optuna_study.db 恢復）
  python transformer_hyperparam_search_auprc.py --max_trials 150

【輸出（儲存於 BASE_OUTPUT_DIR，不影響 transformer_hparam_search 舊結果）】
  - hparam_search_results.csv   所有試驗結果
  - best_config.json            最佳超參數設定（依 best_val_AUPRC 排序）
  - optuna_study.db             Optuna study（供 resume 與視覺化）
  - trial_<N>/                  每次試驗的詳細輸出
"""

import subprocess
import json
import time
import argparse
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
EXTUBATION_ROOT = os.environ.get("EXTUBATION_PROJECT_ROOT")
if not EXTUBATION_ROOT:
    raise RuntimeError(
        "Environment variable EXTUBATION_PROJECT_ROOT is not set. "
        "Copy .env.example to .env (or set it directly) and point it to your "
        "local extubation_failure_prediction project root."
    )


optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 路徑設定
# =====================================================================
TRANSFORMER_SCRIPT = (
    rf"{EXTUBATION_ROOT}"
    r"\model training\transformer_pre_extubation_risk_trajectory.py"
)
DATA_CSV = (
    rf"{EXTUBATION_ROOT}"
    r"\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
)
# ⚠️  獨立輸出目錄，不覆蓋 transformer_hparam_search 的舊結果
BASE_OUTPUT_DIR = Path(
    rf"{EXTUBATION_ROOT}"
    r"\results\transformer_hparam_search_auprc"
)

# =====================================================================
# 架構組合（耦合參數：nhead 必須整除 d_model）
# =====================================================================
MODEL_ARCH_CONFIGS = [
    {"d_model": 32,  "nhead": 4, "dim_ff": 64},
    {"d_model": 32,  "nhead": 4, "dim_ff": 128},
    {"d_model": 64,  "nhead": 4, "dim_ff": 128},
    {"d_model": 64,  "nhead": 4, "dim_ff": 256},
    {"d_model": 128, "nhead": 4, "dim_ff": 128},
    {"d_model": 128, "nhead": 4, "dim_ff": 256},
    {"d_model": 128, "nhead": 8, "dim_ff": 256},
    {"d_model": 128, "nhead": 8, "dim_ff": 512},
]

# =====================================================================
# 固定參數（不參與搜尋）
# =====================================================================
FIXED_PARAMS = {
    "threshold_mode":   "youden",
    "seed":             42,
    "batch_size":       64,
    "patience":         10,
    "save_predictions": 0,
    "use_time_weights": 1,
    "use_causal_mask":  1,
}

# 主要排序指標：val set AUPRC
PRIMARY_METRIC = "best_val_AUPRC"

# =====================================================================
# 超參數建議（Optuna Trial）
# =====================================================================

def suggest_config(trial: optuna.Trial) -> dict:
    """使用 Optuna trial 建議一組超參數設定。"""
    arch_idx = trial.suggest_categorical(
        "arch_idx", list(range(len(MODEL_ARCH_CONFIGS)))
    )
    arch = MODEL_ARCH_CONFIGS[arch_idx]

    return {
        "d_model":         arch["d_model"],
        "nhead":           arch["nhead"],
        "dim_ff":          arch["dim_ff"],
        "num_layers":      trial.suggest_int("num_layers", 2, 4),
        "dropout":         trial.suggest_float("dropout",      0.0,  0.5),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "train_noise":     trial.suggest_float("train_noise",  0.0,  0.1),
        "lr":              trial.suggest_float("lr",            1e-5, 5e-4, log=True),
        "lr_warmup_steps": trial.suggest_int("lr_warmup_steps", 2,   10),
        "lr_decay":        trial.suggest_float("lr_decay",      0.90, 1.0),
        "pe_factor":       trial.suggest_float("pe_factor", 0.001, 1.0, log=True),
        "pooling":         trial.suggest_categorical("pooling", ["last", "mean"]),
    }

# =====================================================================
# 工具函數
# =====================================================================

def config_to_cmd(config: dict, output_dir: Path, epochs: int) -> list:
    cmd = [
        sys.executable, TRANSFORMER_SCRIPT,
        "--data_csv",   DATA_CSV,
        "--output_dir", str(output_dir),
        "--epochs",     str(epochs),
    ]
    for key, val in config.items():
        cmd += [f"--{key}", str(val)]
    for key, val in FIXED_PARAMS.items():
        cmd += [f"--{key}", str(val)]
    return cmd


def read_metrics(output_dir: Path) -> dict:
    csv_path = output_dir / "performance_metrics.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "Metric" not in df.columns and len(df) >= 1:
            raw = df.iloc[0].to_dict()
            rename = {
                "F1 score":    "F1_score",
                "F1_score":    "F1_score",
                "brier_score": "Brier_score",
                "Brier_score": "Brier_score",
            }
            return {rename.get(k, k): float(v) for k, v in raw.items()
                    if v is not None and str(v).strip() != ""}
        return dict(zip(df["Metric"], df["Value"]))
    except Exception as e:
        print(f"    ⚠️  讀取 metrics 失敗: {e}")
        return {}


def format_config_summary(config: dict) -> str:
    arch_str  = (f"d{config['d_model']}_h{config['nhead']}"
                 f"_ff{config['dim_ff']}_l{config['num_layers']}")
    train_str = (f"lr{config['lr']:.1e}_wd{config['weight_decay']:.1e}"
                 f"_do{config['dropout']:.2f}_noise{config['train_noise']:.3f}")
    sched_str = f"warm{config['lr_warmup_steps']}_decay{config['lr_decay']:.3f}"
    arch_opt  = (f"pe{config['pe_factor']:.3f}_pool{config['pooling']}"
                 f"_tw{FIXED_PARAMS['use_time_weights']}"
                 f"_cm{FIXED_PARAMS['use_causal_mask']}")
    return f"{arch_str} | {train_str} | {sched_str} | {arch_opt}"


def append_result(results_csv: Path, row: dict) -> None:
    df_new = pd.DataFrame([row])
    write_header = not results_csv.exists()
    df_new.to_csv(results_csv, mode="a", header=write_header, index=False)


def to_python_type(val):
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def print_top_n(results_csv: Path, n: int = 10) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"].dropna(subset=["best_val_AUPRC"]).copy()
    df = df.sort_values("best_val_AUPRC", ascending=False).drop_duplicates(subset=["trial"])
    if df.empty:
        print("  （尚無成功的試驗）")
        return
    df = df.sort_values(PRIMARY_METRIC, ascending=False).head(n)
    display_cols = [
        "trial", "best_val_AUPRC", "best_val_AUROC",
        "AUPRC", "AUROC", "Sensitivity", "Specificity",
        "F1_score", "Brier_score",
        "d_model", "nhead", "dim_ff", "num_layers",
        "lr", "dropout", "pe_factor", "pooling",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))


def save_best_config(results_csv: Path, output_dir: Path) -> None:
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"].dropna(subset=["best_val_AUPRC"]).copy()
    df = df.sort_values(PRIMARY_METRIC, ascending=False).drop_duplicates(subset=["trial"])
    if df.empty:
        return
    best_row = df.iloc[0]

    config_keys = [
        "d_model", "nhead", "dim_ff", "num_layers",
        "dropout", "weight_decay", "train_noise",
        "lr", "lr_warmup_steps", "lr_decay",
        "pe_factor", "pooling",
    ]
    best_config = {
        k: to_python_type(best_row[k]) for k in config_keys if k in best_row.index
    }
    best_config["best_val_AUPRC"]   = float(best_row["best_val_AUPRC"])
    best_config["best_val_AUROC"]   = float(best_row["best_val_AUROC"])
    best_config["AUPRC"]            = float(best_row["AUPRC"])
    best_config["AUROC"]            = float(best_row["AUROC"])
    best_config["trial"]            = int(best_row["trial"])
    best_config["use_time_weights"] = FIXED_PARAMS["use_time_weights"]
    best_config["use_causal_mask"]  = FIXED_PARAMS["use_causal_mask"]

    out_path = output_dir / "best_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 最佳設定已儲存：{out_path}")
    print(f"   最佳 val_AUPRC ：{best_config['best_val_AUPRC']:.4f}  "
          f"val_AUROC：{best_config['best_val_AUROC']:.4f}  "
          f"test_AUPRC：{best_config['AUPRC']:.4f}  "
          f"test_AUROC：{best_config['AUROC']:.4f}  "
          f"(Trial #{best_config['trial']})")
    print(f"   設定摘要：{format_config_summary(best_config)}")

# =====================================================================
# Optuna Objective
# =====================================================================

def make_objective(args, results_csv: Path):
    def objective(trial: optuna.Trial) -> float:
        trial_num = trial.number + 1
        trial_dir = BASE_OUTPUT_DIR / f"trial_{trial_num:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = suggest_config(trial)

        print(f"\n{'─' * 70}")
        print(f"🔬 Trial {trial_num}  (Optuna #{trial.number})")
        print(f"   設定：{format_config_summary(config)}")
        print(f"   輸出：{trial_dir}")

        cmd = config_to_cmd(config, trial_dir, args.epochs_per_trial)
        (trial_dir / "run_command.txt").write_text(
            " \\\n  ".join(cmd), encoding="utf-8"
        )

        t0     = time.time()
        status = "success"

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"]       = "1"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_per_trial,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            elapsed = time.time() - t0

            (trial_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (trial_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

            if result.returncode != 0:
                status = "failed"
                print(f"   ❌ 執行失敗（returncode={result.returncode}）")
                for line in result.stderr.strip().splitlines()[-10:]:
                    print(f"      {line}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status  = "timeout"
            print(f"   ⏱️  逾時（>{args.timeout_per_trial}s）")

        except Exception as e:
            elapsed = time.time() - t0
            status  = "error"
            print(f"   ⚠️  例外：{e}")

        metrics    = read_metrics(trial_dir) if status == "success" else {}
        val_auprc  = metrics.get("best_val_AUPRC", float("nan"))
        val_auroc  = metrics.get("best_val_AUROC", float("nan"))

        row = {
            "trial":          trial_num,
            "optuna_trial":   trial.number,
            "status":         status,
            "elapsed_sec":    round(elapsed, 1),
            "output_dir":     str(trial_dir),
            # val 指標（選擇依據）
            "best_val_AUPRC": val_auprc,
            "best_val_AUROC": val_auroc,
            # test 指標（最終報告，不用於排名）
            "AUROC":          metrics.get("AUROC",       float("nan")),
            "AUPRC":          metrics.get("AUPRC",       float("nan")),
            "Accuracy":       metrics.get("Accuracy",    float("nan")),
            "Sensitivity":    metrics.get("Sensitivity", float("nan")),
            "Specificity":    metrics.get("Specificity", float("nan")),
            "Precision":      metrics.get("Precision",   float("nan")),
            "F1_score":       metrics.get("F1_score",    float("nan")),
            "Brier_score":    metrics.get("Brier_score", float("nan")),
            "Threshold":      metrics.get("Threshold",   float("nan")),
        }
        row.update(config)
        append_result(results_csv, row)

        if status == "success" and not np.isnan(val_auprc):
            print(f"   ✅ 完成 ({elapsed:.0f}s) │ "
                  f"val_AUPRC={val_auprc:.4f}  val_AUROC={val_auroc:.4f}  "
                  f"test_AUPRC={metrics.get('AUPRC', float('nan')):.4f}  "
                  f"test_AUROC={metrics.get('AUROC', float('nan')):.4f}  "
                  f"F1={metrics.get('F1_score', float('nan')):.4f}")
        else:
            print(f"   ⚠️  耗時 {elapsed:.0f}s  status={status}")

        if trial_num % 10 == 0:
            print(f"\n{'═' * 70}")
            print(f"  📊 目前 Top-10 排行榜（共完成 {trial_num} 次試驗）")
            print(f"{'═' * 70}")
            print_top_n(results_csv)

        return val_auprc if (status == "success" and not np.isnan(val_auprc)) else 0.0

    return objective

# =====================================================================
# 主程式
# =====================================================================

def parse_search_args():
    p = argparse.ArgumentParser(
        description="Transformer Hyperparameter Search - AUPRC Optimized (Optuna TPE)"
    )
    p.add_argument("--max_trials",        type=int, default=100,
                   help="最大試驗次數（預設 100）")
    p.add_argument("--epochs_per_trial",  type=int, default=50,
                   help="每次試驗的最大 epoch 數（預設 50）")
    p.add_argument("--top_n",             type=int, default=10,
                   help="最終輸出 Top-N 結果（預設 10）")
    p.add_argument("--seed",              type=int, default=42,
                   help="Optuna TPE 隨機種子（預設 42）")
    p.add_argument("--timeout_per_trial", type=int, default=1800,
                   help="每次試驗最長等待秒數（預設 1800s）")
    p.add_argument("--study_name",        type=str, default="transformer_hparam_auprc",
                   help="Optuna study 名稱（預設 transformer_hparam_auprc）")
    return p.parse_args()


def main():
    args = parse_search_args()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = BASE_OUTPUT_DIR / "hparam_search_results.csv"
    db_path     = BASE_OUTPUT_DIR / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name     = args.study_name,
        storage        = storage_url,
        direction      = "maximize",
        sampler        = sampler,
        load_if_exists = True,
    )

    completed = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    remaining = max(0, args.max_trials - completed)

    print("=" * 70)
    print("Transformer Hyperparameter Search  [Optuna TPE / AUPRC Optimized]")
    print("=" * 70)
    print(f"  搜尋腳本  : {TRANSFORMER_SCRIPT}")
    print(f"  資料路徑  : {DATA_CSV}")
    print(f"  輸出根目錄: {BASE_OUTPUT_DIR}")
    print(f"  Study DB  : {db_path}")
    print(f"  目標試驗數: {args.max_trials}  |  已完成: {completed}  |  本次執行: {remaining}")
    print(f"  Epochs/trial: {args.epochs_per_trial}")
    print(f"  排序指標  : {PRIMARY_METRIC}  |  Sampler: TPE (seed={args.seed})")
    print(f"  固定設計  : use_time_weights=1  use_causal_mask=1")
    print("=" * 70)

    if completed > 0:
        print(f"📂 Resume 模式：已有 {completed} 筆完成的試驗，繼續執行")

    if remaining == 0:
        print(f"✅ 已達到 max_trials={args.max_trials}，無需繼續執行。")
        print_top_n(results_csv, args.top_n)
        return

    objective   = make_objective(args, results_csv)
    total_start = time.time()

    study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    total_elapsed = time.time() - total_start

    print(f"\n{'═' * 70}")
    print(f"🏁 搜尋完成！共執行 {remaining} 次試驗，總耗時 {total_elapsed / 60:.1f} 分鐘")
    print(f"{'═' * 70}")

    print(f"\n📊 最終 Top-{args.top_n} 排行榜（依 {PRIMARY_METRIC} 排序）：\n")
    print_top_n(results_csv, args.top_n)

    save_best_config(results_csv, BASE_OUTPUT_DIR)

    if study.best_trial is not None:
        best = study.best_trial
        print(f"\n🏆 Optuna Best Trial #{best.number + 1}  val_AUPRC={best.value:.4f}")

    print(f"\n📁 完整結果 CSV : {results_csv}")
    print(f"📦 Optuna Study : {db_path}")
    print(f"   （可用 optuna-dashboard sqlite:///{db_path} 開啟互動式視覺化）")


if __name__ == "__main__":
    main()
