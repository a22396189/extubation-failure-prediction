#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transformer_hyperparam_search.py

使用 Optuna（TPE Bayesian Optimization）對 transformer_pre_extubation_risk_trajectory.py
進行超參數搜尋。

【相較於 Random Search 的改進】
  - TPE 演算法：根據前次試驗結果推測哪個超參數區域更有潛力，比均勻隨機更有效率
  - 連續搜尋空間：lr、dropout、pe_factor 等在連續範圍內精確搜尋，不受離散格點限制
  - 自動 Resume：透過 SQLite 儲存 study，中斷後直接重跑即可繼續，無需額外參數
  - 失敗 trial 回傳 0.0：TPE 自動學習避開造成失敗的超參數區域

【固定設計決策（不參與搜尋）】
  use_time_weights = 1  Prefix Cumulative Training：訓練模型在每個時間前綴做預測，
                        使模型能輸出動態風險軌跡（論文核心設計）
  use_causal_mask  = 1  Causal Self-Attention：每個時間步只能 attend 過去，
                        確保訓練與 prefix inference 行為一致，避免 temporal leakage

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
  python transformer_hyperparam_search.py

  # 自訂試驗次數與每次 epoch 上限（加速搜尋）
  python transformer_hyperparam_search.py --max_trials 150 --epochs_per_trial 30

  # 中斷後繼續（自動從 optuna_study.db 恢復，無需額外參數）
  python transformer_hyperparam_search.py --max_trials 150

【輸出（儲存於 BASE_OUTPUT_DIR）】
  - hparam_search_results.csv   所有試驗結果（逐筆寫入）
  - best_config.json            最佳超參數設定（依 best_val_AUROC 排序）
  - optuna_study.db             Optuna study（供 resume 與事後視覺化）
  - trial_<N>/                  每次試驗的詳細輸出（模型、圖表、log 等）
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


# Optuna 預設日誌較冗長，只保留 WARNING 以上
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 路徑設定（請依實際環境調整）
# =====================================================================
TRANSFORMER_SCRIPT = (
    rf"{EXTUBATION_ROOT}"
    r"\model training\transformer_pre_extubation_risk_trajectory.py"
)
DATA_CSV = (
    rf"{EXTUBATION_ROOT}"
    r"\data\outputs\gap4_52to4\extubation_features_imputed_gap4_52to4.csv"
)
BASE_OUTPUT_DIR = Path(
    rf"{EXTUBATION_ROOT}"
    r"\results\transformer_hparam_search"
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
    # 論文核心設計決策：動態風險軌跡 + Causal Attention
    "use_time_weights": 1,   # Prefix Cumulative Training（產生動態風險軌跡）
    "use_causal_mask":  1,   # Causal mask（訓練與 prefix inference 行為一致）
}

# 主要排序指標
# ⚠️  使用 val set AUROC 排名，避免 test set 洩漏至超參數選擇
PRIMARY_METRIC = "best_val_AUROC"

# =====================================================================
# 超參數建議（Optuna Trial）
# =====================================================================

def suggest_config(trial: optuna.Trial) -> dict:
    """使用 Optuna trial 建議一組超參數設定。"""
    # 耦合架構：整組 categorical 抽樣，確保 nhead 整除 d_model
    arch_idx = trial.suggest_categorical(
        "arch_idx", list(range(len(MODEL_ARCH_CONFIGS)))
    )
    arch = MODEL_ARCH_CONFIGS[arch_idx]

    config = {
        # ── 架構（來自耦合組合）
        "d_model":         arch["d_model"],
        "nhead":           arch["nhead"],
        "dim_ff":          arch["dim_ff"],
        # ── 架構深度
        "num_layers":      trial.suggest_int("num_layers", 2, 4),
        # ── 正則化（連續空間）
        "dropout":         trial.suggest_float("dropout",      0.0,  0.5),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "train_noise":     trial.suggest_float("train_noise",  0.0,  0.1),
        # ── 學習率排程（lr 對數連續）
        "lr":              trial.suggest_float("lr",            1e-5, 5e-4, log=True),
        "lr_warmup_steps": trial.suggest_int("lr_warmup_steps", 2,   10),
        "lr_decay":        trial.suggest_float("lr_decay",      0.90, 1.0),
        # ── Positional Encoding（對數連續）
        "pe_factor":       trial.suggest_float("pe_factor", 0.001, 1.0, log=True),
        # ── Pooling（categorical）
        "pooling":         trial.suggest_categorical("pooling", ["last", "mean"]),
    }
    return config

# =====================================================================
# 工具函數
# =====================================================================

def config_to_cmd(config: dict, output_dir: Path, epochs: int) -> list:
    """將超參數設定轉換為 subprocess 指令列表。"""
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
    """
    從 performance_metrics.csv 讀取評估結果。
    支援兩種格式：
      寬格式（一列數值）: AUROC,AUPRC,...  /  0.81,0.77,...
      長格式（兩欄）:     Metric,Value     /  AUROC,0.81
    """
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
    """將超參數設定格式化為單行摘要。"""
    arch_str  = (f"d{config['d_model']}_h{config['nhead']}"
                 f"_ff{config['dim_ff']}_l{config['num_layers']}")
    train_str = (f"lr{config['lr']:.1e}_wd{config['weight_decay']:.1e}"
                 f"_do{config['dropout']:.2f}_noise{config['train_noise']:.3f}")
    sched_str = (f"warm{config['lr_warmup_steps']}"
                 f"_decay{config['lr_decay']:.3f}")
    arch_opt  = (f"pe{config['pe_factor']:.3f}_pool{config['pooling']}"
                 f"_tw{FIXED_PARAMS['use_time_weights']}"
                 f"_cm{FIXED_PARAMS['use_causal_mask']}")
    return f"{arch_str} | {train_str} | {sched_str} | {arch_opt}"


def append_result(results_csv: Path, row: dict) -> None:
    """將單次試驗結果逐行寫入 CSV（避免中途中斷遺失資料）。"""
    df_new = pd.DataFrame([row])
    write_header = not results_csv.exists()
    df_new.to_csv(results_csv, mode="a", header=write_header, index=False)


def to_python_type(val):
    """將 numpy 數值型別轉換為 Python 原生型別（供 json.dump 使用）。"""
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def print_top_n(results_csv: Path, n: int = 10) -> None:
    """印出目前 Top-N 結果排行榜。"""
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"].dropna(subset=["best_val_AUROC"]).copy()
    df = df.sort_values("best_val_AUROC", ascending=False).drop_duplicates(subset=["trial"])
    if df.empty:
        print("  （尚無成功的試驗）")
        return
    df = df.sort_values(PRIMARY_METRIC, ascending=False).head(n)
    display_cols = [
        "trial", "best_val_AUROC", "AUROC", "AUPRC", "Sensitivity", "Specificity",
        "F1_score", "Brier_score",
        "d_model", "nhead", "dim_ff", "num_layers",
        "lr", "dropout", "pe_factor", "pooling",
        # use_time_weights=1 / use_causal_mask=1 為固定設計，所有 trial 相同，不顯示
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))


def save_best_config(results_csv: Path, output_dir: Path) -> None:
    """將最佳超參數存為 best_config.json（依 best_val_AUROC 排序）。"""
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "success"].dropna(subset=["best_val_AUROC"]).copy()
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
    best_config["best_val_AUROC"]    = float(best_row["best_val_AUROC"])
    best_config["AUROC"]             = float(best_row["AUROC"])
    best_config["trial"]             = int(best_row["trial"])
    # 固定設計決策一併記錄，方便日後完整重現
    best_config["use_time_weights"]  = FIXED_PARAMS["use_time_weights"]
    best_config["use_causal_mask"]   = FIXED_PARAMS["use_causal_mask"]

    out_path = output_dir / "best_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 最佳設定已儲存：{out_path}")
    print(f"   最佳 val_AUROC ：{best_config['best_val_AUROC']:.4f}  "
          f"test_AUROC：{best_config['AUROC']:.4f}  (Trial #{best_config['trial']})")
    print(f"   設定摘要：{format_config_summary(best_config)}")

# =====================================================================
# Optuna Objective
# =====================================================================

def make_objective(args, results_csv: Path):
    """
    建立 Optuna objective function（closure，帶入 args 與 CSV 路徑）。

    回傳值：best_val_AUROC（最大化目標）
      - 成功：回傳實際 val AUROC
      - 失敗 / timeout / error：回傳 0.0，讓 TPE 學習避開此區域
    """
    def objective(trial: optuna.Trial) -> float:
        # trial.number 為 Optuna 內部 0-indexed 編號，轉為 1-indexed 供顯示與目錄命名
        trial_num = trial.number + 1
        trial_dir = BASE_OUTPUT_DIR / f"trial_{trial_num:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = suggest_config(trial)

        print(f"\n{'─' * 70}")
        print(f"🔬 Trial {trial_num}  (Optuna #{trial.number})")
        print(f"   設定：{format_config_summary(config)}")
        print(f"   輸出：{trial_dir}")

        cmd = config_to_cmd(config, trial_dir, args.epochs_per_trial)

        # 將完整指令寫入 trial 目錄，方便事後重現單次試驗
        (trial_dir / "run_command.txt").write_text(
            " \\\n  ".join(cmd), encoding="utf-8"
        )

        t0     = time.time()
        status = "success"

        # 強制子程序使用 UTF-8，避免 emoji 造成 Windows cp950 編碼錯誤
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

        # ── 讀取評估指標 ───────────────────────────────────────────────
        metrics   = read_metrics(trial_dir) if status == "success" else {}
        val_auroc = metrics.get(PRIMARY_METRIC, float("nan"))

        # ── 寫入 CSV（逐筆 append，中途中斷不會遺失已完成資料）──────────
        row = {
            "trial":          trial_num,
            "optuna_trial":   trial.number,
            "status":         status,
            "elapsed_sec":    round(elapsed, 1),
            "output_dir":     str(trial_dir),
            # 超參數選擇指標（val set）— 用於排名
            "best_val_AUROC": val_auroc,
            # 最終報告指標（test set）— 僅供參考
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

        # ── 印出本次結果 ───────────────────────────────────────────────
        if status == "success" and not np.isnan(val_auroc):
            print(f"   ✅ 完成 ({elapsed:.0f}s) │ "
                  f"val_AUROC={val_auroc:.4f}  "
                  f"test_AUROC={metrics.get('AUROC', float('nan')):.4f}  "
                  f"Sens={metrics.get('Sensitivity', float('nan')):.4f}  "
                  f"Spec={metrics.get('Specificity', float('nan')):.4f}  "
                  f"F1={metrics.get('F1_score', float('nan')):.4f}")
        else:
            print(f"   ⚠️  耗時 {elapsed:.0f}s  status={status}")

        # ── 每 10 次試驗印一次排行榜 ───────────────────────────────────
        if trial_num % 10 == 0:
            print(f"\n{'═' * 70}")
            print(f"  📊 目前 Top-10 排行榜（共完成 {trial_num} 次試驗）")
            print(f"{'═' * 70}")
            print_top_n(results_csv)

        # ── 回傳 Optuna 目標值 ─────────────────────────────────────────
        # 失敗時回傳 0.0：TPE 將學習避開造成失敗的超參數區域
        return val_auroc if (status == "success" and not np.isnan(val_auroc)) else 0.0

    return objective

# =====================================================================
# 主程式
# =====================================================================

def parse_search_args():
    p = argparse.ArgumentParser(
        description="Transformer Hyperparameter Search (Optuna TPE)"
    )
    p.add_argument("--max_trials",        type=int, default=100,
                   help="最大試驗次數（預設 100）")
    p.add_argument("--epochs_per_trial",  type=int, default=50,
                   help="每次試驗的最大 epoch 數（預設 50；設為 30 可加速搜尋）")
    p.add_argument("--top_n",             type=int, default=10,
                   help="最終輸出 Top-N 結果（預設 10）")
    p.add_argument("--seed",              type=int, default=42,
                   help="Optuna TPE 隨機種子（預設 42）")
    p.add_argument("--timeout_per_trial", type=int, default=1800,
                   help="每次試驗最長等待秒數（預設 1800s = 30 分鐘）")
    p.add_argument("--study_name",        type=str, default="transformer_hparam",
                   help="Optuna study 名稱（預設 transformer_hparam）")
    return p.parse_args()


def main():
    args = parse_search_args()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = BASE_OUTPUT_DIR / "hparam_search_results.csv"
    db_path     = BASE_OUTPUT_DIR / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"

    # 建立或恢復 Optuna study
    # load_if_exists=True：若 db 已存在則自動 resume，無需任何額外參數
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name   = args.study_name,
        storage      = storage_url,
        direction    = "maximize",      # 最大化 best_val_AUROC
        sampler      = sampler,
        load_if_exists = True,
    )

    completed = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])
    remaining = max(0, args.max_trials - completed)

    print("=" * 70)
    print("Transformer Hyperparameter Search  [Optuna TPE]")
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

    # ── 最終結果 ──────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"🏁 搜尋完成！共執行 {remaining} 次試驗，總耗時 {total_elapsed / 60:.1f} 分鐘")
    print(f"{'═' * 70}")

    print(f"\n📊 最終 Top-{args.top_n} 排行榜（依 {PRIMARY_METRIC} 排序）：\n")
    print_top_n(results_csv, args.top_n)

    save_best_config(results_csv, BASE_OUTPUT_DIR)

    # Optuna 原生最佳 trial 摘要
    if study.best_trial is not None:
        best = study.best_trial
        print(f"\n🏆 Optuna Best Trial #{best.number + 1}  val_AUROC={best.value:.4f}")

    print(f"\n📁 完整結果 CSV : {results_csv}")
    print(f"📦 Optuna Study : {db_path}")
    print(f"   （可用 optuna-dashboard sqlite:///{db_path} 開啟互動式視覺化）")


if __name__ == "__main__":
    main()
