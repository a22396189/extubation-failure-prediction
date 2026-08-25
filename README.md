# 電腦程式（Source Code）說明文件

本資料夾為論文《基於 Transformer 之動態風險軌跡模型用於預測拔管失敗與表型分型》之附錄電腦程式，
內容為研究過程中使用之全部原始程式碼（source code），依處理流程分類存放，並附上本說明文件（README）。

---

## 一、執行環境（Environment）

| 項目 | 版本 / 說明 |
|------|------|
| 作業系統 | Windows 11 Pro Education，OS Build 10.0.26200（64 位元） |
| 程式編輯器（IDE） | Visual Studio Code 1.129.0（含 Python 擴充套件；使用 Conda 環境管理） |
| 程式語言 | Python 3.10.19（Miniconda，conda 環境名稱：`extubation_env`） |
| GPU / CUDA | NVIDIA GPU，驅動版本 591.86；PyTorch 以 CUDA 12.4（cu124）版本編譯 |
| 資料庫 / 分析引擎 | DuckDB 1.4.1（**內嵌式（embedded）分析資料庫，非網路伺服器**；以本機檔案 `mimic.duckdb` 或直接讀取 CSV/Parquet 檔案運作，程式執行時不需啟動任何資料庫服務或網路連線） |
| 網路伺服器 | 本研究之資料處理、模型訓練與統計分析皆為離線批次程式（batch script），**未使用任何網路伺服器（Web Server）**，無需部署 Web/API 服務即可執行 |
| 統計分析語言 | R 4.6.1（用於 `stats_analysis/` 內三支 `.R` 統計表格程式） |

### 主要第三方套件版本

完整套件與版本清單見 [`requirements.txt`](requirements.txt)，主要套件如下：

| 類別 | 套件 | 版本 |
|------|------|------|
| 深度學習 | torch / torchvision / torchaudio | 2.6.0 / 0.21.0 / 2.6.0（cu124） |
| 機器學習 | scikit-learn / xgboost | 1.7.2 / 1.7.6 |
| 資料處理 | pandas / numpy / scipy | 2.3.3 / 2.2.6 / 1.15.2 |
| 資料庫 | duckdb | 1.4.1 |
| 可解釋性 | shap / captum | 0.49.1 / 0.9.0 |
| 超參數搜尋 | optuna | 4.8.0 |
| 降維與分群 | umap-learn | 0.5.9.post2 |
| 存活分析 | lifelines | 0.30.0 |
| 統計檢定 | statsmodels | 0.14.5 |
| 視覺化 | matplotlib / seaborn | 3.10.7 / 0.13.2 |

### 安裝方式（Python）

```bash
conda create -n extubation_env python=3.10
conda activate extubation_env
pip install -r requirements.txt
```

### R 套件版本（`stats_analysis/` 內 `.R` 程式使用）

| 套件 | 版本 |
|------|------|
| dplyr | 1.2.1 |
| readr | 2.2.0 |
| tidyr | 1.3.2 |
| purrr | 1.2.2 |
| gtsummary | 2.5.1 |
| gt | 1.3.0 |
| ggplot2 | 4.0.3 |

安裝方式：

```r
install.packages(c("dplyr", "readr", "tidyr", "purrr", "gtsummary", "gt", "ggplot2"))
```

> 部分程式檔案中包含撰寫時所在機器的絕對路徑（例如 `C:\Users\your-username\Desktop\...`）。
> 這些路徑皆集中放在每個檔案開頭一個獨立的路徑設定區塊內，並以 `【路徑注意】` 註解標示，
> 於其他環境重現時，只需搜尋 `路徑注意` 即可找到所有需要調整的位置，不需搜尋整份程式碼。
> 少數以 `argparse` 讀取路徑的模型訓練／分群腳本（`model training/`、`clustering/` 內含
> `--data_csv`、`--output_dir` 等參數的程式），其路徑預設值同樣以 `DEFAULT_*` 常數集中定義並標註，
> 執行時亦可直接以對應的命令列參數覆寫，不需修改程式碼本身。

---

## 二、資料來源

本研究使用 [MIMIC-IV](https://physionet.org/content/mimiciv/)（v3.1）重症照護資料庫，
需於 PhysioNet 完成資料使用協議（Data Use Agreement）與 CITI 訓練認證後方可取得，
故本資料夾不隨附任何原始或衍生病人資料。

---

## 三、資料夾結構與檔案說明

```
拔管失敗預測_論文電腦程式/
├── README.md                          # 本說明文件
├── requirements.txt                   # Python 套件版本需求
├── to duckdb/                         # 原始 MIMIC-IV 資料匯入與衍生表建立（DuckDB）
├── cohort selection/                  # 研究族群篩選
├── outcome labeling/                  # 結果標籤（拔管失敗）建立
├── static feature extraction/         # 靜態特徵萃取
├── time series feature extraction/    # 時序特徵萃取
├── data Integration/                  # 特徵整合、清理、補值
├── model training/                    # 模型訓練、評估與可解釋性分析
├── clustering/                        # 風險軌跡表型分群（Phenotyping）
├── stats/                             # 敏感度分析用統計輔助程式
└── stats_analysis/                    # 描述性統計分析與論文統計表格（Python + R）
```

### 1. `to duckdb/` — 資料匯入與衍生表建立

| 檔案 | 功能 |
|------|------|
| `import_mimic_core_tables_to_duckdb.py` | 匯入 MIMIC-IV 核心表（admissions、patients、icustays）至 DuckDB |
| `build_vitalsign.py` | 由 chartevents 建立生命徵象衍生表 |
| `build_ventilator_setting.py` | 由 chartevents 建立呼吸器設定衍生表 |
| `build_oxygen_delivery.py` | 由 chartevents 建立給氧裝置衍生表 |
| `build_ventilation.py` | 整合上述兩表，判斷各時間點呼吸支持狀態（含氣管內管、非侵襲、HFNC 等） |
| `build_ventilation_mv3d_continuous.py` | 篩選連續機械通氣（invasive ventilation）事件 |
| `build_stay_subject_map.py` | 建立 stay_id ↔ subject_id 對照表 |

### 2. `cohort selection/` — 族群篩選

| 檔案 | 功能 |
|------|------|
| `compute_mv_day_from_continuous.py` | 萃取每次住院最長連續機械通氣時段（需 ≥ 3 天） |
| `filter_unique_subject.py` | 每位病患保留最早拔管紀錄，轉為以病人為單位 |
| `filter_tracheostomy.py` | 排除氣切後拔管案例 |

### 3. `outcome labeling/` — 結果標籤

| 檔案 | 功能 |
|------|------|
| `build_extubation_outcome.py` | 建立二元標籤：拔管後 48 小時內再插管、使用 NIV 或死亡 |

### 4. `static feature extraction/` — 靜態特徵萃取

| 檔案 | 特徵 |
|------|------|
| `build_extubation_features_age.py` | 年齡 |
| `build_extubation_features_bmi.py` | BMI |
| `build_extubation_features_sex.py` | 性別 |
| `build_extubation_features_CCI.py` | Charlson 共病指數 |
| `build_extubation_features_dx_pipeline.py` | 主診斷分類（ICD → CCS/CCSR，供亞群分析） |

### 5. `time series feature extraction/` — 時序特徵萃取（每位病患 12 筆，對應 12 個 4 小時區間）

| 檔案 | 特徵 |
|------|------|
| `build_extubation_features_vitalsign_gap4_52to4.py` | 生命徵象（心跳、呼吸、血氧、血壓、體溫、血糖） |
| `build_extubation_features_bg_gap4_52to4.py` | 動脈血氣（pH、PaO2、PaCO2、BE、PaO2/FiO2） |
| `build_extubation_features_lab_gap4_52to4.py` | 檢驗值（肌酸酐、WBC、血色素、血小板、Anion Gap、乳酸） |
| `build_extubation_features_gcs_gap4_52to4.py` | 昏迷指數（GCS） |
| `build_extubation_features_io_gap4_52to4.py` | 輸入輸出量與體液平衡 |
| `build_extubation_features_ventilator_gap4_52to4.py` | 呼吸器參數（FiO2、MAP、PEEP、潮氣容積、MV 天數） |
| `build_extubation_features_vasopressor_gap4_52to4.py` | 升壓劑使用 |
| `build_extubation_features_rrt_gap4_52to4.py` | 透析使用 |

### 6. `data integration/` — 特徵整合與前處理

| 檔案 | 功能 |
|------|------|
| `merge_extubation_features_gap4_52to4.py` | 合併靜態與時序特徵 |
| `reorder_extubation_features_gap4_52to4.py` | 依臨床語意重排欄位 |
| `clean_extubation_features_gap4_52to4.py` | 依生理合理範圍清除異常值 |
| `add_derived_features_gap4_52to4.py` | 計算衍生特徵（IBW、TV/kg、氧合指數 OI） |
| `impute_extubation_features_gap4_52to4.py` | 缺失值補值（僅用訓練集統計量，避免資料洩漏），建立缺失遮罩並切分 train/val/test |

### 7. `model training/` — 模型訓練、評估與可解釋性

| 檔案 | 功能 |
|------|------|
| `transformer_pre_extubation_risk_trajectory.py` | 主模型：Transformer Encoder 動態風險軌跡預測 |
| `lstm_pre_extubation_risk_trajectory.py` | 基線模型：LSTM |
| `xgb_baseline_pre_extubation_risk_trajectory.py` | 基線模型：XGBoost |
| `rf_baseline_pre_extubation_risk_trajectory.py` | 基線模型：Random Forest |
| `transformer_hyperparam_search.py` / `transformer_hyperparam_search_auprc.py` | 超參數搜尋（Optuna，分別以 AUROC / AUPRC 為目標） |
| `compute_model_comparison_stats.py` | 模型間效能統計比較（Bootstrap、DeLong 檢定） |
| `plot_model_comparison_curves.py` | 繪製 ROC/PR 比較曲線 |
| `subgroup_analysis_transformer.py` | 各臨床亞群之模型效能分析 |
| `ig_individual_interpretability.py` | 以 Integrated Gradients 進行個案層級可解釋性分析 |
| `transformer_waterfall_shap.py` | SHAP 特徵重要性瀑布圖 |
| `plot_phenotype_risk_trajectories.py` / `plot_trajectory_inference_only.py` / `plot_trajectory_poster.py` | 風險軌跡視覺化 |

### 8. `clustering/` — 表型分群（Phenotyping）

| 檔案 | 功能 |
|------|------|
| `extubation_failure_phenotyping.py` | 以模型嵌入向量進行 UMAP + 分群，建立風險軌跡表型 |
| `extubation_failure_phenotyping_with_static.py` | 加入靜態特徵之表型分群版本 |
| `early_cluster_prediction_shap_timebin_minus52.py` | 早期時間點（-52h）表型預測與 SHAP 分析 |
| `sensitivity_leave_first_bin_out.py` | 敏感度分析：排除首個時間區間 |
| `plot_km_outcomes_by_phenotype.py` / `plot_km_poster.py` | 各表型之 Kaplan-Meier 存活曲線 |
| `replot_umap_with_labels.py` | 重繪標註表型標籤之 UMAP 圖 |

### 9. `stats/`、`stats_analysis/` — 統計輔助分析

| 檔案 | 功能 |
|------|------|
| `stats/compute_hours_to_death.py` | 計算拔管後至死亡時數分布，作為敏感度分析排除門檻依據 |
| `stats/build_excl_earlydeath6h_csv.py` | 建立排除「拔管後極早期死亡」個案之敏感度分析資料集 |
| `stats_analysis/fio2_trend_descriptive_stats.py` | FiO2 趨勢描述性統計 |
| `stats_analysis/extubation_stats_analysis.R` | 產出 Table 3（人口學與共病）、Table 4（拔管前 4–8h snapshot）、Table 5（48 小時觀測窗最差值與趨勢 slope），並繪製附錄圖 2 的 12 個時間步之臨床軌跡圖（`trajectory_plot.png`） |

---

## 四、程式執行流程（Pipeline）

```
1. to duckdb/                     匯入 MIMIC-IV 原始資料，建立衍生表
        │
        ▼
2. cohort selection/               篩選機械通氣 ≥ 3 天之研究族群
        │
        ▼
3. outcome labeling/                建立拔管失敗二元標籤
        │
        ▼
4. static feature extraction/       靜態特徵萃取
   time series feature extraction/  時序特徵萃取（12 個 4 小時區間）
        │
        ▼
5. data Integration/                合併、清理、補值，切分 train/val/test
        │
        ▼
6. model training/                  Transformer / LSTM / XGBoost / RF 訓練與評估
        │
        ▼
7. clustering/                      風險軌跡表型分群
        │
        ▼
8. stats/、stats_analysis/          敏感度分析與描述性統計
```

各步驟之輸入／輸出檔案路徑於各程式檔案開頭以絕對路徑寫明，執行前請依實際環境調整。
觀測視窗設計、補值策略、模型架構等方法學細節，請參閱論文正文與各程式檔案內之
docstring／區塊註解。

---

## 五、程式碼撰寫慣例

- 所有程式檔案均以 UTF-8 編碼撰寫，中文註解與英文變數名稱混用。
- 每個檔案開頭皆附有目的、輸入、輸出說明之區塊註解（docstring）。
- 關鍵計算步驟（如 SQL 各 CTE 階段、特徵補值邏輯、模型架構設計）均附行內註解說明。
- 為防止資料洩漏，所有標準化（StandardScaler）與補值統計量均僅由訓練集（train set）計算，
  相關寫法於 `data Integration/` 與 `model training/` 各檔案中均有註解標示。
