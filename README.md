# Computer Program (Source Code) Documentation

This folder contains the source code appendix for the thesis *"Transformer-Based Dynamic Risk
Trajectories for Extubation Failure Prediction and Phenotyping"*. It includes all original
source code used in the study, organized by processing stage, along with this documentation
(README).

---

## 1. Execution Environment

| Item | Version / Description |
|------|------|
| Operating System | Windows 11 Pro Education, OS Build 10.0.26200 (64-bit) |
| Code Editor (IDE) | Visual Studio Code 1.129.0 (with Python extension; Conda environment management) |
| Programming Language | Python 3.10.19 (Miniconda, conda environment name: `extubation_env`) |
| GPU / CUDA | NVIDIA GPU, driver version 591.86; PyTorch built with CUDA 12.4 (cu124) |
| Database / Analytics Engine | DuckDB 1.4.1 (**embedded analytical database, not a network server**; operates on the local file `mimic.duckdb` or directly on CSV/Parquet files — no database service or network connection is required at runtime) |
| Web Server | All data processing, model training, and statistical analysis in this study are offline batch scripts. **No web server is used**, and no web/API service needs to be deployed to run this code |
| Statistical Language | R 4.6.1 (used by the `.R` scripts in `stats_analysis/` for producing the statistical tables) |

### Key Third-Party Package Versions

See [`requirements.txt`](requirements.txt) for the complete package/version list. Main packages:

| Category | Package | Version |
|------|------|------|
| Deep Learning | torch / torchvision / torchaudio | 2.6.0 / 0.21.0 / 2.6.0 (cu124) |
| Machine Learning | scikit-learn / xgboost | 1.7.2 / 1.7.6 |
| Data Processing | pandas / numpy / scipy | 2.3.3 / 2.2.6 / 1.15.2 |
| Database | duckdb | 1.4.1 |
| Interpretability | shap / captum | 0.49.1 / 0.9.0 |
| Hyperparameter Search | optuna | 4.8.0 |
| Dimensionality Reduction / Clustering | umap-learn | 0.5.9.post2 |
| Survival Analysis | lifelines | 0.30.0 |
| Statistical Testing | statsmodels | 0.14.5 |
| Visualization | matplotlib / seaborn | 3.10.7 / 0.13.2 |

### Installation (Python)

```bash
conda create -n extubation_env python=3.10
conda activate extubation_env
pip install -r requirements.txt
```

### R Package Versions (used by the `.R` scripts in `stats_analysis/`)

| Package | Version |
|------|------|
| dplyr | 1.2.1 |
| readr | 2.2.0 |
| tidyr | 1.3.2 |
| purrr | 1.2.2 |
| gtsummary | 2.5.1 |
| gt | 1.3.0 |
| ggplot2 | 4.0.3 |

Installation:

```r
install.packages(c("dplyr", "readr", "tidyr", "purrr", "gtsummary", "gt", "ggplot2"))
```

> Some scripts contain absolute file paths from the machine on which they were originally written
> (e.g., `C:\Users\your-username\Desktop\...`). These paths are all consolidated into a single
> path-configuration block at the top of each file, marked with a `【路徑注意】` ("Path Notice")
> comment. To reproduce this pipeline on another machine, simply search for `路徑注意` to find
> every location that needs adjustment — there is no need to search the entire codebase.
> For the small number of scripts that read paths via `argparse` (most programs in
> `model training/` and `clustering/` that accept `--data_csv`, `--output_dir`, etc.), the default
> path values are likewise consolidated into `DEFAULT_*` constants with the same marker, and can
> also be overridden directly via the corresponding command-line arguments at runtime without
> editing the source code.

---

## 2. Data Source

This study uses the [MIMIC-IV](https://physionet.org/content/mimiciv/) (v3.1) critical care
database. Access requires completing the Data Use Agreement and CITI training certification on
PhysioNet; therefore, this folder does not include any raw or derived patient data.

---

## 3. Folder Structure and File Descriptions

```
extubation_failure_prediction_source_code/
├── README.md                          # This documentation file
├── requirements.txt                   # Python package version requirements
├── to duckdb/                         # Raw MIMIC-IV data import and derived table construction (DuckDB)
├── cohort selection/                  # Study cohort selection
├── outcome labeling/                  # Outcome (extubation failure) labeling
├── static feature extraction/         # Static feature extraction
├── time series feature extraction/    # Time-series feature extraction
├── data integration/                  # Feature merging, cleaning, and imputation
├── model training/                    # Model training, evaluation, and interpretability analysis
├── clustering/                        # Risk-trajectory phenotype clustering
├── stats/                             # Statistical helper scripts for sensitivity analysis
└── stats_analysis/                    # Descriptive statistics and thesis statistical tables (Python + R)
```

### 1. `to duckdb/` — Data Import and Derived Table Construction

| File | Function |
|------|------|
| `import_mimic_core_tables_to_duckdb.py` | Imports core MIMIC-IV tables (admissions, patients, icustays) into DuckDB |
| `build_vitalsign.py` | Builds the derived vital-sign table from chartevents |
| `build_ventilator_setting.py` | Builds the derived ventilator-settings table from chartevents |
| `build_oxygen_delivery.py` | Builds the derived oxygen-delivery-device table from chartevents |
| `build_ventilation.py` | Integrates the two tables above to determine respiratory support status at each time point (endotracheal tube, non-invasive, HFNC, etc.) |
| `build_ventilation_mv3d_continuous.py` | Filters continuous invasive mechanical ventilation events |
| `build_stay_subject_map.py` | Builds a stay_id ↔ subject_id lookup table |

### 2. `cohort selection/` — Cohort Selection

| File | Function |
|------|------|
| `compute_mv_day_from_continuous.py` | Extracts the longest continuous mechanical ventilation episode per admission (must be ≥ 3 days) |
| `filter_unique_subject.py` | Keeps each patient's earliest extubation record, converting the dataset to one row per patient |
| `filter_tracheostomy.py` | Excludes extubation cases following tracheostomy |

### 3. `outcome labeling/` — Outcome Labeling

| File | Function |
|------|------|
| `build_extubation_outcome.py` | Builds the binary outcome label: reintubation, NIV use, or death within 48 hours of extubation |

### 4. `static feature extraction/` — Static Feature Extraction

| File | Feature |
|------|------|
| `build_extubation_features_age.py` | Age |
| `build_extubation_features_bmi.py` | BMI |
| `build_extubation_features_sex.py` | Sex |
| `build_extubation_features_CCI.py` | Charlson Comorbidity Index |
| `build_extubation_features_dx_pipeline.py` | Primary diagnosis classification (ICD → CCS/CCSR, for subgroup analysis) |

### 5. `time series feature extraction/` — Time-Series Feature Extraction (12 rows per patient, one per 4-hour bin)

| File | Feature |
|------|------|
| `build_extubation_features_vitalsign_gap4_52to4.py` | Vital signs (heart rate, respiratory rate, SpO2, blood pressure, temperature, glucose) |
| `build_extubation_features_bg_gap4_52to4.py` | Arterial blood gas (pH, PaO2, PaCO2, BE, PaO2/FiO2) |
| `build_extubation_features_lab_gap4_52to4.py` | Lab values (creatinine, WBC, hemoglobin, platelets, anion gap, lactate) |
| `build_extubation_features_gcs_gap4_52to4.py` | Glasgow Coma Scale (GCS) |
| `build_extubation_features_io_gap4_52to4.py` | Fluid input/output and balance |
| `build_extubation_features_ventilator_gap4_52to4.py` | Ventilator parameters (FiO2, MAP, PEEP, tidal volume, MV day) |
| `build_extubation_features_vasopressor_gap4_52to4.py` | Vasopressor use |
| `build_extubation_features_rrt_gap4_52to4.py` | Dialysis use |

### 6. `data integration/` — Feature Integration and Preprocessing

| File | Function |
|------|------|
| `merge_extubation_features_gap4_52to4.py` | Merges static and time-series features |
| `reorder_extubation_features_gap4_52to4.py` | Reorders columns by clinical semantics |
| `clean_extubation_features_gap4_52to4.py` | Removes outliers based on physiologically plausible ranges |
| `add_derived_features_gap4_52to4.py` | Computes derived features (IBW, TV/kg, oxygenation index OI) |
| `impute_extubation_features_gap4_52to4.py` | Imputes missing values (using training-set statistics only, to prevent data leakage), builds missingness masks, and splits into train/val/test |

### 7. `model training/` — Model Training, Evaluation, and Interpretability

| File | Function |
|------|------|
| `transformer_pre_extubation_risk_trajectory.py` | Main model: Transformer Encoder for dynamic risk trajectory prediction |
| `lstm_pre_extubation_risk_trajectory.py` | Baseline model: LSTM |
| `xgb_baseline_pre_extubation_risk_trajectory.py` | Baseline model: XGBoost |
| `rf_baseline_pre_extubation_risk_trajectory.py` | Baseline model: Random Forest |
| `transformer_hyperparam_search.py` / `transformer_hyperparam_search_auprc.py` | Hyperparameter search (Optuna, optimizing AUROC / AUPRC respectively) |
| `compute_model_comparison_stats.py` | Statistical comparison across models (Bootstrap CIs, DeLong's test) |
| `plot_model_comparison_curves.py` | Plots ROC/PR comparison curves |
| `subgroup_analysis_transformer.py` | Model performance analysis across clinical subgroups |
| `ig_individual_interpretability.py` | Case-level interpretability analysis using Integrated Gradients |
| `transformer_waterfall_shap.py` | SHAP feature-importance waterfall plot |
| `plot_phenotype_risk_trajectories.py` / `plot_trajectory_inference_only.py` / `plot_trajectory_poster.py` | Risk trajectory visualization |

### 8. `clustering/` — Phenotyping

| File | Function |
|------|------|
| `extubation_failure_phenotyping.py` | Performs UMAP + clustering on model embeddings to derive risk-trajectory phenotypes |
| `extubation_failure_phenotyping_with_static.py` | Phenotype clustering variant that also incorporates static features |
| `early_cluster_prediction_shap_timebin_minus52.py` | Early time-point (-52h) phenotype prediction and SHAP analysis |
| `sensitivity_leave_first_bin_out.py` | Sensitivity analysis: excluding the first time bin |
| `plot_km_outcomes_by_phenotype.py` / `plot_km_poster.py` | Kaplan-Meier survival curves by phenotype |
| `replot_umap_with_labels.py` | Redraws the UMAP plot with phenotype labels |

### 9. `stats/`, `stats_analysis/` — Statistical Analyses

| File | Function |
|------|------|
| `stats/compute_hours_to_death.py` | Computes the distribution of time-to-death after extubation, used to set the sensitivity-analysis exclusion threshold |
| `stats/build_excl_earlydeath6h_csv.py` | Builds the sensitivity-analysis dataset excluding "very early post-extubation death" cases |
| `stats_analysis/fio2_trend_descriptive_stats.py` | Descriptive statistics of FiO2 trends |
| `stats_analysis/extubation_stats_analysis.R` | Produces Table 3 (demographics and comorbidities), Table 4 (4–8h pre-extubation snapshot), Table 5 (48-hour observation window worst values and trend slopes), and the 12-time-point clinical trajectory plot for Appendix Figure 2 (`trajectory_plot.png`) |

---

## 4. Program Execution Pipeline

```
1. to duckdb/                     Import raw MIMIC-IV data, build derived tables
        │
        ▼
2. cohort selection/               Select study cohort (mechanical ventilation ≥ 3 days)
        │
        ▼
3. outcome labeling/                Build the binary extubation-failure label
        │
        ▼
4. static feature extraction/       Static feature extraction
   time series feature extraction/  Time-series feature extraction (12 four-hour bins)
        │
        ▼
5. data integration/                Merge, clean, impute; split into train/val/test
        │
        ▼
6. model training/                  Train and evaluate Transformer / LSTM / XGBoost / RF
        │
        ▼
7. clustering/                      Risk-trajectory phenotype clustering
        │
        ▼
8. stats/, stats_analysis/          Sensitivity analysis and descriptive statistics
```

The input/output file paths for each step are specified as absolute paths at the top of each
script; adjust them to your environment before running. For methodological details on the
observation-window design, imputation strategy, model architecture, etc., please refer to the
main text of the thesis and the docstrings/block comments in each script.

---

## 5. Coding Conventions

- All source files are written in UTF-8 encoding, mixing Chinese comments with English variable names.
- Every file begins with a block comment (docstring) describing its purpose, inputs, and outputs.
- Key computational steps (e.g., each CTE stage in SQL, feature-imputation logic, model architecture design) are documented with inline comments.
- To prevent data leakage, all standardization (StandardScaler) and imputation statistics are computed from the training set only; this is documented with comments throughout `data integration/` and `model training/`.
