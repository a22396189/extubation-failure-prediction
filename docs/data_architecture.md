# Data Architecture

This document describes how data flows through the project, from the raw MIMIC-IV
download to the final machine-learning dataset. It complements the pipeline
overview in [`README.md`](../README.md) §4 by making the **storage layers**
explicit.

The design follows a simple rule: **keep large source data in columnar files
(Parquet), use DuckDB as an in-process SQL engine for cohort selection and
derived-table construction, and materialise only small, analysis-ready tables.**
No database server (PostgreSQL / MySQL) is used or required.

---

## 1. Layered view

```
                         MIMIC-IV v3.1 (PhysioNet)
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
  L0    │  Raw CSV  ($MIMIC_DATA_DIR/data/mimic-iv-3.1/)         │  immutable,
        │  hosp/*.csv, icu/*.csv                                 │  never edited
        └───────────────────────────────┬───────────────────────┘
                                        │  one-off conversion of the
                                        │  large event tables
                                        ▼
        ┌───────────────────────────────────────────────────────┐
  L1    │  Parquet  ($MIMIC_DATA_DIR/data/mimic-iv-3.1/)         │  columnar,
        │  icu_parquet/chartevents.parquet                       │  compressed,
        │  icu/icustays.parquet                                  │  pushdown-
        │  derived/*.parquet  (bg, rrt, vasoactive_agent,        │  friendly
        │                      first_day_height/weight, …)       │
        └───────────────────────────────┬───────────────────────┘
                                        │  DuckDB reads CSV / Parquet
                                        │  directly (no import needed)
                                        ▼
        ┌───────────────────────────────────────────────────────┐
  L2    │  DuckDB  ($MIMIC_DATA_DIR/mimic.duckdb)                │  in-process
        │                                                       │  SQL engine
        │  "raw / core"  – small dimension tables, imported      │
        │      mimiciv_hosp.admissions / patients                │
        │      mimiciv_icu.icustays                              │
        │                                                       │
        │  "derived"     – MIT-LCP mimic-code concepts, rebuilt  │
        │      ventilator_setting, oxygen_delivery               │
        │      ventilation            (respiratory-support state)│
        │      vitalsign              (emitted as Parquet, L1)   │
        │                                                       │
        │  "study"       – this thesis's cohort / outcome tables │
        │      (materialised to CSV under $EXTUBATION_PROJECT_ROOT│
        │       rather than kept in the DB — see §4)             │
        └───────────────────────────────┬───────────────────────┘
                                        │  feature extraction
                                        │  (SQL + pandas)
                                        ▼
        ┌───────────────────────────────────────────────────────┐
  L3    │  Feature tables  ($EXTUBATION_PROJECT_ROOT/data/)      │  one file per
        │  data/outputs/extubation_features_*.csv     (static)   │  feature group
        │  data/outputs/gap4_52to4/*_gap4_52to4.csv   (12 bins)  │
        └───────────────────────────────┬───────────────────────┘
                                        │  merge → reorder → clean →
                                        │  add derived → impute + split
                                        ▼
        ┌───────────────────────────────────────────────────────┐
  L4    │  ML dataset                                            │  model input
        │  data/outputs/gap4_52to4/                              │
        │      extubation_features_imputed_gap4_52to4.csv        │
        │      (long format, N × 12 rows, has a `split` column)  │
        └───────────────────────────────────────────────────────┘
                                        │
                                        ▼
                       Transformer / LSTM / XGBoost / RF
```

---

## 2. Why this shape

| Question | Answer |
|----------|--------|
| Why DuckDB and not PostgreSQL / MySQL? | The workload is **analytical, single-user, batch**: cohort selection, time alignment, derived-feature construction over the whole of MIMIC-IV. DuckDB is an in-process analytical (OLAP) engine that runs on a local file and reads CSV/Parquet natively, so there is no server to install, secure, or keep running. A client/server RDBMS solves concurrency and transactional-write problems this project does not have. |
| Why keep `chartevents` as Parquet instead of importing it into `mimic.duckdb`? | `chartevents` is by far the largest MIMIC-IV table (hundreds of millions of rows). The `build_*` scripts only ever need a handful of `itemid`s from it. Reading straight from `chartevents.parquet` lets DuckDB apply projection + predicate pushdown and touch only the needed columns/rows, and avoids doubling the data on disk. Only the small dimension tables (`admissions`, `patients`, `icustays`) are imported as physical DuckDB tables, because they are joined repeatedly and are cheap to store. |
| Why is the ML dataset CSV and not Parquet? | The downstream consumers include R (`stats_analysis/*.R`) and several ad-hoc scripts; CSV is the lowest-common-denominator interchange format and the merged dataset (N × 12 rows) is small enough that read time is not a bottleneck. Parquet is used where it matters — the multi-GB event data at L1. |

---

## 3. Script → layer map

| Stage / script | Reads | Writes |
|----------------|-------|--------|
| `to duckdb/import_mimic_core_tables_to_duckdb.py` | L0 `admissions.csv`, `patients.csv`, `icustays.csv` | L2 `mimiciv_hosp.*`, `mimiciv_icu.icustays` |
| `to duckdb/build_ventilator_setting.py` | L1 `chartevents.parquet` | L2 table `ventilator_setting` + L1 `derived/ventilator_setting.parquet` |
| `to duckdb/build_oxygen_delivery.py` | L1 `chartevents.parquet` | L2 table `oxygen_delivery` |
| `to duckdb/build_ventilation.py` | L2 `ventilator_setting`, `oxygen_delivery` | L2 table `ventilation` |
| `to duckdb/build_ventilation_mv3d_continuous.py` | L2 `ventilation` | L3 `data/outputs/ventilation_mv3d_continuous.csv` |
| `to duckdb/build_vitalsign.py` | L1 `chartevents.parquet` | L1 `derived/vitalsign.parquet` |
| `to duckdb/build_stay_subject_map.py` | L0 `icustays.csv` | L3 `data/outputs/stay_subject_map.csv` |
| `cohort selection/*` | L3 cohort CSVs, L2 (`filter_tracheostomy` uses `chartevents` via DuckDB) | L3 cohort CSVs |
| `outcome labeling/build_extubation_outcome.py` | L3 cohort CSV, L0 `admissions.csv`, L2 | L3 `extubation_outcome.csv` |
| `static feature extraction/*` | L3 outcome CSV, L0/L1 dimension + `derived/` files | L3 `data/outputs/extubation_features_*.csv` |
| `time series feature extraction/*` | L3 outcome CSV, L1 `chartevents.parquet` / `derived/*.parquet`, L0 `labevents.csv` / `inputevents.csv` / `outputevents.csv` | L3 `data/outputs/gap4_52to4/*_gap4_52to4.csv` |
| `data integration/merge → reorder → clean → add_derived → impute` | L3 feature CSVs | L4 `extubation_features_imputed_gap4_52to4.csv` |
| `model training/*`, `clustering/*` | L4 ML dataset (`--data_csv`) | `results/` (metrics, predictions, plots) |

---

## 4. Design notes / scope

This is a research pipeline written incrementally alongside the thesis. A few
points are simplifications rather than the "ideal" version of the architecture
above; they are recorded here for anyone extending the code.

1. **DuckDB schemas are not namespaced.** The derived tables
   (`ventilator_setting`, `oxygen_delivery`, `ventilation`) live in DuckDB's
   default `main` schema, not in a dedicated `derived` schema. The "raw / derived
   / study" split in §1 is therefore a *conceptual* layering, enforced by naming
   and by this document, not by the database.

2. **The `study` layer is file-based.** Cohort, outcome and feature tables are
   passed between stages as CSV under `$EXTUBATION_PROJECT_ROOT/data/`, not as
   DuckDB tables. This keeps each stage independently runnable and diff-able, at
   the cost of some redundant re-reads.

3. **Derived-table output format is not uniform.** `build_vitalsign.py` emits
   Parquet only (the pattern this project would standardise on).
   `build_ventilator_setting.py` creates a DuckDB table (consumed by
   `build_ventilation.py`) *and* exports a Parquet copy to `derived/` for the
   time-series feature scripts. `build_oxygen_delivery.py` creates a DuckDB
   table only, since nothing downstream reads it as Parquet.

4. **Intermediate feature files are CSV.** Converting L3 (`data/outputs/**.csv`)
   to Parquet would speed up re-runs of `data integration/` and
   `model training/`; it was left as CSV for readability and R interop. The ML
   dataset at L4 is deliberately CSV (see §2).

5. **No cloud / blob storage.** Everything runs from local disk on a single
   workstation. If the project ever needs cloud backup, multi-machine access, or
   an automated pipeline, the L0/L1 file layers can move to object storage
   (S3 / Azure Blob) with no change to the DuckDB or feature-extraction code —
   DuckDB can query remote Parquet directly.
