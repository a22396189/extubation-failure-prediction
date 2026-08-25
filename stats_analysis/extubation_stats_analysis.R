library(dplyr)
library(readr)
library(gtsummary)
library(gt)
library(tidyr)
library(purrr)
library(ggplot2)

# ============================================================
# 資料說明
# ============================================================
# 來源：extubation_features_enhanced（補值前，真實 NaN）
# 對象：全部 6,809 位病患（不需 train/val/test split）
# 時序：每位病患 12 個 time_bin（-52, -48, ..., -8），每步 4h
#
# 表格架構：
#   Table 1  人口學與共病（靜態特徵，每人一筆）
#   Table 2  拔管前 4-8h 臨床狀態（time_bin == -8 的 snapshot）
#   Table 3  48h 觀測窗摘要（最差值 + 趨勢，各變數依資料密度選策略）
#   Figure   臨床軌跡圖（12 個時間步的中位數 ± IQR）
# ============================================================

# 路徑設定：讀取環境變數 EXTUBATION_PROJECT_ROOT（見 repo 根目錄的 .env.example）
EXTUBATION_ROOT <- Sys.getenv("EXTUBATION_PROJECT_ROOT")
if (EXTUBATION_ROOT == "") {
  stop(
    "Environment variable EXTUBATION_PROJECT_ROOT is not set. ",
    "Copy .env.example to .env (or set it directly) and point it to your ",
    "local extubation_failure_prediction project root."
  )
}
DATA_PATH <- file.path(EXTUBATION_ROOT, "data/outputs/gap4_52to4/extubation_features_enhanced_gap4_52to4.csv")
RESULTS_DIR <- file.path(EXTUBATION_ROOT, "results")

raw_df <- read_csv(DATA_PATH)

# ============================================================
# 輔助函數：計算線性斜率（每 4h 的變化量）
# 要求至少 min_obs 筆非缺失觀測，否則回傳 NA
# ============================================================
calc_slope <- function(time_bins, values, min_obs = 3) {
  valid <- !is.na(values)
  if (sum(valid) < min_obs) return(NA_real_)
  coef(lm(values[valid] ~ time_bins[valid]))[2] * 4  # ×4 → 每 4h 的變化量
}

# ============================================================
# 第一步：建立各分析子集
# ============================================================

# --- (A) 靜態特徵（每人一筆） ---
df_static <- raw_df %>%
  distinct(stay_id, .keep_all = TRUE) %>%
  mutate(
    Group            = factor(Extubation_failure, levels = c(0, 1), labels = c("Success", "Failure")),
    sex              = factor(sex),
    Vasopressor_any  = factor(as.integer(stay_id %in% {
      raw_df %>% filter(Vasopressor_use  == 1) %>% pull(stay_id) %>% unique()
    }), levels = c(0, 1), labels = c("No", "Yes")),
    Hemodialysis_any = factor(as.integer(stay_id %in% {
      raw_df %>% filter(Hemodialysis_use == 1) %>% pull(stay_id) %>% unique()
    }), levels = c(0, 1), labels = c("No", "Yes"))
  )

# --- (B) Snapshot：time_bin == -8（拔管決策前 4-8h） ---
df_snapshot <- raw_df %>%
  filter(time_bin == -8) %>%
  mutate(
    Group            = factor(Extubation_failure, levels = c(0, 1), labels = c("Success", "Failure")),
    sex              = factor(sex),
    Vasopressor_use  = factor(Vasopressor_use,  levels = c(0, 1), labels = c("No", "Yes")),
    Hemodialysis_use = factor(Hemodialysis_use, levels = c(0, 1), labels = c("No", "Yes"))
  )

# --- (C) 48h 摘要：min / max / slope ---
#
# 聚合策略說明（依資料密度）：
#   生命徵象 (HR, RR, SpO2, MBP, Temp, FiO2, PEEP, MAP, GCS, io_balance, Glucose)
#     → min + max + slope（幾乎所有病患有 ≥3 筆，slope 可靠）
#
#   血氣 (pH, PaO2, PaCO2, BE, PaO2_FiO2_Ratio, OI)
#     → min + max（約 60-70% 病患有資料）
#     → slope 需 ≥3 筆（min_obs=3），約 58-59% 可計算
#
#   Lab (Cr, WBC, Hb, PLT, AnionGap)
#     → min + max 即可（每 4h 一格，lab 通常每天只測 1-2 次；≥3 筆只有 42-68%）
#
#   Lactate（83.6% 缺失，僅 29% 有 ≥3 筆）
#     → 僅 max（峰值乳酸有臨床意義）
#
#   Tidal_Volume / TV_per_kg（54-63% 缺失）
#     → min + max
#
#   二元暴露 (Vasopressor, Hemodialysis)
#     → any_use（48h 內曾使用）

df_window <- raw_df %>%
  group_by(stay_id, Extubation_failure) %>%
  summarise(
    # ── 生命徵象 ──
    HR_min      = min(heart_rate,  na.rm = TRUE),
    HR_max      = max(heart_rate,  na.rm = TRUE),
    HR_slope    = calc_slope(time_bin, heart_rate),    # bpm / 4h
    RR_min      = min(resp_rate,   na.rm = TRUE),
    RR_max      = max(resp_rate,   na.rm = TRUE),
    RR_slope    = calc_slope(time_bin, resp_rate),
    SpO2_min    = min(spo2,        na.rm = TRUE),
    SpO2_slope  = calc_slope(time_bin, spo2),
    MBP_min     = min(mbp,         na.rm = TRUE),
    MBP_slope   = calc_slope(time_bin, mbp),
    Temp_min    = min(temperature, na.rm = TRUE),
    Temp_max    = max(temperature, na.rm = TRUE),
    GCS_min     = min(GCS,         na.rm = TRUE),
    GCS_slope   = calc_slope(time_bin, GCS),

    # ── 呼吸器設定 ──
    FiO2_max    = max(FiO2,         na.rm = TRUE),
    FiO2_slope  = calc_slope(time_bin, FiO2),          # ΔFiO2 / 4h（負值 = 下調 = 改善）
    PEEP_max    = max(PEEP,         na.rm = TRUE),
    PEEP_slope  = calc_slope(time_bin, PEEP),
    MAP_max     = max(MAP,          na.rm = TRUE),     # Mean Airway Pressure
    TV_per_kg_min = min(TV_per_kg,  na.rm = TRUE),
    TV_per_kg_max = max(TV_per_kg,  na.rm = TRUE),
    MV_day_max  = max(MV_day,       na.rm = TRUE),

    # ── 液體平衡 ──
    io_balance_min   = min(io_balance, na.rm = TRUE),
    io_balance_max   = max(io_balance, na.rm = TRUE),
    io_balance_slope = calc_slope(time_bin, io_balance),

    # ── 血氣（約 60% 病患有 ≥3 筆，slope 有意義） ──
    pH_min          = min(pH,              na.rm = TRUE),
    pH_slope        = calc_slope(time_bin, pH),
    PaO2_min        = min(PaO2,            na.rm = TRUE),
    PaO2_slope      = calc_slope(time_bin, PaO2),
    PF_ratio_min    = min(PaO2_FiO2_Ratio, na.rm = TRUE),
    PF_ratio_slope  = calc_slope(time_bin, PaO2_FiO2_Ratio),  # 正 = 改善
    PaCO2_max       = max(PaCO2,           na.rm = TRUE),
    BE_min          = min(BE,              na.rm = TRUE),
    OI_max          = max(OI,              na.rm = TRUE),

    # ── 實驗室（僅 min/max，lab 通常每天測 1-2 次） ──
    Cr_max          = max(Cr,       na.rm = TRUE),
    WBC_max         = max(WBC,      na.rm = TRUE),
    Hb_min          = min(Hb,       na.rm = TRUE),
    PLT_min         = min(PLT,      na.rm = TRUE),
    AnionGap_max    = max(AnionGap, na.rm = TRUE),
    Glucose_max     = max(Glucose,  na.rm = TRUE),

    # ── Lactate（83.6% 缺失，僅 max 有意義） ──
    Lactate_max     = max(Lactate,  na.rm = TRUE),

    # ── 二元暴露 ──
    Vasopressor_any  = factor(as.integer(any(Vasopressor_use  == 1, na.rm = TRUE)),
                              levels = c(0, 1), labels = c("No", "Yes")),
    Hemodialysis_any = factor(as.integer(any(Hemodialysis_use == 1, na.rm = TRUE)),
                              levels = c(0, 1), labels = c("No", "Yes")),

    .groups = "drop"
  ) %>%
  # 修正 Inf/-Inf（全部缺失時 min/max 回傳 Inf/-Inf）
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA_real_, .))) %>%
  mutate(Group = factor(Extubation_failure, levels = c(0, 1), labels = c("Success", "Failure")))

# ============================================================
# 第二步：製作統計表格
# ============================================================

theme_gtsummary_journal(journal = "jama")

# ----------------------------------------------------------
# Table 1: 人口學與共病（靜態）
# ----------------------------------------------------------
table1 <- df_static %>%
  select(Group, age, sex, BMI, Charlson_Score,
         Vasopressor_any, Hemodialysis_any) %>%
  tbl_summary(
    by = Group,
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    label = list(
      age              = "Age (years)",
      sex              = "Sex",
      BMI              = "BMI (kg/m²)",
      Charlson_Score   = "Charlson Comorbidity Index",
      Vasopressor_any  = "Vasopressor Use (any, 48h window)",
      Hemodialysis_any = "Hemodialysis Use (any, 48h window)"
    ),
    digits  = list(all_continuous() ~ 1),
    missing = "no"
  ) %>%
  add_p(test = list(all_continuous()  ~ "wilcox.test",
                    all_categorical() ~ "chisq.test")) %>%
  add_overall() %>%
  modify_caption("**Table 1. Baseline Characteristics**") %>%
  bold_labels()

# ----------------------------------------------------------
# Table 2: 拔管前 4-8h Snapshot（time_bin == -8）
# ----------------------------------------------------------
table2 <- df_snapshot %>%
  select(Group,
         heart_rate, resp_rate, spo2, mbp, temperature, GCS,
         FiO2, MAP, PEEP, TV_per_kg, MV_day, io_balance,
         pH, PaO2, PaCO2, BE, PaO2_FiO2_Ratio, OI,
         Cr, WBC, Hb, PLT, AnionGap, Lactate, Glucose,
         Vasopressor_use, Hemodialysis_use) %>%
  tbl_summary(
    by = Group,
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    label = list(
      heart_rate      = "Heart Rate (bpm)",
      resp_rate       = "Respiratory Rate (breaths/min)",
      spo2            = "SpO2 (%)",
      mbp             = "Mean Arterial Pressure (mmHg)",
      temperature     = "Temperature (°C)",
      GCS             = "GCS Score",
      FiO2            = "FiO2",
      MAP             = "Mean Airway Pressure (cmH2O)",
      PEEP            = "PEEP (cmH2O)",
      TV_per_kg       = "Tidal Volume per IBW (mL/kg)",
      MV_day          = "Duration of IMV (days)",
      io_balance      = "Fluid Balance (mL)",
      pH              = "Arterial pH",
      PaO2            = "PaO2 (mmHg)",
      PaCO2           = "PaCO2 (mmHg)",
      BE              = "Base Excess (mEq/L)",
      PaO2_FiO2_Ratio = "PaO2/FiO2 Ratio",
      OI              = "Oxygenation Index",
      Cr              = "Creatinine (mg/dL)",
      WBC             = "WBC (×10⁹/L)",
      Hb              = "Hemoglobin (g/dL)",
      PLT             = "Platelet Count (×10⁹/L)",
      AnionGap        = "Anion Gap (mEq/L)",
      Lactate         = "Lactate (mmol/L)",
      Glucose         = "Glucose (mg/dL)",
      Vasopressor_use = "Vasopressor Use",
      Hemodialysis_use = "Hemodialysis Use"
    ),
    digits = list(all_continuous() ~ 1, pH ~ 2, Lactate ~ 2, Cr ~ 2, FiO2 ~ 2),
    missing = "no"   # Unknown 行移除；高缺失率變數見腳注
  ) %>%
  add_p(test = list(all_continuous()  ~ "wilcox.test",
                    all_categorical() ~ "fisher.test")) %>%
  modify_caption(
    "**Table 2. Clinical Parameters at Pre-Extubation Window (4-8h Before Extubation)**"
  ) %>%
  modify_footnote(
    update = everything() ~ "Values are median (Q1, Q3) or n (%). P-values from Mann-Whitney U test for continuous variables and Fisher's exact test for categorical variables. Variables with substantial missing data at this time window: arterial blood gas variables (pH, PaO2, PaCO2, BE, PaO2/FiO2 ratio: ~30% available), tidal volume (~45% available), and lactate (~17% available), reflecting the intermittent nature of these measurements in clinical practice."
  ) %>%
  bold_labels()

# ----------------------------------------------------------
# Table 3: 48h 觀測窗摘要（最差值 + 趨勢）
#
# 臨床詮釋方向：
#   FiO2_slope < 0    → FiO2 下調（改善）
#   PF_ratio_slope > 0 → PaO2/FiO2 上升（改善）
#   GCS_slope > 0     → 意識改善
#   RR_slope > 0      → 呼吸惡化
#
# slope 單位：每 4h 的變化量
# slope 僅對有 ≥3 筆觀測的病患計算（其餘 NA）
# ----------------------------------------------------------
table3 <- df_window %>%
  select(Group,
         # 生命徵象（min + max + slope）
         HR_min, HR_max, HR_slope,
         RR_min, RR_max, RR_slope,
         SpO2_min, SpO2_slope,
         MBP_min, MBP_slope,
         GCS_min, GCS_slope,
         # 呼吸器（max + slope）
         FiO2_max, FiO2_slope,
         PEEP_max, PEEP_slope,
         TV_per_kg_min, TV_per_kg_max,
         MV_day_max,
         io_balance_min, io_balance_max,
         # 血氣（min/max + slope）
         pH_min, pH_slope,
         PaO2_min, PaO2_slope,
         PF_ratio_min, PF_ratio_slope,
         PaCO2_max, BE_min, OI_max,
         # Lab（min/max）
         Cr_max, WBC_max, Hb_min, PLT_min, AnionGap_max,
         Glucose_max, Lactate_max,
         # 二元
         Vasopressor_any, Hemodialysis_any
  ) %>%
  tbl_summary(
    by = Group,
    statistic = list(
      all_continuous()  ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    label = list(
      HR_min           = "Heart Rate, minimum (bpm)",
      HR_max           = "Heart Rate, maximum (bpm)",
      HR_slope         = "Heart Rate, trend (bpm/4h)",
      RR_min           = "Respiratory Rate, minimum (breaths/min)",
      RR_max           = "Respiratory Rate, maximum (breaths/min)",
      RR_slope         = "Respiratory Rate, trend (breaths/min per 4h)",
      SpO2_min         = "SpO2, minimum (%)",
      SpO2_slope       = "SpO2, trend (%/4h)",
      MBP_min          = "Mean Arterial Pressure, minimum (mmHg)",
      MBP_slope        = "Mean Arterial Pressure, trend (mmHg/4h)",
      GCS_min          = "GCS, minimum",
      GCS_slope        = "GCS, trend (points/4h)",
      FiO2_max         = "FiO2, maximum",
      FiO2_slope       = "FiO2, trend per 4h (negative = weaning)",
      PEEP_max         = "PEEP, maximum (cmH2O)",
      PEEP_slope       = "PEEP, trend (cmH2O/4h)",
      TV_per_kg_min    = "Tidal Volume/IBW, minimum (mL/kg)",
      TV_per_kg_max    = "Tidal Volume/IBW, maximum (mL/kg)",
      MV_day_max       = "Duration of IMV (days)",
      io_balance_min   = "Fluid Balance, minimum (mL)",
      io_balance_max   = "Fluid Balance, maximum (mL)",
      pH_min           = "Arterial pH, minimum",
      pH_slope         = "Arterial pH, trend per 4h",
      PaO2_min         = "PaO2, minimum (mmHg)",
      PaO2_slope       = "PaO2, trend (mmHg/4h)",
      PF_ratio_min     = "PaO2/FiO2 Ratio, minimum",
      PF_ratio_slope   = "PaO2/FiO2 Ratio, trend per 4h (positive = improving)",
      PaCO2_max        = "PaCO2, maximum (mmHg)",
      BE_min           = "Base Excess, minimum (mEq/L)",
      OI_max           = "Oxygenation Index, maximum",
      Cr_max           = "Creatinine, maximum (mg/dL)",
      WBC_max          = "WBC, maximum (×10⁹/L)",
      Hb_min           = "Hemoglobin, minimum (g/dL)",
      PLT_min          = "Platelet Count, minimum (×10⁹/L)",
      AnionGap_max     = "Anion Gap, maximum (mEq/L)",
      Glucose_max      = "Glucose, maximum (mg/dL)",
      Lactate_max      = "Lactate, maximum (mmol/L)",
      Vasopressor_any  = "Vasopressor Use (any)",
      Hemodialysis_any = "Hemodialysis Use (any)"
    ),
    digits = list(
      all_continuous() ~ 2,
      GCS_min ~ 0, GCS_slope ~ 2,
      MV_day_max ~ 1
    ),
    missing = "no"   # Unknown 行移除；高缺失率變數見腳注
  ) %>%
  add_p(test = list(all_continuous()  ~ "wilcox.test",
                    all_categorical() ~ "chisq.test")) %>%
  modify_caption(
    "**Table 3. Summary of 48-hour Pre-Extubation Window (Worst Values and Trends)**"
  ) %>%
  modify_footnote(
    update = everything() ~ "Values are median (Q1, Q3) or n (%). Trend (slope) estimated by simple linear regression over up to 12 time points (4-h intervals); computed only for patients with ≥3 non-missing observations. Negative FiO2/PEEP slope indicates ventilator weaning; positive PaO2/FiO2 slope indicates improving oxygenation. Variables with >50% missing data (arterial blood gas, tidal volume, lactate) reflect intermittent measurement practice; denominators vary accordingly."
  ) %>%
  bold_labels()

# ============================================================
# 第三步：顯示結果
# ============================================================
table1
table2
table3

# ============================================================
# 第四步：輸出 Word 檔（選用）
# ============================================================
# 需要 flextable 套件：install.packages("flextable")
# table1 %>% as_flex_table() %>% flextable::save_as_docx(path = file.path(RESULTS_DIR, "table1.docx"))
# table2 %>% as_flex_table() %>% flextable::save_as_docx(path = file.path(RESULTS_DIR, "table2.docx"))
# table3 %>% as_flex_table() %>% flextable::save_as_docx(path = file.path(RESULTS_DIR, "table3.docx"))

# ============================================================
# 第五步：臨床軌跡圖（12 個時間步）
# ============================================================
trajectory_vars <- c(
  "heart_rate", "resp_rate", "spo2", "FiO2",
  "PaO2_FiO2_Ratio", "PEEP", "TV_per_kg", "GCS", "Lactate"
)

var_labels <- c(
  heart_rate      = "Heart Rate (bpm)",
  resp_rate       = "Resp Rate (breaths/min)",
  spo2            = "SpO2 (%)",
  FiO2            = "FiO2",
  PaO2_FiO2_Ratio = "PaO2/FiO2 Ratio",
  PEEP            = "PEEP (cmH2O)",
  TV_per_kg       = "Tidal Volume/IBW (mL/kg)",
  GCS             = "GCS Score",
  Lactate         = "Lactate (mmol/L)"
)

df_traj <- raw_df %>%
  mutate(Group = factor(Extubation_failure, levels = c(0, 1),
                        labels = c("Success", "Failure"))) %>%
  select(stay_id, time_bin, Group, all_of(trajectory_vars)) %>%
  pivot_longer(cols = all_of(trajectory_vars),
               names_to = "variable", values_to = "value") %>%
  mutate(variable = recode(variable, !!!var_labels))

df_traj_summary <- df_traj %>%
  group_by(Group, time_bin, variable) %>%
  summarise(
    median_val = median(value, na.rm = TRUE),
    q25        = quantile(value, 0.25, na.rm = TRUE),
    q75        = quantile(value, 0.75, na.rm = TRUE),
    .groups    = "drop"
  )

p_traj <- ggplot(df_traj_summary,
                 aes(x = time_bin, y = median_val, color = Group, fill = Group)) +
  geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.12, color = NA) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  scale_x_continuous(
    breaks = seq(-52, -8, by = 8),
    labels = paste0(seq(-52, -8, by = 8), "h")
  ) +
  scale_color_manual(values = c("Success" = "#1565C0", "Failure" = "#C62828")) +
  scale_fill_manual(values  = c("Success" = "#1565C0", "Failure" = "#C62828")) +
  labs(
    title    = "Clinical Trajectory in the 48h Pre-Extubation Window",
    subtitle = "Median (IQR shaded) by extubation outcome",
    x        = "Time Relative to Extubation",
    y        = NULL,
    color    = "Outcome", fill = "Outcome"
  ) +
  theme_bw(base_size = 11) +
  theme(
    strip.background = element_rect(fill = "grey92"),
    strip.text       = element_text(face = "bold", size = 9),
    legend.position  = "bottom",
    axis.text.x      = element_text(angle = 30, hjust = 1)
  )

ggsave(
  file.path(RESULTS_DIR, "trajectory_plot.png"),
  plot = p_traj, width = 12, height = 10, dpi = 300
)

print(p_traj)
