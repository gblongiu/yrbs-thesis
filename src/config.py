"""Central configuration for the locked YRBS thesis pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
SPLITS_DIR = OUTPUTS_DIR / "splits"

RAW_FILE_2023 = RAW_DIR / "YRBS_2023_MH_subset.xlsx"

# Dataset and experiment identifiers (used in outputs/ metadata)
DATASET_VERSION = "yrbs_2023_modeling_v1"
EXPERIMENT_NAMESPACE = "week04_models_v1"

TARGET_PRIMARY = "QN26"
BULLYING_EXPOSURES = ["QN24", "QN25"]
SECONDARY_TARGETS = ["QN27", "QN28", "QN29", "QN30"]

# Dataset-build configuration (source-column names).
# Baseline model = demographic covariates only.
# Bullying-augmented model = baseline covariates plus QN24 and QN25.
BASELINE_COVARIATES = ["q1", "q2", "q3", "raceeth"]

# Survey design fields are used for weighted descriptive summaries only (not as model features).
SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]

# Modeling configuration (analysis-column names in the processed parquet).
TARGET_COL = "y_qn26"

EXPOSURE_COLS = ["x_qn24", "x_qn25"]
COVARIATE_COLS = ["q1", "q2", "q3", "raceeth"]
DESIGN_COLS = ["weight", "stratum", "psu"]

FEATURES_BASELINE = COVARIATE_COLS
FEATURES_FULL = COVARIATE_COLS + EXPOSURE_COLS

# Frozen validation protocol
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_SEEDS = [2026, 2027, 2028]
POSITIVE_LABEL = 1
MIN_GROUP_N = 200
MIN_GROUP_POS = 20
MIN_GROUP_NEG = 20
MIN_GROUP_EVENTRATE = None

# Post-hoc calibration defaults for the predictive pipeline.
CALIBRATION_FINAL_STRATEGY = "cv_stacking"  # choices: cv_stacking, train_split
CALIBRATION_HOLDOUT_SIZE = 0.2

# Week 6 defaults
PROB_BINS = 10
