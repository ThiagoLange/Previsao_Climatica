from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

for d in (PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR):
    d.mkdir(exist_ok=True)

ATMOS_VARS = {
    "t2": "treino_t2.nc",
    "cloud_cover": "treino_cloud_cover.nc",
    "surface_pressure": "treino_surface_pressure.nc",
    "shum_850": "treino_shum_850.nc",
    "rel_hum_850": "treino_rel_hum_850.nc",
    "temperature_850": "treino_temperature_850.nc",
    "geopotential_850": "treino_geopotential_850.nc",
    "u_850": "treino_u_850.nc",
    "v_850": "treino_v_850.nc",
}

TP_FILE = "treino_tp.nc"
TP_ALVO_FILE = "treino_tp_alvo.nc"
TEST_FEATURES_FILE = "teste_features.nc"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

TP_VAR = "tp"
TP_ALVO_VAR = "tp_alvo"

TRAIN_START = "1940-01-01"
TRAIN_END = "2022-12-01"

HOLDOUT_TRAIN_END = "2020-12-01"
HOLDOUT_VAL_START = "2021-01-01"
HOLDOUT_VAL_END = "2022-12-01"

GRID_SHAPE = (301, 261)
N_GRID_POINTS = 301 * 261
