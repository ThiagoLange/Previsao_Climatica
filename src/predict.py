"""CLI: predicts on features_test.parquet with a trained booster and writes submission.csv.

    uv run python -m src.predict --split full --output submissions/xgb_full.csv

Requires `processed/features_test.parquet` (from `build_dataset --split full`) and
a trained booster at `models/xgb_<split>.json` (from `train.py`).

Blends the model prediction with the `clima_alvo` climatology feature already present
in the parquet: `pred = alpha*model + (1-alpha)*clima_alvo`. Uses a per-lag_meses alpha
from `models/blend_alpha_by_lag.json` (written by `train.py --split holdout`) when
available, since low lag (more atmospheric signal) and high lag (degrades toward pure
climatology) want different blend weights. Falls back to a single --alpha (default
0.45, the holdout-tuned global value: RMSE 1.8733 vs 1.9095 pure model / 1.8938 pure
climatology) when the file is missing. The real test set has no labels to retune
against, so whatever was tuned on the holdout is carried over as-is.
"""

import argparse
import json

import pandas as pd
import xgboost as xgb

from . import config
from .make_submission import build_submission
from .train import NON_FEATURE_COLS, apply_lag_alpha

BLEND_ALPHA = 0.45


def predict(split: str, model_path=None, alpha: float = BLEND_ALPHA) -> pd.DataFrame:
    test_path = config.PROCESSED_DIR / "features_test.parquet"
    df = pd.read_parquet(test_path)

    model_path = model_path or config.MODELS_DIR / f"xgb_{split}.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    cols_path = config.MODELS_DIR / f"feature_cols_{split}.json"
    if cols_path.exists():
        feature_cols = json.loads(cols_path.read_text())
        X = df[feature_cols]
    else:
        X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    model_pred = booster.predict(xgb.DMatrix(X))

    alpha_path = config.MODELS_DIR / "blend_alpha_by_lag.json"
    if alpha_path.exists():
        alpha_config = json.loads(alpha_path.read_text())
        alphas_by_lag = {int(k): v for k, v in alpha_config["by_lag"].items()}
        blended = apply_lag_alpha(df, model_pred, alphas_by_lag, alpha_config["default"])
        print(f"blend: usando alpha por lag de {alpha_path}")
    else:
        blended = alpha * model_pred + (1 - alpha) * df["clima_alvo"].values
        print(f"blend: {alpha_path} nao encontrado, usando alpha global={alpha}")

    return pd.DataFrame({"id": df["id"], "tp_mm_day": blended})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], default="full")
    ap.add_argument("--model", help="override model path (default models/xgb_<split>.json)")
    ap.add_argument("--alpha", type=float, default=BLEND_ALPHA, help="fallback global weight on model pred vs climatology")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_df = predict(args.split, args.model, args.alpha)
    build_submission(pred_df, args.output)


if __name__ == "__main__":
    main()
