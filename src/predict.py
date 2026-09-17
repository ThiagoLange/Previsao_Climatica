"""CLI: predicts on features_test.parquet with a trained booster and writes submission.csv.

    uv run python -m src.predict --split full --output submissions/xgb_full.csv

Requires `processed/features_test.parquet` (from `build_dataset --split full`) and
a trained booster at `models/xgb_<split>.json` (from `train.py`).

Blends the model prediction with the `clima_alvo` climatology feature already present
in the parquet: `pred = alpha*model + (1-alpha)*clima_alvo`. alpha=0.45 was tuned on
the holdout split (train<=2020, val=2021-2022), where it beat both the pure model
(RMSE 1.9095) and pure climatology (1.8938) with RMSE 1.8733. The real test set has no
labels to retune alpha against, so this value is carried over as-is.
"""

import argparse

import pandas as pd
import xgboost as xgb

from . import config
from .make_submission import build_submission
from .train import NON_FEATURE_COLS

BLEND_ALPHA = 0.45


def predict(split: str, model_path=None, alpha: float = BLEND_ALPHA) -> pd.DataFrame:
    test_path = config.PROCESSED_DIR / "features_test.parquet"
    df = pd.read_parquet(test_path)

    model_path = model_path or config.MODELS_DIR / f"xgb_{split}.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    model_pred = booster.predict(xgb.DMatrix(X))
    blended = alpha * model_pred + (1 - alpha) * df["clima_alvo"].values
    return pd.DataFrame({"id": df["id"], "tp_mm_day": blended})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], default="full")
    ap.add_argument("--model", help="override model path (default models/xgb_<split>.json)")
    ap.add_argument("--alpha", type=float, default=BLEND_ALPHA, help="weight on model pred vs climatology")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_df = predict(args.split, args.model, args.alpha)
    build_submission(pred_df, args.output)


if __name__ == "__main__":
    main()
