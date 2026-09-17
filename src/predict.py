"""CLI: predicts on features_test.parquet with a trained booster and writes submission.csv.

    uv run python -m src.predict --split full --output submissions/lgbm_full.csv

Requires `processed/features_test.parquet` (from `build_dataset --split full`) and
a trained booster at `models/lgbm_<split>.txt` (from `train.py`).
"""

import argparse

import lightgbm as lgb
import pandas as pd

from . import config
from .make_submission import build_submission
from .train import NON_FEATURE_COLS


def predict(split: str, model_path=None) -> pd.DataFrame:
    test_path = config.PROCESSED_DIR / "features_test.parquet"
    df = pd.read_parquet(test_path)

    model_path = model_path or config.MODELS_DIR / f"lgbm_{split}.txt"
    booster = lgb.Booster(model_file=str(model_path))

    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    pred = booster.predict(X)
    return pd.DataFrame({"id": df["id"], "tp_mm_day": pred})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], default="full")
    ap.add_argument("--model", help="override model path (default models/lgbm_<split>.txt)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_df = predict(args.split, args.model)
    build_submission(pred_df, args.output)


if __name__ == "__main__":
    main()
