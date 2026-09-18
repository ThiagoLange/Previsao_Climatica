"""CLI: predicts on features_test.parquet with a trained booster and writes submission.csv.

    uv run python -m src.predict --split full --output submissions/xgb_full.csv

Requires `processed/features_test.parquet` (from `build_dataset --split full`) and
a trained booster at `models/xgb_<split>.json` (from `train.py`).

Blends the model prediction with the `clima_alvo` climatology feature already present
in the parquet. Models trained with `--target-mode residual` have their anomaly added
back before blending. The real test set has no labels to retune against, so holdout
configuration is carried over as-is. Month-specific blending is used only when its
year-held-out validation beats the global blend; otherwise the validated fallback is
used. Predictions are clipped at zero before submission.
"""

import argparse
import json

import pandas as pd
import xgboost as xgb

from . import config
from .make_submission import build_submission
from .train import NON_FEATURE_COLS, apply_lag_alpha, apply_month_alpha, apply_region_alpha

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
    meta_path = config.MODELS_DIR / f"xgb_{split}_meta.json"
    target_mode = "raw"
    if meta_path.exists():
        target_mode = json.loads(meta_path.read_text()).get("target_mode", "raw")
    if target_mode == "residual":
        model_pred = model_pred + df["clima_alvo"].to_numpy()
    model_pred = model_pred.clip(min=0.0)

    region_path = config.MODELS_DIR / "blend_alpha_by_region.json"
    month_path = config.MODELS_DIR / "blend_alpha_by_month.json"
    lag_path = config.MODELS_DIR / "blend_alpha_by_lag.json"
    if region_path.exists() and json.loads(region_path.read_text()).get("enabled", False):
        region_config = json.loads(region_path.read_text())
        blended = apply_region_alpha(df, model_pred, region_config["by_lag_latband"], region_config["default"])
        print(f"blend: usando alpha por (lag, faixa lat) de {region_path}")
    elif month_path.exists() and json.loads(month_path.read_text()).get("enabled", False):
        month_config = json.loads(month_path.read_text())
        alphas_by_month = {int(k): v for k, v in month_config["by_month"].items()}
        blended = apply_month_alpha(df, model_pred, alphas_by_month, month_config["default"])
        print(f"blend: usando alpha por mês de {month_path}")
    elif lag_path.exists():
        alpha_config = json.loads(lag_path.read_text())
        alphas_by_lag = {int(k): v for k, v in alpha_config["by_lag"].items()}
        blended = apply_lag_alpha(df, model_pred, alphas_by_lag, alpha_config["default"])
        print(f"blend: usando alpha por lag de {lag_path}")
    else:
        blended = alpha * model_pred + (1 - alpha) * df["clima_alvo"].values
        print(f"blend: nenhum json de alpha encontrado, usando alpha global={alpha}")

    return pd.DataFrame({"id": df["id"], "tp_mm_day": blended.clip(min=0.0)})


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
