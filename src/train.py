"""CLI: trains an XGBoost regressor on the parquet features from build_dataset.

    uv run python -m src.train --split holdout                 # CPU
    uv run python -m src.train --split holdout --device cuda    # GPU, pip-installed xgboost already ships CUDA support
    uv run python -m src.train --split full --device cuda --n-estimators 2000

--device cuda needs an NVIDIA GPU + driver visible to WSL/Linux (check `nvidia-smi`).
No special build required: the standard `pip install xgboost` / `uv sync` wheel
bundles CUDA support, unlike LightGBM which needs a from-source GPU build.
"""

import argparse
import time

import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

from . import config

TARGET = "tp_alvo_true"
NON_FEATURE_COLS = {"time", "id", TARGET}


def load_xy(path) -> tuple[pd.DataFrame, "pd.Series", pd.DataFrame]:
    df = pd.read_parquet(path)
    y = df[TARGET]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    return X, y, df


def train(split: str, device: str, n_estimators: int, learning_rate: float) -> None:
    train_path = config.PROCESSED_DIR / f"features_train_{split}.parquet"
    X_train, y_train, _ = load_xy(train_path)

    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        random_state=42,
        eval_metric="rmse",
    )

    t0 = time.time()
    if split == "holdout":
        val_path = config.PROCESSED_DIR / "features_val_holdout.parquet"
        X_val, y_val, _ = load_xy(val_path)
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        pred_val = model.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, pred_val)
        print(f"[holdout] val RMSE = {val_rmse:.4f} mm/day | best_iteration={model.best_iteration}")
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

    print(f"trained on device={device} in {time.time() - t0:.1f}s | rows={len(X_train)}")

    out = config.MODELS_DIR / f"xgb_{split}.json"
    model.get_booster().save_model(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    train(args.split, args.device, args.n_estimators, args.learning_rate)


if __name__ == "__main__":
    main()
