"""CLI: trains a LightGBM regressor on the parquet features from build_dataset.

    uv run python -m src.train --split holdout                    # CPU, train<=2020-12, val=2021-2022
    uv run python -m src.train --split holdout --device cuda       # GPU, needs a CUDA-enabled LightGBM build
    uv run python -m src.train --split full --device cuda --n-estimators 2000

GPU (--device gpu|cuda) requires a LightGBM binary built with GPU support; the
default `pip install lightgbm` / `uv sync` wheel is CPU-only. On the GPU
machine, rebuild it before using --device:

    pip uninstall -y lightgbm
    pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON   # --device cuda, needs CUDA toolkit
    # or, for the OpenCL backend instead:
    pip install lightgbm --config-settings=cmake.define.USE_GPU=ON    # --device gpu, needs OpenCL + Boost

With a plain CPU wheel, --device cuda/gpu fails at fit() with a "GPU Tree
Learner was not enabled" error — fall back to --device cpu.
"""

import argparse
import time

import lightgbm as lgb
import pandas as pd
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
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    if device != "cpu":
        params["device_type"] = device  # "gpu" (OpenCL) or "cuda"

    model = lgb.LGBMRegressor(**params)

    t0 = time.time()
    if split == "holdout":
        val_path = config.PROCESSED_DIR / "features_val_holdout.parquet"
        X_val, y_val, _ = load_xy(val_path)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        pred_val = model.predict(X_val)
        val_rmse = root_mean_squared_error(y_val, pred_val)
        print(f"[holdout] val RMSE = {val_rmse:.4f} mm/day | best_iteration={model.best_iteration_}")
    else:
        model.fit(X_train, y_train)

    print(f"trained on device={device} in {time.time() - t0:.1f}s | rows={len(X_train)}")

    out = config.MODELS_DIR / f"lgbm_{split}.txt"
    model.booster_.save_model(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    ap.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    train(args.split, args.device, args.n_estimators, args.learning_rate)


if __name__ == "__main__":
    main()
