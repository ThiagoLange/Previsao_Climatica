"""CLI: trains an XGBoost regressor on the parquet features from build_dataset.

    uv run python -m src.train --split holdout                 # CPU
    uv run python -m src.train --split holdout --device cuda    # GPU, pip-installed xgboost already ships CUDA support
    uv run python -m src.train --split full --device cuda --n-estimators 2000

--device cuda needs an NVIDIA GPU + driver visible to WSL/Linux (check `nvidia-smi`).
No special build required: the standard `pip install xgboost` / `uv sync` wheel
bundles CUDA support, unlike LightGBM which needs a from-source GPU build.

The training parquet (features_train_<split>.parquet) spans 80+ years x 78.561 grid
points and does not fit in RAM as a single pandas DataFrame on a 16GB machine, so it
is streamed in row-group batches into an xgb.QuantileDMatrix via a DataIter instead
of `pd.read_parquet` + `XGBRegressor.fit`.
"""

import argparse
import time

import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

from . import config

TARGET = "tp_alvo_true"
NON_FEATURE_COLS = {"time", "id", TARGET}


def _feature_cols(path) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    return [c for c in names if c not in NON_FEATURE_COLS]


class ParquetBatchIter(xgb.DataIter):
    """Streams a parquet file's row groups as (X, y) batches for QuantileDMatrix,
    so training never needs the whole file resident in pandas at once."""

    def __init__(self, path, feature_cols: list[str], target_col: str, batch_rows: int = 4_000_000):
        self.path = path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_rows = batch_rows
        self._batches = None
        super().__init__()

    def reset(self) -> None:
        pf = pq.ParquetFile(self.path)
        self._batches = pf.iter_batches(batch_size=self.batch_rows, columns=[*self.feature_cols, self.target_col])

    def next(self, input_data) -> int:
        try:
            batch = next(self._batches)
        except StopIteration:
            return 0
        df = batch.to_pandas()
        input_data(data=df[self.feature_cols], label=df[self.target_col])
        return 1


def load_xy(path) -> tuple[pd.DataFrame, "pd.Series", pd.DataFrame]:
    """Small parquet (val/test, ~24 months) -> load fully in memory, unlike the train split."""
    df = pd.read_parquet(path)
    y = df[TARGET]
    X = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])
    return X, y, df


def report_holdout(df_val: pd.DataFrame, y_val: pd.Series, pred_val) -> float:
    overall = root_mean_squared_error(y_val, pred_val)
    print(f"[holdout] val RMSE overall = {overall:.4f} mm/day")

    sq_err = pd.Series((pred_val - y_val.values) ** 2, index=df_val.index)

    by_lag = sq_err.groupby(df_val["lag_meses"].astype(int)).mean().pow(0.5)
    for lag, val in by_lag.items():
        print(f"  [holdout] RMSE lag={lag:02d} = {val:.4f}")

    by_month = sq_err.groupby(df_val["time"].dt.month).mean().pow(0.5)
    for month, val in by_month.items():
        print(f"  [holdout] RMSE month={month:02d} = {val:.4f}")

    return overall


def train(
    split: str,
    device: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int = 5,
    min_child_weight: float = 5.0,
    reg_lambda: float = 5.0,
    reg_alpha: float = 0.5,
) -> None:
    train_path = config.PROCESSED_DIR / f"features_train_{split}.parquet"
    feature_cols = _feature_cols(train_path)

    it = ParquetBatchIter(train_path, feature_cols, TARGET)
    dtrain = xgb.QuantileDMatrix(it)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        eta=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        seed=42,
    )

    t0 = time.time()
    if split == "holdout":
        val_path = config.PROCESSED_DIR / "features_val_holdout.parquet"
        X_val, y_val, df_val = load_xy(val_path)
        dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=50,
        )
        pred_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        report_holdout(df_val, y_val, pred_val)
        print(f"[holdout] best_iteration={booster.best_iteration}")
    else:
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

    print(f"trained on device={device} in {time.time() - t0:.1f}s")

    out = config.MODELS_DIR / f"xgb_{split}.json"
    booster.save_model(str(out))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--min-child-weight", type=float, default=5.0)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--reg-alpha", type=float, default=0.5)
    args = ap.parse_args()

    train(
        args.split,
        args.device,
        args.n_estimators,
        args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
    )


if __name__ == "__main__":
    main()
