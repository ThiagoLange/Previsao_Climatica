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
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

from . import config

TARGET = "tp_alvo_true"
NON_FEATURE_COLS = {"time", "id", "time_origem", TARGET}


def _nonnegative(values) -> np.ndarray:
    """Precipitation cannot be negative; keep validation/blending consistent with submission."""
    return np.maximum(np.asarray(values, dtype="float32"), 0.0)


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


def load_xy(path, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, "pd.Series", pd.DataFrame]:
    """Small parquet (val/test, ~24 months) -> load fully in memory, unlike the train split.
    feature_cols forces an exact column order (see train()) instead of trusting that this
    parquet's schema happens to match the training parquet's -- xarray doesn't guarantee
    consistent data_var ordering across differently-built Datasets."""
    df = pd.read_parquet(path)
    y = df[TARGET]
    if feature_cols is not None:
        X = df[feature_cols]
    else:
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


def search_blend_alpha(df_val: pd.DataFrame, y_val: pd.Series, pred_val) -> float:
    """Scans pred_final = alpha*model + (1-alpha)*climatology on the holdout, since the
    model roughly ties climatology overall but loses in low-variance months -- a convex
    blend can keep the model's gains where it wins without its noise where it doesn't."""
    y = y_val.values
    clima = _nonnegative(df_val["clima_alvo"].values)
    pred_val = _nonnegative(pred_val)

    rmse_model = root_mean_squared_error(y, pred_val)
    rmse_clima = root_mean_squared_error(y, clima)
    print(f"[blend] alpha=1.00 (modelo puro)      RMSE = {rmse_model:.4f}")
    print(f"[blend] alpha=0.00 (climatologia pura) RMSE = {rmse_clima:.4f}")

    best_alpha, best_rmse = 0.0, rmse_clima
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * pred_val + (1 - alpha) * clima
        rmse = root_mean_squared_error(y, blended)
        if rmse < best_rmse:
            best_alpha, best_rmse = alpha, rmse

    print(f"[blend] melhor alpha={best_alpha:.2f} RMSE = {best_rmse:.4f}")
    return best_alpha


def search_blend_alpha_by_lag(df_val: pd.DataFrame, y_val: pd.Series, pred_val, default_alpha: float) -> dict[int, float]:
    """Per-lag alpha instead of one global value: low lag (more atmospheric signal) and
    high lag (degrades toward pure climatology) likely want different blend weights."""
    y = y_val.values
    clima = _nonnegative(df_val["clima_alvo"].values)
    pred_val = _nonnegative(pred_val)
    lag = df_val["lag_meses"].astype(int).values

    alphas: dict[int, float] = {}
    print("[blend] alpha por lag:")
    for l in sorted(set(lag)):
        mask = lag == l
        y_l, pred_l, clima_l = y[mask], pred_val[mask], clima[mask]

        best_alpha, best_rmse = 1.0, root_mean_squared_error(y_l, pred_l)
        for alpha in np.arange(0.0, 1.01, 0.05):
            blended = alpha * pred_l + (1 - alpha) * clima_l
            rmse = root_mean_squared_error(y_l, blended)
            if rmse < best_rmse:
                best_alpha, best_rmse = alpha, rmse

        alphas[int(l)] = round(float(best_alpha), 2)
        print(f"  lag={l:02d} alpha={best_alpha:.2f} RMSE={best_rmse:.4f}")

    blended_all = apply_lag_alpha(df_val, pred_val, alphas, default_alpha)
    rmse_all = root_mean_squared_error(y, blended_all)
    print(f"[blend] alpha por lag (aplicado geral) RMSE = {rmse_all:.4f}")

    return alphas


def apply_lag_alpha(df: pd.DataFrame, model_pred, alphas_by_lag: dict[int, float], default_alpha: float):
    lag = df["lag_meses"].astype(int).values
    alpha_arr = np.array([alphas_by_lag.get(int(l), default_alpha) for l in lag])
    clima = df["clima_alvo"].values
    return alpha_arr * model_pred + (1 - alpha_arr) * clima


LAT_MIN, LAT_MAX, N_LAT_BINS = -60.0, 15.0, 3


def _lat_band(lat_values) -> np.ndarray:
    edges = np.linspace(LAT_MIN, LAT_MAX, N_LAT_BINS + 1)
    return np.clip(np.digitize(lat_values, edges) - 1, 0, N_LAT_BINS - 1)


def search_blend_alpha_by_region(
    df_val: pd.DataFrame, y_val: pd.Series, pred_val, default_alpha: float
) -> dict[str, float]:
    """Alpha per (lag, latitude band) instead of per-lag alone: Andes/Amazonia/Pampas have
    very different rainfall regimes, so how much to trust the model vs. climatology likely
    varies by region too, not just by how stale the last real observation is."""
    y = y_val.values
    clima = _nonnegative(df_val["clima_alvo"].values)
    pred_val = _nonnegative(pred_val)
    lag = df_val["lag_meses"].astype(int).values
    lat_band = _lat_band(df_val["lat"].values)

    groups = pd.DataFrame({"lag": lag, "lat_band": lat_band}).groupby(["lag", "lat_band"]).indices

    alphas: dict[tuple[int, int], float] = {}
    for (l, b), idx in groups.items():
        y_g, pred_g, clima_g = y[idx], pred_val[idx], clima[idx]
        best_alpha, best_rmse = 1.0, root_mean_squared_error(y_g, pred_g)
        for alpha in np.arange(0.0, 1.01, 0.05):
            blended = alpha * pred_g + (1 - alpha) * clima_g
            rmse = root_mean_squared_error(y_g, blended)
            if rmse < best_rmse:
                best_alpha, best_rmse = alpha, rmse
        alphas[(int(l), int(b))] = round(float(best_alpha), 2)

    blended_all = apply_region_alpha(df_val, pred_val, alphas, default_alpha)
    rmse_all = root_mean_squared_error(y, blended_all)
    print(f"[blend] alpha por (lag, faixa lat) (aplicado geral) RMSE = {rmse_all:.4f}")

    return {f"{l}_{b}": a for (l, b), a in alphas.items()}


def apply_region_alpha(df: pd.DataFrame, model_pred, alphas: dict, default_alpha: float, n_lag: int = 25):
    if alphas and isinstance(next(iter(alphas)), str):
        alphas = {tuple(int(x) for x in k.split("_")): v for k, v in alphas.items()}

    table = np.full((n_lag, N_LAT_BINS), default_alpha, dtype="float32")
    for (l, b), a in alphas.items():
        table[l, b] = a

    lag = df["lag_meses"].astype(int).values
    lat_band = _lat_band(df["lat"].values)
    alpha_arr = table[lag, lat_band]
    clima = df["clima_alvo"].values
    return alpha_arr * model_pred + (1 - alpha_arr) * clima


def cv_check_region_alpha(
    df_val: pd.DataFrame, y_val: pd.Series, pred_val, default_alpha: float
) -> tuple[float, float] | None:
    """Return (regional OOF RMSE, global-alpha OOF RMSE) before enabling regional blending."""
    years = df_val["time"].dt.year.values
    uniq_years = sorted(set(years))
    if len(uniq_years) < 2:
        print("[blend-cv] holdout tem so 1 ano, pulando checagem cruzada")
        return None

    mid = len(uniq_years) // 2
    fold_defs = [set(uniq_years[:mid]), set(uniq_years[mid:])]

    y = y_val.values
    pred_val = _nonnegative(pred_val)
    oof_region = []
    oof_global = []
    for test_years in fold_defs:
        test_mask = np.isin(years, list(test_years))
        train_mask = ~test_mask

        train_df = df_val[train_mask].reset_index(drop=True)
        train_y = y_val[train_mask].reset_index(drop=True)
        test_df = df_val[test_mask].reset_index(drop=True)

        fold_global_alpha = search_blend_alpha(train_df, train_y, pred_val[train_mask])
        global_test = _nonnegative(
            fold_global_alpha * pred_val[test_mask]
            + (1 - fold_global_alpha) * test_df["clima_alvo"].values
        )
        global_rmse = root_mean_squared_error(y[test_mask], global_test)

        alphas = search_blend_alpha_by_region(
            train_df, train_y, pred_val[train_mask], fold_global_alpha
        )
        regional_test = _nonnegative(
            apply_region_alpha(test_df, pred_val[test_mask], alphas, fold_global_alpha)
        )
        regional_rmse = root_mean_squared_error(y[test_mask], regional_test)

        oof_global.append(global_rmse)
        oof_region.append(regional_rmse)
        print(
            f"[blend-cv] tunado em {sorted(set(years) - test_years)}, "
            f"testado em {sorted(test_years)}: global={global_rmse:.4f} regional={regional_rmse:.4f}"
        )

    global_mean = float(np.mean(oof_global))
    region_mean = float(np.mean(oof_region))
    print(
        f"[blend-cv] OOF global={global_mean:.4f} regional={region_mean:.4f}; "
        "regional só será habilitado se vencer fora da amostra"
    )
    return region_mean, global_mean


def train(
    split: str,
    device: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int = 5,
    min_child_weight: float = 5.0,
    reg_lambda: float = 5.0,
    reg_alpha: float = 0.5,
    max_bin: int = 64,
) -> None:
    train_path = config.PROCESSED_DIR / f"features_train_{split}.parquet"
    feature_cols = _feature_cols(train_path)

    cols_path = config.MODELS_DIR / f"feature_cols_{split}.json"
    cols_path.write_text(json.dumps(feature_cols))

    it = ParquetBatchIter(train_path, feature_cols, TARGET)
    dtrain = xgb.QuantileDMatrix(it, max_bin=max_bin)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        eta=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        max_bin=max_bin,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        seed=42,
    )

    t0 = time.time()
    if split == "holdout":
        val_path = config.PROCESSED_DIR / "features_val_holdout.parquet"
        X_val, y_val, df_val = load_xy(val_path, feature_cols)
        dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, max_bin=max_bin)

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=50,
        )
        pred_val = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        pred_val = _nonnegative(pred_val)
        report_holdout(df_val, y_val, pred_val)
        print(f"[holdout] best_iteration={booster.best_iteration}")
        global_alpha = search_blend_alpha(df_val, y_val, pred_val)
        alphas_by_lag = search_blend_alpha_by_lag(df_val, y_val, pred_val, global_alpha)

        alpha_path = config.MODELS_DIR / "blend_alpha_by_lag.json"
        alpha_path.write_text(json.dumps({"default": global_alpha, "by_lag": alphas_by_lag}, indent=2))
        print(f"wrote {alpha_path}")

        alphas_by_region = search_blend_alpha_by_region(df_val, y_val, pred_val, global_alpha)
        region_cv = cv_check_region_alpha(df_val, y_val, pred_val, global_alpha)
        region_enabled = bool(region_cv is not None and region_cv[0] < region_cv[1] - 1e-4)
        region_path = config.MODELS_DIR / "blend_alpha_by_region.json"
        region_path.write_text(
            json.dumps(
                {
                    "enabled": region_enabled,
                    "default": global_alpha,
                    "by_lag_latband": alphas_by_region,
                },
                indent=2,
            )
        )
        print(f"wrote {region_path} | enabled={region_enabled}")

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
    ap.add_argument("--max-bin", type=int, default=64, help="histogram resolution; lower = less GPU memory")
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
        max_bin=args.max_bin,
    )


if __name__ == "__main__":
    main()
