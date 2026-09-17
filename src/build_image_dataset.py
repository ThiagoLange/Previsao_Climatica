"""CLI: builds (time, channel, lat, lon) image tensors for the CNN.

    uv run python -m src.build_image_dataset --split holdout
    uv run python -m src.build_image_dataset --split full

Reuses build_features (same predictors as the tabular XGBoost pipeline) but keeps one
sample per MONTH instead of one per grid point per month, so the training array here is
~1000 samples instead of ~76M rows -- fits comfortably in RAM without the streaming
tricks build_dataset.py needed. The natural-pairs training array is still written via a
disk-backed memmap (filled chunk by chunk) rather than concatenated in RAM, just to be
safe on this machine's RAM budget (~8-9GB for the full array is still non-trivial).
"""

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .build_dataset import _all_climatologies, _shift_forward_1m
from .data_loading import load_test_features, load_train_atmos
from .features import build_features
from .image_features import dataset_to_tensor


def _write_channel_names(prefix: str, names: list[str]) -> None:
    (config.PROCESSED_DIR / f"{prefix}_channels.json").write_text(json.dumps(names))


def build_natural_pairs_images(
    atmos_ds: xr.Dataset, cutoff_end: str, clim_tp: xr.DataArray, clim_atmos: dict, out_prefix: str, chunk_years: int = 5
) -> int:
    shifted = _shift_forward_1m(atmos_ds)
    tp_target = atmos_ds[config.TP_VAR].rename("tp_target_obs")
    combined = xr.merge([shifted, tp_target], join="inner").sel(time=slice(None, cutoff_end))

    times = combined.time.to_index()
    chunk_months = chunk_years * 12
    total = len(times)

    X_mm, y_mm, channel_names = None, None, None
    idx = 0
    for start in range(0, total, chunk_months):
        chunk_times = times[start : start + chunk_months]
        chunk = combined.sel(time=chunk_times)
        base = chunk.drop_vars("tp_target_obs")
        target = chunk["tp_target_obs"]

        tp_ultima_obs = base[config.TP_VAR]
        tp_ultima_obs_time = xr.DataArray(
            base.time.to_index() - pd.DateOffset(months=1), dims="time", coords={"time": base.time}
        )
        feat = build_features(base, tp_ultima_obs, tp_ultima_obs_time, clim_tp, clim_atmos)
        X, names = dataset_to_tensor(feat)
        y = target.transpose("time", "lat", "lon").values.astype("float32")

        if X_mm is None:
            channel_names = names
            n_time, c, h, w = total, X.shape[1], X.shape[2], X.shape[3]
            X_mm = np.lib.format.open_memmap(
                config.PROCESSED_DIR / f"{out_prefix}_X.npy", mode="w+", dtype="float32", shape=(n_time, c, h, w)
            )
            y_mm = np.lib.format.open_memmap(
                config.PROCESSED_DIR / f"{out_prefix}_y.npy", mode="w+", dtype="float32", shape=(n_time, h, w)
            )
            _write_channel_names(out_prefix, channel_names)

        n = X.shape[0]
        X_mm[idx : idx + n] = X
        y_mm[idx : idx + n] = y
        idx += n
        print(f"  {chunk_times[0].date()}..{chunk_times[-1].date()} shape={X.shape}")

    X_mm.flush()
    y_mm.flush()
    return idx


def build_holdout_val_images(atmos_ds: xr.Dataset, clim_tp: xr.DataArray, clim_atmos: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_times = pd.date_range(config.HOLDOUT_VAL_START, config.HOLDOUT_VAL_END, freq="MS")
    obs_times = target_times - pd.DateOffset(months=1)

    base = atmos_ds.sel(time=obs_times)
    base = base.assign_coords(time=target_times)

    frozen_time = pd.Timestamp(config.HOLDOUT_TRAIN_END)
    frozen_tp = atmos_ds[config.TP_VAR].sel(time=frozen_time, drop=True)
    tp_ultima_obs, _ = xr.broadcast(frozen_tp, base[config.TP_VAR])
    tp_ultima_obs_time = xr.DataArray(
        pd.DatetimeIndex([frozen_time] * len(target_times)), dims="time", coords={"time": base.time}
    )

    target = atmos_ds[config.TP_VAR].sel(time=target_times)

    feat = build_features(base, tp_ultima_obs, tp_ultima_obs_time, clim_tp, clim_atmos)
    X, names = dataset_to_tensor(feat)
    y = target.transpose("time", "lat", "lon").values.astype("float32")
    return X, y, names


def build_test_images(clim_tp: xr.DataArray, clim_atmos: dict) -> tuple[np.ndarray, list[str], list[str]]:
    test_ds = load_test_features()
    base = test_ds[list(config.ATMOS_VARS.keys())]

    lag = test_ds["lag_meses"].values.astype(int)
    last_obs_times = [t - pd.DateOffset(months=int(k)) for t, k in zip(test_ds.time.values, lag)]
    tp_ultima_obs_time = xr.DataArray(
        pd.DatetimeIndex(last_obs_times), dims="time", coords={"time": test_ds.time}
    )

    feat = build_features(base, test_ds["tp_ultima_obs"], tp_ultima_obs_time, clim_tp, clim_atmos)
    X, names = dataset_to_tensor(feat)

    time_index = test_ds.time.to_index()
    time_strs = [t.strftime("%Y-%m") for t in time_index]
    return X, names, time_strs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    args = ap.parse_args()

    atmos_ds = load_train_atmos()

    if args.split == "holdout":
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.HOLDOUT_TRAIN_END)

        n = build_natural_pairs_images(atmos_ds, config.HOLDOUT_TRAIN_END, clim_tp, clim_atmos, "images_train_holdout")
        print(f"wrote images_train_holdout | months={n}")

        X_val, y_val, names = build_holdout_val_images(atmos_ds, clim_tp, clim_atmos)
        np.save(config.PROCESSED_DIR / "images_val_holdout_X.npy", X_val)
        np.save(config.PROCESSED_DIR / "images_val_holdout_y.npy", y_val)
        _write_channel_names("images_val_holdout", names)
        print(f"wrote images_val_holdout | shape={X_val.shape}")

    else:
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.TRAIN_END)

        n = build_natural_pairs_images(atmos_ds, config.TRAIN_END, clim_tp, clim_atmos, "images_train_full")
        print(f"wrote images_train_full | months={n}")

        X_test, names, time_strs = build_test_images(clim_tp, clim_atmos)
        np.save(config.PROCESSED_DIR / "images_test_X.npy", X_test)
        _write_channel_names("images_test", names)
        (config.PROCESSED_DIR / "images_test_times.json").write_text(json.dumps(time_strs))
        print(f"wrote images_test | shape={X_test.shape}")


if __name__ == "__main__":
    main()
