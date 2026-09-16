import argparse

import pandas as pd
import xarray as xr

from . import config, evaluate
from .climatology import apply_climatology, compute_climatology
from .data_loading import load_train_atmos


def run_holdout():
    ds = load_train_atmos()
    tp = ds[config.TP_VAR]

    clim = compute_climatology(tp, cutoff=config.HOLDOUT_TRAIN_END)

    val = tp.sel(time=slice(config.HOLDOUT_VAL_START, config.HOLDOUT_VAL_END))
    pred = apply_climatology(clim, val.time).assign_coords(time=val.time).drop_vars("month")

    evaluate.report(pred, val, label="climatology-holdout")


def run_full():
    ds = load_train_atmos()
    tp = ds[config.TP_VAR]

    clim = compute_climatology(tp, cutoff=config.TRAIN_END)

    test_time = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    test_time_da = xr.DataArray(test_time, dims="time", coords={"time": test_time})

    pred = apply_climatology(clim, test_time_da).assign_coords(time=test_time_da).drop_vars("month")

    out_path = config.PROCESSED_DIR / "climatology_predictions_full.nc"
    pred.rename("tp_mm_day").to_dataset().to_netcdf(out_path)
    print(f"wrote {out_path} | shape={pred.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    args = ap.parse_args()

    if args.split == "holdout":
        run_holdout()
    else:
        run_full()
