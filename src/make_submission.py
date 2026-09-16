"""Turn a (time, lat, lon) prediction array into a valid submission.csv.

Never reconstructs `id` from scratch for the final file: always merges predictions
onto sample_submission.csv by `id`, so row order/count/formatting always matches
exactly what Kaggle expects.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from . import config


def _clean_coord(x: float) -> float:
    # avoid "-0.00" from floating point noise around zero
    return 0.0 if abs(x) < 1e-8 else float(x)


def predictions_to_long_df(da: xr.DataArray, value_name: str = "tp_mm_day") -> pd.DataFrame:
    """da: DataArray with dims (time, lat, lon). Returns long df with columns id, value_name."""
    df = da.to_dataframe(name=value_name).reset_index()
    years = df["time"].dt.year.astype(str)
    months = df["time"].dt.month.astype(str).str.zfill(2)
    lat_str = df["lat"].map(_clean_coord).map(lambda v: f"{v:.2f}")
    lon_str = df["lon"].map(_clean_coord).map(lambda v: f"{v:.2f}")
    df["id"] = years + "_" + months + "_" + lat_str + "_" + lon_str
    return df[["id", value_name]]


def build_submission(pred_df: pd.DataFrame, out_path, value_name: str = "tp_mm_day") -> pd.DataFrame:
    sample = pd.read_csv(config.DATA_DIR / config.SAMPLE_SUBMISSION_FILE)
    assert "id" in sample.columns and value_name in sample.columns

    merged = sample[["id"]].merge(pred_df, on="id", how="left")

    n_missing = merged[value_name].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing} ids in sample_submission had no matching prediction")

    n_negative = (merged[value_name] < 0).sum()
    merged[value_name] = merged[value_name].clip(lower=0)

    assert len(merged) == len(sample) == config.N_GRID_POINTS * 24, (
        f"row count mismatch: {len(merged)} vs expected {config.N_GRID_POINTS * 24}"
    )
    assert (merged["id"].values == sample["id"].values).all(), "id order diverged from sample_submission"
    assert merged[value_name].isna().sum() == 0
    assert merged[value_name].min() >= 0

    merged.to_csv(out_path, index=False)
    print(f"wrote {out_path} | rows={len(merged)} | negatives clipped={n_negative} | "
          f"tp_mm_day range=({merged[value_name].min():.3f}, {merged[value_name].max():.3f})")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="netcdf file with (time,lat,lon) predictions")
    ap.add_argument("--var", default="tp_mm_day", help="variable name inside the input netcdf")
    ap.add_argument("--output", required=True, help="output csv path")
    args = ap.parse_args()

    da = xr.open_dataset(args.input)[args.var]
    pred_df = predictions_to_long_df(da, value_name="tp_mm_day")
    build_submission(pred_df, args.output)


if __name__ == "__main__":
    main()
