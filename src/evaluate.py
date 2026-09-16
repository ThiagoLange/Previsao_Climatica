import numpy as np
import xarray as xr


def rmse(pred: xr.DataArray, truth: xr.DataArray) -> float:
    diff = (pred - truth) ** 2
    return float(np.sqrt(diff.mean().values))


def rmse_by_group(pred: xr.DataArray, truth: xr.DataArray, group: str) -> xr.DataArray:
    """True RMSE per group (e.g. 'time.year' or 'time.month'): squared error averaged
    over time-within-group and space, sqrt applied once at the end."""
    sq = (pred - truth) ** 2
    per_group = sq.groupby(group).mean(dim="time")
    return np.sqrt(per_group.mean(dim=["lat", "lon"]))


def report(pred: xr.DataArray, truth: xr.DataArray, label: str = "") -> float:
    overall = rmse(pred, truth)
    print(f"[{label}] RMSE overall = {overall:.4f} mm/day")

    by_year = rmse_by_group(pred, truth, "time.year")
    for year, val in zip(by_year["year"].values, by_year.values):
        print(f"  [{label}] RMSE {year} = {val:.4f}")

    by_month = rmse_by_group(pred, truth, "time.month")
    for month, val in zip(by_month["month"].values, by_month.values):
        print(f"  [{label}] RMSE month={month:02d} = {val:.4f}")

    return overall
