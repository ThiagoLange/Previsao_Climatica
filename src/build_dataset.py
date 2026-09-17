"""CLI: builds climatology + feature parquet files for every split used downstream.

    uv run python -m src.build_dataset --split holdout   # train<=2020-12, val=2021-2022
    uv run python -m src.build_dataset --split full       # train<=2022-12, test=2023-2024
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from . import config
from .climatology import compute_climatology
from .data_loading import load_test_features, load_train_atmos
from .features import build_features, flatten


def _all_climatologies(atmos_ds: xr.Dataset, cutoff: str) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    clim_tp = compute_climatology(atmos_ds[config.TP_VAR], cutoff=cutoff)
    clim_atmos = {name: compute_climatology(atmos_ds[name], cutoff=cutoff) for name in config.ATMOS_VARS}
    return clim_tp, clim_atmos


def _save_climatology(clim_tp: xr.DataArray, clim_atmos: dict[str, xr.DataArray], out_path) -> None:
    ds = xr.Dataset({config.TP_VAR: clim_tp, **clim_atmos})
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}")


def _shift_forward_1m(ds: xr.Dataset) -> xr.Dataset:
    """Relabels time so that it represents target month instead of observation month."""
    new_time = ds.time.to_index() + pd.DateOffset(months=1)
    return ds.assign_coords(time=new_time)


def build_natural_pairs(
    atmos_ds: xr.Dataset,
    cutoff_end: str,
    clim_tp: xr.DataArray,
    clim_atmos: dict[str, xr.DataArray],
    out_path,
    chunk_years: int = 5,
) -> int:
    """Streams natural pairs to parquet in yearly chunks instead of materializing the
    full 1940-cutoff history in pandas at once (spans 80+ years x 78.561 points x ~20
    cols, easily 10s of GB as a single DataFrame -> OOMs constrained machines)."""
    shifted = _shift_forward_1m(atmos_ds)
    tp_target = atmos_ds[config.TP_VAR].rename("tp_target_obs")

    combined = xr.merge([shifted, tp_target], join="inner")
    combined = combined.sel(time=slice(None, cutoff_end))

    times = combined.time.to_index()
    chunk_months = chunk_years * 12

    writer = None
    total_rows = 0
    try:
        for start in range(0, len(times), chunk_months):
            chunk_times = times[start : start + chunk_months]
            chunk = combined.sel(time=chunk_times)

            base = chunk.drop_vars("tp_target_obs")
            target = chunk["tp_target_obs"]

            tp_ultima_obs = base[config.TP_VAR]
            tp_ultima_obs_time = xr.DataArray(
                base.time.to_index() - pd.DateOffset(months=1), dims="time", coords={"time": base.time}
            )

            feat = build_features(base, tp_ultima_obs, tp_ultima_obs_time, clim_tp, clim_atmos)
            df = flatten(feat, target=target, with_id=False)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            total_rows += len(df)
            print(f"  {chunk_times[0].date()}..{chunk_times[-1].date()} rows={len(df)}")
    finally:
        if writer is not None:
            writer.close()

    return total_rows


def build_holdout_val(
    atmos_ds: xr.Dataset, clim_tp: xr.DataArray, clim_atmos: dict[str, xr.DataArray]
) -> pd.DataFrame:
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
    return flatten(feat, target=target, with_id=False)


def build_test(clim_tp: xr.DataArray, clim_atmos: dict[str, xr.DataArray]) -> pd.DataFrame:
    test_ds = load_test_features()
    base = test_ds[list(config.ATMOS_VARS.keys())]

    lag = test_ds["lag_meses"].values.astype(int)
    last_obs_times = [t - pd.DateOffset(months=int(k)) for t, k in zip(test_ds.time.values, lag)]
    tp_ultima_obs_time = xr.DataArray(
        pd.DatetimeIndex(last_obs_times), dims="time", coords={"time": test_ds.time}
    )

    feat = build_features(base, test_ds["tp_ultima_obs"], tp_ultima_obs_time, clim_tp, clim_atmos)
    return flatten(feat, target=None, with_id=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    args = ap.parse_args()

    atmos_ds = load_train_atmos()

    if args.split == "holdout":
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.HOLDOUT_TRAIN_END)
        _save_climatology(clim_tp, clim_atmos, config.PROCESSED_DIR / "climatology_holdout.nc")

        out = config.PROCESSED_DIR / "features_train_holdout.parquet"
        n_rows = build_natural_pairs(atmos_ds, config.HOLDOUT_TRAIN_END, clim_tp, clim_atmos, out)
        print(f"wrote {out} | rows={n_rows}")

        df_val = build_holdout_val(atmos_ds, clim_tp, clim_atmos)
        out = config.PROCESSED_DIR / "features_val_holdout.parquet"
        df_val.to_parquet(out, index=False)
        print(f"wrote {out} | rows={len(df_val)}")

    else:
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.TRAIN_END)
        _save_climatology(clim_tp, clim_atmos, config.PROCESSED_DIR / "climatology_full.nc")

        out = config.PROCESSED_DIR / "features_train_full.parquet"
        n_rows = build_natural_pairs(atmos_ds, config.TRAIN_END, clim_tp, clim_atmos, out)
        print(f"wrote {out} | rows={n_rows}")

        df_test = build_test(clim_tp, clim_atmos)
        out = config.PROCESSED_DIR / "features_test.parquet"
        df_test.to_parquet(out, index=False)
        print(f"wrote {out} | rows={len(df_test)}")


if __name__ == "__main__":
    main()
