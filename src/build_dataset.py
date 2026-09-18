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

class _FeatureParquetWriter:
    """Write several feature batches to one parquet without materializing them."""

    def __init__(self, path):
        self.path = path
        self._writer = None

    def write(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


STALE_LAGS = (2, 3, 6, 12, 18, 24)



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
    spatial_stride: int = 1,
    writer: _FeatureParquetWriter | None = None,
) -> int:
    """Streams natural pairs to parquet in yearly chunks instead of materializing the
    full 1940-cutoff history in pandas at once (spans 80+ years x 78.561 points x ~40
    cols with the spatial+ONI features, tens of GB as a single DataFrame -> OOMs even
    with the QuantileDMatrix DataIter). spatial_stride>1 additionally subsamples grid
    points AFTER build_features (so the 5x5 neighbor-mean features are still computed
    from the full-resolution grid) -- since lat/lon are model features, training on a
    coarser spatial subset still generalizes to the full grid at inference/holdout."""
    shifted = _shift_forward_1m(atmos_ds)
    tp_target = atmos_ds[config.TP_VAR].rename("tp_target_obs")

    combined = xr.merge([shifted, tp_target], join="inner")
    combined = combined.sel(time=slice(None, cutoff_end))

    times = combined.time.to_index()
    chunk_months = chunk_years * 12

    own_writer = writer is None
    writer = writer or _FeatureParquetWriter(out_path)
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

            if spatial_stride > 1:
                feat = feat.isel(lat=slice(None, None, spatial_stride), lon=slice(None, None, spatial_stride))
                target = target.isel(lat=slice(None, None, spatial_stride), lon=slice(None, None, spatial_stride))

            df = flatten(feat, target=target, with_id=False)

            writer.write(df)

            total_rows += len(df)
            print(f"  {chunk_times[0].date()}..{chunk_times[-1].date()} rows={len(df)}")
    finally:
        if own_writer:
            writer.close()

    return total_rows


def build_stale_pairs(
    atmos_ds: xr.Dataset,
    cutoff_end: str,
    clim_tp: xr.DataArray,
    clim_atmos: dict[str, xr.DataArray],
    writer: _FeatureParquetWriter,
    spatial_stride: int = 2,
    origin_step_months: int = 3,
    stale_lags: tuple[int, ...] = STALE_LAGS,
    origins_per_chunk: int = 8,
) -> int:
    """Add historical pseudo-forecasts with a frozen last precipitation observation."""
    stale_lags = tuple(sorted(set(int(l) for l in stale_lags if int(l) >= 2)))
    if not stale_lags:
        return 0
    if origin_step_months < 1:
        raise ValueError("origin_step_months must be >= 1")

    first_time = pd.Timestamp(atmos_ds.time.values[0]).to_period("M").to_timestamp()
    last_origin = pd.Timestamp(cutoff_end) - pd.DateOffset(months=max(stale_lags))
    origins = pd.date_range(first_time, last_origin, freq=f"{origin_step_months}MS")

    total_rows = 0
    for start in range(0, len(origins), origins_per_chunk):
        origin_chunk = origins[start : start + origins_per_chunk]
        target_times = []
        origin_for_target = []
        for origin in origin_chunk:
            for lag in stale_lags:
                target = origin + pd.DateOffset(months=lag)
                if target <= pd.Timestamp(cutoff_end):
                    target_times.append(target)
                    origin_for_target.append(origin)

        if not target_times:
            continue

        target_times = pd.DatetimeIndex(target_times)
        base_times = target_times - pd.DateOffset(months=1)
        base = atmos_ds.sel(time=base_times).assign_coords(time=target_times)
        target = atmos_ds[config.TP_VAR].sel(time=target_times)

        origin_indexer = xr.DataArray(pd.DatetimeIndex(origin_for_target), dims="time")
        tp_ultima_obs = atmos_ds[config.TP_VAR].sel(time=origin_indexer)
        tp_ultima_obs = tp_ultima_obs.assign_coords(time=target_times)
        tp_ultima_obs_time = xr.DataArray(
            pd.DatetimeIndex(origin_for_target), dims="time", coords={"time": target_times}
        )

        feat = build_features(base, tp_ultima_obs, tp_ultima_obs_time, clim_tp, clim_atmos)
        if spatial_stride > 1:
            feat = feat.isel(lat=slice(None, None, spatial_stride), lon=slice(None, None, spatial_stride))
            target = target.isel(lat=slice(None, None, spatial_stride), lon=slice(None, None, spatial_stride))

        df = flatten(feat, target=target, with_id=False)
        writer.write(df)
        total_rows += len(df)
        print(
            f"  stale origins {origin_chunk[0].date()}..{origin_chunk[-1].date()} "
            f"lags={stale_lags} rows={len(df)}"
        )

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
    ap.add_argument(
        "--spatial-stride",
        type=int,
        default=1,
        help="subsample natural training grid points every N; does not affect val/test resolution",
    )
    ap.add_argument(
        "--stale-spatial-stride",
        type=int,
        default=2,
        help="grid stride for frozen-observation augmentation; natural pairs stay at --spatial-stride",
    )
    ap.add_argument(
        "--stale-origin-step",
        type=int,
        default=3,
        help="months between historical pseudo-forecast origins",
    )
    ap.add_argument(
        "--stale-lags",
        default=",".join(str(x) for x in STALE_LAGS),
        help="comma-separated frozen-observation lags to add",
    )
    args = ap.parse_args()
    stale_lags = tuple(int(x) for x in args.stale_lags.split(",") if x.strip())

    atmos_ds = load_train_atmos()

    if args.split == "holdout":
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.HOLDOUT_TRAIN_END)
        _save_climatology(clim_tp, clim_atmos, config.PROCESSED_DIR / "climatology_holdout.nc")

        out = config.PROCESSED_DIR / "features_train_holdout.parquet"
        writer = _FeatureParquetWriter(out)
        try:
            n_rows = build_natural_pairs(
                atmos_ds, config.HOLDOUT_TRAIN_END, clim_tp, clim_atmos, out,
                spatial_stride=args.spatial_stride, writer=writer
            )
            n_rows += build_stale_pairs(
                atmos_ds, config.HOLDOUT_TRAIN_END, clim_tp, clim_atmos, writer,
                spatial_stride=args.stale_spatial_stride,
                origin_step_months=args.stale_origin_step,
                stale_lags=stale_lags,
            )
        finally:
            writer.close()
        print(f"wrote {out} | rows={n_rows}")

        df_val = build_holdout_val(atmos_ds, clim_tp, clim_atmos)
        out = config.PROCESSED_DIR / "features_val_holdout.parquet"
        df_val.to_parquet(out, index=False)
        print(f"wrote {out} | rows={len(df_val)}")

    else:
        clim_tp, clim_atmos = _all_climatologies(atmos_ds, config.TRAIN_END)
        _save_climatology(clim_tp, clim_atmos, config.PROCESSED_DIR / "climatology_full.nc")

        out = config.PROCESSED_DIR / "features_train_full.parquet"
        writer = _FeatureParquetWriter(out)
        try:
            n_rows = build_natural_pairs(
                atmos_ds, config.TRAIN_END, clim_tp, clim_atmos, out,
                spatial_stride=args.spatial_stride, writer=writer
            )
            n_rows += build_stale_pairs(
                atmos_ds, config.TRAIN_END, clim_tp, clim_atmos, writer,
                spatial_stride=args.stale_spatial_stride,
                origin_step_months=args.stale_origin_step,
                stale_lags=stale_lags,
            )
        finally:
            writer.close()
        print(f"wrote {out} | rows={n_rows}")

        df_test = build_test(clim_tp, clim_atmos)
        out = config.PROCESSED_DIR / "features_test.parquet"
        df_test.to_parquet(out, index=False)
        print(f"wrote {out} | rows={len(df_test)}")


if __name__ == "__main__":
    main()
