"""Builds the feature table shared by every model (LightGBM tabular or CNN-as-image).

One function, `build_features`, covers all three cases the plan calls for:
  - natural training pairs (obs month M -> target M+1, persistence always lag=1)
  - the holdout validation set (last real observation frozen, lag grows 1..24,
    mirroring the real test structure exactly)
  - the real test set (teste_features.nc already gives obs/lag/last-obs directly)

because in all three the predictors are always "atmospheric state at target-1",
and the only thing that differs is where `tp_ultima_obs` comes from and how stale it is.
"""

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .climatology import climatology_for_months
from .external_data import oni_features


def _prev_month(month: xr.DataArray) -> xr.DataArray:
    return ((month - 2) % 12) + 1


def _neighbor_mean(da: xr.DataArray, window: int = 5) -> xr.DataArray:
    """Spatial smoothing over a window x window box (0.25deg grid, so window=5 ~= 1.25deg),
    giving the model regional context (Andes/Amazonia correlation) it otherwise lacks when
    every grid point is featurized independently."""
    return da.rolling(lat=window, lon=window, center=True, min_periods=1).mean()


def build_features(
    base: xr.Dataset,
    tp_ultima_obs: xr.DataArray,
    tp_ultima_obs_time: xr.DataArray,
    clim_tp: xr.DataArray,
    clim_atmos: dict[str, xr.DataArray],
) -> xr.Dataset:
    """base: dataset with dim 'time' = TARGET month, holding the 9 atmospheric
    variables each valued at target-1 (the "mês anterior ao alvo").
    tp_ultima_obs_time: DataArray over 'time' with the timestamp of the last real
    tp observation available for that row (target-1 for natural pairs, a frozen
    date for the holdout, or target - lag_meses for the real test)."""
    target_month = base["time"].dt.month
    obs_month = _prev_month(target_month)
    last_obs_month = tp_ultima_obs_time.dt.month

    feat = xr.Dataset()
    feat["mes_sin"] = np.sin(2 * np.pi * target_month / 12)
    feat["mes_cos"] = np.cos(2 * np.pi * target_month / 12)
    feat["ano"] = base["time"].dt.year

    feat["clima_alvo"] = climatology_for_months(clim_tp, target_month)
    feat["clima_m1"] = climatology_for_months(clim_tp, obs_month)

    for name in config.ATMOS_VARS:
        feat[name] = base[name]
        feat[f"{name}_anom"] = base[name] - climatology_for_months(clim_atmos[name], obs_month)
        feat[f"{name}_nbr5"] = _neighbor_mean(base[name])

    feat["tp_ultima_obs"] = tp_ultima_obs
    clima_last_obs = climatology_for_months(clim_tp, last_obs_month)
    feat["tp_ultima_obs_anom"] = tp_ultima_obs - clima_last_obs
    feat["tp_ultima_obs_nbr5"] = _neighbor_mean(tp_ultima_obs)

    target_ym = target_month + 12 * base["time"].dt.year
    last_obs_ym = last_obs_month + 12 * tp_ultima_obs_time.dt.year
    feat["lag_meses"] = target_ym - last_obs_ym

    obs_actual_time = base["time"].to_index() - pd.DateOffset(months=1)
    oni, oni_available = oni_features(obs_actual_time)
    feat["oni"] = xr.DataArray(oni, dims="time", coords={"time": base["time"]})
    feat["oni_available"] = xr.DataArray(oni_available, dims="time", coords={"time": base["time"]})

    return feat


def flatten(ds: xr.Dataset, target: xr.DataArray | None = None, with_id: bool = False) -> "pd.DataFrame":
    import pandas as pd

    ds = ds.astype("float32")
    if target is not None:
        ds = ds.assign(tp_alvo_true=target.astype("float32"))

    df = ds.to_dataframe().reset_index()
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")

    if with_id:
        years = df["time"].dt.year.astype(str)
        months = df["time"].dt.month.astype(str).str.zfill(2)
        lat_str = df["lat"].map(lambda v: 0.0 if abs(v) < 1e-8 else v).map(lambda v: f"{v:.2f}")
        lon_str = df["lon"].map(lambda v: 0.0 if abs(v) < 1e-8 else v).map(lambda v: f"{v:.2f}")
        df["id"] = years + "_" + months + "_" + lat_str + "_" + lon_str

    return df
