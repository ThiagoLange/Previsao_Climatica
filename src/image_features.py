"""Converts a build_features() Dataset (mixed time-only and time/lat/lon data vars) into
an image tensor (time, channel, lat, lon), for the CNN. Reuses build_features so the CNN
sees the same predictors as the tabular XGBoost model, minus the hand-built spatial
neighbor-mean features (a CNN's convolutions already give it spatial context directly).
"""

import numpy as np
import xarray as xr


def dataset_to_tensor(feat: xr.Dataset) -> tuple[np.ndarray, list[str]]:
    """Returns (array, channel_names). array has shape (time, channel, lat, lon).
    Scalar-per-month variables (mes_sin, ano, oni, lag_meses, ...) are broadcast across
    the grid. Two extra channels (lat_norm, lon_norm) encode absolute position, since a
    CNN is translation-invariant by construction but rainfall regimes here are not
    (Andes vs Amazonia vs Pampas)."""
    names = list(feat.data_vars)
    full_dim_var = next(name for name in names if feat[name].dims == ("time", "lat", "lon"))
    reference = feat[full_dim_var]

    channels = [feat[name].broadcast_like(reference).transpose("time", "lat", "lon").values.astype("float32") for name in names]
    stacked = np.stack(channels, axis=1)  # (time, channel, lat, lon)

    lat = feat["lat"].values
    lon = feat["lon"].values
    n_time, _, h, w = stacked.shape

    lat2d, lon2d = np.meshgrid(lat / 90.0, lon / 180.0, indexing="ij")
    lat_channel = np.broadcast_to(lat2d.astype("float32"), (n_time, h, w))[:, None, :, :]
    lon_channel = np.broadcast_to(lon2d.astype("float32"), (n_time, h, w))[:, None, :, :]

    stacked = np.concatenate([stacked, lat_channel, lon_channel], axis=1)
    return stacked, [*names, "lat_norm", "lon_norm"]
