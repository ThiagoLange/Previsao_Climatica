import xarray as xr

from . import config


def load_train_atmos() -> xr.Dataset:
    """Merge all 9 atmospheric predictors + tp into one Dataset, time = month of observation."""
    das = {}
    for name, fname in config.ATMOS_VARS.items():
        ds = xr.open_dataset(config.DATA_DIR / fname)
        das[name] = ds[name]
    tp_ds = xr.open_dataset(config.DATA_DIR / config.TP_FILE)
    das[config.TP_VAR] = tp_ds[config.TP_VAR]

    merged = xr.Dataset(das)
    _validate_grid(merged)
    return merged


def load_train_target() -> xr.DataArray:
    """tp_alvo: precipitation of month M+1, indexed at month M. Last month is NaN."""
    ds = xr.open_dataset(config.DATA_DIR / config.TP_ALVO_FILE)
    return ds[config.TP_ALVO_VAR]


def load_test_features() -> xr.Dataset:
    ds = xr.open_dataset(config.DATA_DIR / config.TEST_FEATURES_FILE)
    _validate_grid(ds)
    return ds


def _validate_grid(ds: xr.Dataset) -> None:
    assert ds.sizes["lat"] == config.GRID_SHAPE[0], f"lat size {ds.sizes['lat']}"
    assert ds.sizes["lon"] == config.GRID_SHAPE[1], f"lon size {ds.sizes['lon']}"
    lat = ds.lat.values
    assert (lat[1:] > lat[:-1]).all(), "lat must be increasing"
