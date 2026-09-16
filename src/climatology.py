import xarray as xr


def compute_climatology(da: xr.DataArray, cutoff: str | None = None) -> xr.DataArray:
    """Mean of da grouped by calendar month, using only data with time <= cutoff.

    Returns a DataArray indexed by (month, lat, lon), month = 1..12.
    """
    if cutoff is not None:
        da = da.sel(time=slice(None, cutoff))
    clim = da.groupby("time.month").mean("time", skipna=True)
    return clim


def apply_climatology(clim: xr.DataArray, months: xr.DataArray) -> xr.DataArray:
    """Broadcast climatology (month, lat, lon) onto an arbitrary time axis given its calendar months."""
    return clim.sel(month=months.dt.month)


def climatology_for_months(clim: xr.DataArray, month_of_year: xr.DataArray) -> xr.DataArray:
    """Vectorized lookup: month_of_year is an int array (any dims, values 1..12).
    Returns clim indexed pointwise along those dims, keeping (lat, lon)."""
    return clim.sel(month=month_of_year)
