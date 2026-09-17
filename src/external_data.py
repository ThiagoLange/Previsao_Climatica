"""Loads the NOAA CPC Oceanic Nino Index (ONI) as an external predictor.

Public domain, freely downloadable, no login: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
Allowed under the competition rules (Section 6, "Dados Externos e Ferramentas") since it's
public and equally accessible to every participant at no cost.

ENSO phase (captured by ONI) is a well-documented driver of monthly/seasonal precipitation
anomalies over South America (e.g. drier north/wetter south during El Nino) that the
reanalysis-only atmospheric features don't carry directly.

The record starts in 1950; training data goes back to 1940, so months before 1950 get
oni=0.0 (neutral) with oni_available=0 so the model can tell the difference.
"""

import numpy as np
import pandas as pd

from . import config

_SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

ONI_PATH = config.ROOT / "external" / "oni_monthly.txt"


def _load_oni_table() -> dict[tuple[int, int], float]:
    df = pd.read_csv(ONI_PATH, sep=r"\s+")
    df["month"] = df["SEAS"].map(_SEASON_TO_MONTH)
    return dict(zip(zip(df["YR"], df["month"]), df["ANOM"]))


_ONI = _load_oni_table()


def oni_features(times) -> tuple[np.ndarray, np.ndarray]:
    """times: array-like of datetime64 (one entry per timestep, not per grid point).
    Returns (oni_anom, oni_available) float32 arrays of the same length."""
    times = pd.DatetimeIndex(times)
    oni = np.zeros(len(times), dtype="float32")
    available = np.zeros(len(times), dtype="float32")
    for i, t in enumerate(times):
        val = _ONI.get((t.year, t.month))
        if val is not None:
            oni[i] = val
            available[i] = 1.0
    return oni, available
