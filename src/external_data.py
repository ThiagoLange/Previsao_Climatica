"""Loads public climate indices as causal external predictors.

The original ONI file is kept for compatibility. The causal indices below are
monthly CPC products and are evaluated at the last observed month (target - 1),
so they do not use a seasonal product that can include the target month.
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import config

_SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

ONI_PATH = config.ROOT / "external" / "oni_monthly.txt"
NINO34_PATH = config.ROOT / "external" / "nino34_monthly.txt"
SOI_PATH = config.ROOT / "external" / "soi_monthly.txt"

CAUSAL_INDEX_URLS = {
    "nino34_monthly.txt": "https://www.cpc.ncep.noaa.gov/data/indices/Rnino34.ascii.txt",
    "soi_monthly.txt": "https://www.cpc.ncep.noaa.gov/data/indices/soi",
}


def _load_oni_table() -> dict[tuple[int, int], float]:
    df = pd.read_csv(ONI_PATH, sep=r"\s+")
    df["month"] = df["SEAS"].map(_SEASON_TO_MONTH)
    return dict(zip(zip(df["YR"], df["month"]), df["ANOM"]))


_ONI = _load_oni_table()


def _load_long_monthly_table(path: Path) -> dict[tuple[int, int], float]:
    """Load CPC's ``YR MTH ANOM`` Niño-3.4 text table."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep=r"\s+", comment="#")
    if not {"YR", "MTH", "ANOM"}.issubset(df.columns):
        return {}
    out = {}
    for row in df.itertuples(index=False):
        try:
            year, month, value = int(row.YR), int(row.MTH), float(row.ANOM)
        except (TypeError, ValueError):
            continue
        if 1 <= month <= 12 and np.isfinite(value):
            out[(year, month)] = value
    return out


def _load_cpc_wide_monthly_table(path: Path) -> dict[tuple[int, int], float]:
    """Load the first ``YEAR JAN ... DEC`` table from a CPC index file."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.split()[:2] == ["YEAR", "JAN"]),
        None,
    )
    if header is None:
        return {}
    out = {}
    for line in lines[header + 1 :]:
        parts = line.split()
        if len(parts) < 13:
            if out:
                break
            continue
        try:
            year = int(parts[0])
            values = [float(x) for x in parts[1:13]]
        except ValueError:
            if out:
                break
            continue
        for month, value in enumerate(values, start=1):
            if np.isfinite(value) and abs(value) < 900:
                out[(year, month)] = value
    return out


_NINO34 = _load_long_monthly_table(NINO34_PATH)
_SOI = _load_cpc_wide_monthly_table(SOI_PATH)


def download_causal_indices(output_dir: str | Path | None = None) -> list[Path]:
    """Download CPC monthly indices used without target-month leakage."""
    output_dir = Path(output_dir or NINO34_PATH.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in CAUSAL_INDEX_URLS.items():
        path = output_dir / filename
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
        paths.append(path)
    return paths


def oni_features(times) -> tuple[np.ndarray, np.ndarray]:
    """Return the legacy seasonal ONI feature for compatibility."""
    times = pd.DatetimeIndex(times)
    oni = np.zeros(len(times), dtype="float32")
    available = np.zeros(len(times), dtype="float32")
    for i, t in enumerate(times):
        val = _ONI.get((t.year, t.month))
        if val is not None:
            oni[i] = val
            available[i] = 1.0
    return oni, available


def _causal_lookup_features(
    times: pd.DatetimeIndex, values: dict[tuple[int, int], float], prefix: str
) -> dict[str, np.ndarray]:
    current = np.zeros(len(times), dtype="float32")
    ma3 = np.zeros(len(times), dtype="float32")
    ma6 = np.zeros(len(times), dtype="float32")
    available = np.zeros(len(times), dtype="float32")
    for i, t in enumerate(times):
        months = [t - pd.DateOffset(months=k) for k in range(6)]
        vals = [values.get((m.year, m.month), np.nan) for m in months]
        if np.isfinite(vals[0]):
            current[i] = vals[0]
            available[i] = 1.0
        valid3 = np.asarray(vals[:3], dtype="float32")
        valid6 = np.asarray(vals, dtype="float32")
        if np.isfinite(valid3).any():
            ma3[i] = np.nanmean(valid3)
        if np.isfinite(valid6).any():
            ma6[i] = np.nanmean(valid6)
    return {
        prefix: current,
        f"{prefix}_ma3": ma3,
        f"{prefix}_ma6": ma6,
        f"{prefix}_available": available,
    }


def causal_index_features(times) -> dict[str, np.ndarray]:
    """Return monthly Niño-3.4 and SOI features using only data at ``times``."""
    times = pd.DatetimeIndex(times)
    features = _causal_lookup_features(times, _NINO34, "nino34")
    features.update(_causal_lookup_features(times, _SOI, "soi"))
    features["oni_causal_ma3"] = features["nino34_ma3"].copy()
    features["oni_causal_available"] = features["nino34_available"].copy()
    return features
