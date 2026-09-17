"""CLI: predicts on images_test_X.npy with a trained CNN and writes submission.csv.

    uv run python -m src.predict_cnn --output submissions/cnn_full.csv

Requires processed/images_test_X.npy (from build_image_dataset --split full) and
models/cnn_full.pt (from train_cnn.py --split full).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import config
from .cnn_model import PrecipCNN
from .make_submission import build_submission


def _clean_coord(x: float) -> float:
    return 0.0 if abs(x) < 1e-8 else float(x)


def predict(model_path=None, device: str = "cpu") -> pd.DataFrame:
    X = np.load(config.PROCESSED_DIR / "images_test_X.npy")
    times = json.loads((config.PROCESSED_DIR / "images_test_times.json").read_text())

    meta = torch.load(config.MODELS_DIR / "cnn_full_meta.pt", weights_only=True)
    model = PrecipCNN(in_channels=meta["in_channels"], clima_channel_idx=meta["clima_idx"]).to(device)
    model_path = model_path or config.MODELS_DIR / "cnn_full.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).squeeze(1).cpu().numpy()

    import xarray as xr

    grid = xr.open_dataset(config.DATA_DIR / config.TEST_FEATURES_FILE)
    lat_vals, lon_vals = grid.lat.values, grid.lon.values

    rows = []
    for ti, t in enumerate(times):
        year, month = t.split("-")
        for yi, lat_v in enumerate(lat_vals):
            lat_s = f"{_clean_coord(lat_v):.2f}"
            for xi, lon_v in enumerate(lon_vals):
                lon_s = f"{_clean_coord(lon_v):.2f}"
                rows.append((f"{year}_{month}_{lat_s}_{lon_s}", pred[ti, yi, xi]))

    return pd.DataFrame(rows, columns=["id", "tp_mm_day"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="override model path (default models/cnn_full.pt)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_df = predict(args.model, args.device)
    build_submission(pred_df, args.output)


if __name__ == "__main__":
    main()
