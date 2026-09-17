"""CLI: trains the residual CNN on image tensors from build_image_dataset.

    uv run python -m src.train_cnn --split holdout --device cuda
    uv run python -m src.train_cnn --split full --device cuda --epochs 80

Each sample is a full grid (40, 301, 261) so batches are cheap on an 8GB GPU (batch=8 is
~30MB) -- unlike the tabular XGBoost path, there's no memory fight here.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .cnn_model import PrecipCNN


def _load_channel_names(prefix: str) -> list[str]:
    return json.loads((config.PROCESSED_DIR / f"{prefix}_channels.json").read_text())


def train(split: str, device: str, epochs: int, batch_size: int, lr: float, patience: int = 20) -> None:
    X = np.load(config.PROCESSED_DIR / f"images_train_{split}_X.npy", mmap_mode="r")
    y = np.load(config.PROCESSED_DIR / f"images_train_{split}_y.npy", mmap_mode="r")
    channel_names = _load_channel_names(f"images_train_{split}")
    clima_idx = channel_names.index("clima_alvo")

    X_t = torch.from_numpy(np.asarray(X))
    y_t = torch.from_numpy(np.asarray(y)).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model = PrecipCNN(in_channels=X.shape[1], clima_channel_idx=clima_idx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber: more robust to heavy-tailed precip outliers than MSE

    has_val = split == "holdout"
    if has_val:
        X_val = np.load(config.PROCESSED_DIR / "images_val_holdout_X.npy")
        y_val = np.load(config.PROCESSED_DIR / "images_val_holdout_y.npy")
        Xv_t = torch.from_numpy(X_val).to(device)
        yv_t = torch.from_numpy(y_val).to(device)

    best_rmse, bad = float("inf"), 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(loader.dataset)

        if has_val:
            model.eval()
            with torch.no_grad():
                pred_v = model(Xv_t).squeeze(1)
                rmse = torch.sqrt(torch.mean((pred_v - yv_t) ** 2)).item()
            print(f"epoch {epoch:03d} train-mse={train_loss:.4f} val-rmse={rmse:.4f}")

            if rmse < best_rmse:
                best_rmse, bad = rmse, 0
                torch.save(model.state_dict(), config.MODELS_DIR / f"cnn_{split}_best.pt")
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stop at epoch {epoch}, best val-rmse={best_rmse:.4f}")
                    break
        else:
            print(f"epoch {epoch:03d} train-mse={train_loss:.4f}")

    out_path = config.MODELS_DIR / f"cnn_{split}.pt"
    torch.save(model.state_dict(), out_path)
    torch.save({"channel_names": channel_names, "in_channels": X.shape[1], "clima_idx": clima_idx}, config.MODELS_DIR / f"cnn_{split}_meta.pt")
    print(f"wrote {out_path}")
    print(f"trained on device={device} in {time.time() - t0:.1f}s")

    if has_val:
        print(f"[holdout] best val-rmse = {best_rmse:.4f} mm/day")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["holdout", "full"], required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    train(args.split, args.device, args.epochs, args.batch_size, args.lr, args.patience)


if __name__ == "__main__":
    main()
