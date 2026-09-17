"""Residual CNN: predicts climatology + a learned spatial correction, instead of the
raw precipitation value from scratch. With ~1000 training months and a domain where
climatology alone is already a strong baseline, this gives the network a sane starting
point (zero-initialized head -> output equals climatology on step zero) and lets it
focus capacity on the correction rather than relearning known seasonal/regional means.

No pooling/upsampling: the grid is 301x261 (not a nice power of 2), so plain dilated
convolutions (stride 1, padding=same) avoid shape-matching headaches while still growing
the receptive field to cover regional-scale patterns (Andes, Amazonia).
"""

import torch
import torch.nn as nn


class PrecipCNN(nn.Module):
    def __init__(self, in_channels: int, clima_channel_idx: int, hidden: int = 64, n_layers: int = 8):
        super().__init__()
        self.clima_idx = clima_channel_idx

        dilations = [1, 1, 2, 2, 4, 4, 2, 1]
        layers = []
        c_in = in_channels
        for i in range(n_layers):
            d = dilations[i % len(dilations)]
            layers += [
                nn.Conv2d(c_in, hidden, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
            c_in = hidden
        self.body = nn.Sequential(*layers)

        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clima = x[:, self.clima_idx : self.clima_idx + 1, :, :]
        correction = self.head(self.body(x))
        return clima + correction
