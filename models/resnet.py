"""ResNet -- exact copy of ldctbench/methods/resnet/network.py
(github.com/eeulig/ldct-benchmark, commit 09b1011bc2fb77ef4fc734cec1e961a20c754910).

No architectural changes. Plain nn.Module subclass without Namespace arg.
"""

import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, groups=8),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return self.layers(x) + x


class ResNet(nn.Module):
    """Residual Network (Park et al. 2017).

    Copied exactly from ldctbench/methods/resnet/network.py.
    10 residual blocks, 128 channels.
    NOISE SUBTRACTION: output = input - predicted_noise.
    """

    def __init__(self, n_channels: int = 128, n_blocks: int = 10):
        super().__init__()
        self.in_conv = nn.Conv2d(1, n_channels, 9, padding=4)
        self.blocks = nn.ModuleList([_ResBlock(n_channels) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(n_channels, 1, 3, padding=1)

    def forward(self, x):
        res = x
        out = self.in_conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.out_conv(out)
        return res - out

    @staticmethod
    def model_config():
        return {"n_channels": 128, "n_blocks": 10}
