"""EfficientNet-B0 backbone (ImageNet weights) with a small regression head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class AgeRegressor(nn.Module):
    def __init__(self, pretrained: bool = True, init_age: float = 33.0):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        # Start predictions around the mean age so early training is stable
        nn.init.constant_(self.head[-1].bias, init_age)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(-1)

    def freeze_backbone(self, frozen: bool = True):
        for p in self.features.parameters():
            p.requires_grad = not frozen


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
