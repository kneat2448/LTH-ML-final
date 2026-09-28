"""Models: ResNet-20 for CIFAR-10 (main study) and Conv-4 (pilot reference only)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import set_seed


class BasicBlock(nn.Module):
    """He et al. 2016 basic block with an option-A (zero-padding, parameter-free) shortcut."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = stride != 1 or cin != cout
        self.extra = (cout - cin) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = x
        if self.downsample:  # option A: subsample spatially, zero-pad the new channels
            shortcut = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.extra, self.extra))
        return F.relu(out + shortcut)


class ResNet20(nn.Module):
    """CIFAR ResNet-20 (He et al. 2016, Sec. 4.2): 3 stages x 3 blocks, 16/32/64 channels.

    This is the network Frankle & Carbin (2019) call "Resnet-18"; ~0.27M parameters.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        widths, cin, stages = [16, 32, 64], 16, []
        for i, w in enumerate(widths):
            blocks = [BasicBlock(cin if j == 0 else w, w, 2 if (i > 0 and j == 0) else 1) for j in range(3)]
            stages.append(nn.Sequential(*blocks))
            cin = w
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():  # He init, as in the original CIFAR ResNet
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class Conv4(nn.Module):
    """Conv-4 without BatchNorm, identical to pilot/lt_sharp.py (Fashion-MNIST, 28x28x1)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 32, 3, padding=1)
        self.c4 = nn.Conv2d(32, 32, 3, padding=1)
        self.f1 = nn.Linear(32 * 7 * 7, 128)
        self.f2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.c2(F.relu(self.c1(x)))), 2)
        x = F.max_pool2d(F.relu(self.c4(F.relu(self.c3(x)))), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


MODELS = {"resnet20": ResNet20, "conv4": Conv4}


def init_model(name: str, seed: int, save_path: Path | None = None, device: str = "cpu") -> nn.Module:
    """Seeded initialization theta_0. If `save_path` is given, also save theta_0 there."""
    set_seed(seed)
    model = MODELS[name]().to(device).to(memory_format=torch.channels_last)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model
