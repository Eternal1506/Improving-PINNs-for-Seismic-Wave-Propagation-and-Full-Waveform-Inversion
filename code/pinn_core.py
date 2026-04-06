import torch
import torch.nn as nn
import numpy as np


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    def __init__(self, n=10.0):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.tanh(self.n * self.a * x)


class FourierEncoding(nn.Module):
    def __init__(self, d_in: int, m: int, sigma: float, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(m, d_in, generator=rng) * sigma
        self.register_buffer('B', B)
        self.d_out = 2 * m

    def forward(self, x):
        proj = (2 * torch.pi * x) @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='tanh',
                 fourier=None, ub=None):
        super().__init__()
        self.fourier = fourier

        if ub is not None:
            self.register_buffer('ub', ub.float())
        else:
            self.ub = None

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(self._act_module(activation, i))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.net = nn.Sequential(*layers)

        self._init_weights(activation)

    def _act_module(self, name, layer_idx):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'sin':
            return SinActivation()
        elif name == 'adaptive':
            return AdaptiveActivation()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _init_weights(self, activation):
        gain = 1.0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def _normalise(self, x):
        if self.ub is not None:
            return 2.0 * (x / self.ub) - 1.0
        return x

    def forward(self, x):
        h = self._normalise(x)
        if self.fourier is not None:
            h = self.fourier(h)
        return self.net(h)
