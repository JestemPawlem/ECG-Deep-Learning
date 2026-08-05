import math
import torch
import torch.nn as nn


class WavKANLayer(nn.Module):
    def __init__(self,
                 in_features=1000,
                 out_features=16,
                 kernel_size=31,
                 pool_size=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.scale = nn.Parameter(torch.rand(out_features, 1, 1) * 1.5 + 0.5)

        self.base_weights = nn.Parameter(torch.Tensor(out_features, in_features, kernel_size))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features, 1))

        nn.init.kaiming_uniform_(self.base_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))

        t = torch.linspace(-2, 2, kernel_size).view(1, 1, kernel_size)
        self.register_buffer('t', t)

        self.pool = nn.AdaptiveMaxPool1d(pool_size)

    def get_kernels(self):
        t_scaled = self.t / (torch.abs(self.scale) + 1e-8)
        t2 = t_scaled * t_scaled
        wavelet = torch.exp(-0.5 * t2) * torch.cos(5.0 * t_scaled)

        return wavelet * self.wavelet_weights

    def forward(self, x):

        kernel = self.base_weights + self.get_kernels()

        out = nn.functional.conv1d(x, kernel, padding=self.kernel_size // 2)
        pooled = self.pool(out)
        return pooled.flatten(start_dim=1)