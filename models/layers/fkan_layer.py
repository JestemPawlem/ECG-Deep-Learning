import math
import torch
import torch.nn as nn


class FKANLayer(nn.Module):
    def __init__(self,
                 in_features=12,
                 out_features=16,
                 kernel_size=31,
                 pool_size=8,
                 degree=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.degree = degree

        self.a_weights = nn.Parameter(torch.Tensor(out_features, in_features, degree))
        self.b_weights = nn.Parameter(torch.Tensor(out_features, in_features, degree))

        std_fourier = 1.0 / math.sqrt(in_features * degree * 2)
        nn.init.normal_(self.a_weights, mean=0.0, std=std_fourier)
        nn.init.normal_(self.b_weights, mean=0.0, std=std_fourier)

        t = torch.linspace(-math.pi, math.pi, kernel_size).view(1, 1, 1, kernel_size)
        self.register_buffer('t', t)

        freq = torch.arange(1, degree + 1, dtype=torch.float32).view(1, 1, degree, 1)
        self.register_buffer('freq', freq)

        self.pool = nn.AdaptiveMaxPool1d(pool_size)

    def get_kernels(self):
        kt = self.freq * self.t

        cos_kt = torch.cos(kt)
        sin_kt = torch.sin(kt)

        fourier_part = (self.a_weights.unsqueeze(-1) * cos_kt + self.b_weights.unsqueeze(-1) * sin_kt).sum(dim=2)
        return fourier_part

    def forward(self, x):
        kernel = self.get_kernels()
        out = nn.functional.conv1d(x, kernel, padding=self.kernel_size // 2)
        
        return self.pool(out).flatten(start_dim=1)