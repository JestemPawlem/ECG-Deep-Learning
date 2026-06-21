import math
import torch
import torch.nn as nn


class FKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.register_buffer('freq', torch.arange(1, degree + 1).float())

        std = 1 / math.sqrt(in_features * degree * 2)
        self.a = nn.Parameter(torch.randn(out_features, in_features, degree) * std)
        self.b = nn.Parameter(torch.randn(out_features, in_features, degree) * std)
        
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)  
        kx = x_expanded * self.freq

        sin_kx = torch.sin(kx)
        cos_kx = torch.cos(kx)

        y_sin = torch.einsum('nik,oik->no', sin_kx, self.a) 
        y_cos = torch.einsum('nik,oik->no', cos_kx, self.b)

        return y_sin + y_cos + self.linear(x)