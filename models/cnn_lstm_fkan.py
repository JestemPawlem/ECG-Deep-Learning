import math
import torch
import torch.nn as nn
from .layers.conv_pool_dropout_layer import ConvPoolDropoutLayer
from .layers.fkan_layer import FKANLayer

class CNN_LSTM_FKAN(nn.Module):
    def __init__(self, num_classes=5, degree=5):
        super().__init__()

        self.degree = degree

        self.features = nn.Sequential(
            ConvPoolDropoutLayer(1, 32),
            ConvPoolDropoutLayer(32, 64)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        self.norm = nn.BatchNorm1d(8)

        self.fkan1 = FKANLayer(in_features=128 * 2,
                               out_features=8,
                               degree=degree)
        self.fkan2 = FKANLayer(in_features=8,
                               out_features=num_classes,
                               degree=degree)

    def forward(self, x):
        x = self.features(x)

        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)

        out = out.permute(0, 2, 1)
        out = self.global_avg_pool(out)
        out = self.flatten(out)

        out = torch.tanh(out) * (math.pi / self.degree)
        out = self.fkan1(out)
        
        out = self.norm(out)
        out = torch.tanh(out) * (math.pi / self.degree)
        out = self.fkan2(out)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)