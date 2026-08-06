import torch
import torch.nn as nn

from .layers.fkan_layer import FKANLayer


class FKAN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 input_length=1000,
                 fourier_features=16,
                 pool_size=8,
                 kernel_size=31,
                 degree=5,
                 fc_hidden=[64, 32],
                 fc_dropout=0.3):
        super().__init__()

        self.input_channels = input_channels
        self.fourier_features = fourier_features

        self.fkan = FKANLayer(in_features=input_channels,
                              out_features=fourier_features,
                              kernel_size=kernel_size,
                              pool_size=pool_size,
                              degree=degree)

        in_feat = fourier_features * pool_size
        fc_layers = []

        for out_feat in fc_hidden:
            fc_layers.extend([
                nn.Dropout(fc_dropout),
                nn.Linear(in_feat, out_feat),
                nn.BatchNorm1d(out_feat),
                nn.SiLU()
            ])
            in_feat = out_feat

        fc_layers.extend([
            nn.Dropout(fc_dropout),
            nn.Linear(in_feat, num_classes)
        ])

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        fourier_out = self.fkan(x)
        return self.classifier(fourier_out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)