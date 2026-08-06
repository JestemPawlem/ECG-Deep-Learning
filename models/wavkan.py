import torch
import torch.nn as nn

from .layers.wavkan_layer import WavKANLayer


class WavKAN(nn.Module):
    def __init__(
        self,
        num_classes=4,
        input_channels=12,
        input_length=1000,
        wavelet_features=16,
        pool_size=8,
        kernel_size=31,
        fc_hidden=[64, 32],
        fc_dropout=0.3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.wavelet_features = wavelet_features

        self.wavkan = WavKANLayer(in_features=input_channels,
                                  out_features=wavelet_features,
                                  kernel_size=kernel_size,
                                  pool_size=pool_size)

        in_feat = wavelet_features * pool_size
        fc_layers = []

        for out_feat in fc_hidden:
            fc_layers.extend([
                    nn.Dropout(fc_dropout),
                    nn.Linear(in_feat, out_feat),
                    nn.BatchNorm1d(out_feat),
                    nn.SiLU(),
            ])
            in_feat = out_feat

        fc_layers.extend([
                nn.Dropout(fc_dropout),
                nn.Linear(in_feat, num_classes),
        ])

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        wav_out = self.wavkan(x)
        return self.classifier(wav_out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)