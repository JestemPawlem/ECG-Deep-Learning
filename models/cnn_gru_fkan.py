import math
import torch
import torch.nn as nn

from .layers.conv_pool_dropout_layer import ConvPoolDropoutLayer
from .layers.fkan_layer import FKANLayer


class CNN_GRU_FKAN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 conv_dropout=0.1,
                 conv_channels=[16, 32, 64],
                 gru_dropout=0.3,
                 gru_hidden=64,
                 gru_layers=1,
                 gru_bidirectional=True,
                 fkan_degree=4,
                 fkan_features=[16]):
        super().__init__()

        self.gru_bidirectional = gru_bidirectional
        self.fkan_degree = fkan_degree

        feature_layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            feature_layers.append(ConvPoolDropoutLayer(in_channels,
                                                       out_channels,
                                                       kernel_size=9,
                                                       padding=4,
                                                       dropout=conv_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*feature_layers)

        self.gru_input_dropout = nn.Dropout(conv_dropout)

        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            bidirectional=gru_bidirectional,
            batch_first=True,
            dropout=gru_dropout
        )

        gru_out_dim = gru_hidden * 2 if gru_bidirectional else gru_hidden

        classifier_layers = []
        in_features = gru_out_dim

        for out_features in fkan_features:
            classifier_layers.extend([
                FKANLayer(in_features=in_features,
                          out_features=out_features,
                          degree=fkan_degree),
                nn.BatchNorm1d(out_features)
            ])
            in_features = out_features
        
        classifier_layers.extend([
            FKANLayer(in_features=in_features,
                      out_features=num_classes,
                      degree=fkan_degree)
        ])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        
        x = x.permute(0, 2, 1)
        x = self.gru_input_dropout(x)
        
        _, h_n = self.gru(x)

        if self.gru_bidirectional:
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            out = h_n[-1, :, :]

        for layer in self.classifier:
            if isinstance(layer, FKANLayer):
                out = torch.tanh(out) * (math.pi / self.fkan_degree)
                out = layer(out)
            else:
                out = layer(out)
        
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)