import math
import torch
import torch.nn as nn

from .layers.conv_pool_dropout_layer import ConvPoolDropoutLayer
from .layers.fkan_layer import FKANLayer


class CNN_FKAN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 conv_dropout=0.1,
                 conv_channels=[32, 64, 64],
                 fkan_degree=4,
                 fkan_features=[16]):
        super().__init__()

        self.degree = fkan_degree

        feature_layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            feature_layers.append(ConvPoolDropoutLayer(in_channels,
                                                       out_channels,
                                                       kernel_size=7,
                                                       padding=3,
                                                       dropout=conv_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*feature_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        classifier_layers = []
        in_features = conv_channels[-1]
        
        for out_features in fkan_features:
            classifier_layers.extend([
                FKANLayer(in_features=in_features,
                          out_features=out_features,
                          degree=fkan_degree),
                nn.BatchNorm1d(out_features)
            ])
            in_features = out_features
            
        classifier_layers.append(
            FKANLayer(in_features=in_features,
                      out_features=num_classes,
                      degree=fkan_degree)
        )
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)

        for layer in self.classifier:
            if isinstance(layer, FKANLayer):
                x = torch.tanh(x) * (math.pi / self.degree)
                x = layer(x)
            else:
                x = layer(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)