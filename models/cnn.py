import torch
import torch.nn as nn

from .layers.conv_pool_dropout_layer import ConvPoolDropoutLayer


class CNN(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 conv_dropout=0.1,
                 conv_channels=[32, 64, 128],
                 fc_dropout=0.3,
                 fc_features=[64]):
        super().__init__()

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

        classifier_layers = []
        in_features = conv_channels[-1]
        for out_features in fc_features:
            classifier_layers.extend([
                nn.Dropout(fc_dropout),
                nn.Linear(in_features, out_features),
                nn.SiLU()
            ])
            in_features = out_features
        classifier_layers.extend([
            nn.Dropout(fc_dropout),
            nn.Linear(in_features, num_classes)
        ])
        self.classifier = nn.Sequential(*classifier_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
