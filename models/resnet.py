import torch
import torch.nn as nn

from .layers.resnet_layer import ResnetLayer


class Resnet(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 conv_dropout=0.3,
                 conv_channels=[32, 64, 64],
                 fc_dropout=0.3,
                 fc_features=[64]):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=conv_channels[0],
                      kernel_size=7,
                      padding=3,
                      stride=2,
                      bias=False),
            nn.BatchNorm1d(conv_channels[0]),
            nn.SiLU()
        )

        resnet_layers = []
        in_channels = conv_channels[0]

        for out_channels in conv_channels[1:]:
            stride = 2 if in_channels != out_channels else 1
            resnet_layers.append(ResnetLayer(in_channels=in_channels,
                                             out_channels=out_channels,
                                             stride=stride,
                                             drop_path=conv_dropout))
            in_channels = out_channels
        
        self.resnet_block = nn.Sequential(*resnet_layers)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

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
    
    def forward(self, x):
        out = self.stem(x)
        out = self.resnet_block(out)

        out = self.global_avg_pool(out)
        out = self.flatten(out)

        logits = self.classifier(out)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)