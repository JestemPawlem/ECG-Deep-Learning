import math
import torch
import torch.nn as nn

from .layers.fkan_layer import FKANLayer


class FKAN(nn.Module):
    def __init__(self,
                 input_dim=3,
                 num_classes=5,
                 degree=1,
                 hidden_dims=[64, 64, 32, 32]):
        super().__init__()
        self.degree = degree
        self.flatten = nn.Flatten()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(FKANLayer(in_features=current_dim,
                                    out_features=hidden_dim,
                                    degree=degree))
            layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        
        self.fkan = nn.ModuleList(layers)
        self.classifier = FKANLayer(in_features=current_dim,
                                    out_features=num_classes,
                                    degree=degree)
    
    def forward(self, x):
        x = self.flatten(x)

        for i in range(0, len(self.fkan), 2):
            x = torch.tanh(x) * (math.pi / self.self.degree)

            x = self.fkan[i](x)
            x = self.fkan[i+1](x)
        
        x = torch.tanh(x) * (math.pi / self.self.degree)
        x = self.classifier(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)