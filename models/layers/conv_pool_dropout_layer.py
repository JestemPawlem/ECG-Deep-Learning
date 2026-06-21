import torch
import torch.nn as nn


class ConvPoolDropoutLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1,
                 dropout=0.0):
        super().__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout1d(dropout)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)

        return out