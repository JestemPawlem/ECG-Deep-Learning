import torch
import torch.nn as nn


class ResnetLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=7,
                 stride=1,
                 drop_path=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.batch1 = nn.BatchNorm1d(in_channels)
        self.batch2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()

        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               bias=False)
        
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      padding=0,
                                      stride=stride,
                                      bias=False)
        
        self.drop_path_prob = drop_path

    def forward(self, x):
        out = self.batch1(x)
        out = self.activation(out)

        if isinstance(self.shortcut, nn.Conv1d):
            shortcut_x = self.shortcut(out)
        else:
            shortcut_x = self.shortcut(x)

        out = self.conv1(out)
        
        out = self.batch2(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.training and self.drop_path_prob > 0.0:
            keep_prob = 1.0 - self.drop_path_prob
            mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), keep_prob, device=x.device))
            out = (out / keep_prob) * mask

        return out + shortcut_x