import torch
import torch.nn as nn

from .layers.conv_pool_dropout_layer import ConvPoolDropoutLayer
from .layers.bahdanau_attention import BahdanauAttention


class CNN_LSTM_Attention(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.2):
        super().__init__()

        self.features = nn.Sequential(
            ConvPoolDropoutLayer(12, 64),
            ConvPoolDropoutLayer(64, 64)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.attention = BahdanauAttention(hidden_dim=64 * 2)

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)

        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.attention(out)

        out = self.flatten(out)

        logits = self.classifier(out)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)