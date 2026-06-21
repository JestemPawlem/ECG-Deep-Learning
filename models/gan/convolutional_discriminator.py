import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, input_dim=180, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Conv1d(1 + num_classes, 64, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(128 * 45, 1),
            nn.Sigmoid()
        )
    
    def forward(self, signal, labels):
        c = F.one_hot(labels, num_classes=self.num_classes).float()
        c = c.unsqueeze(-1).repeat(1, 1, self.input_dim)
        x = torch.cat([signal, c], dim=1)

        return self.model(x)