import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * 45),
            nn.BatchNorm1d(128 * 45),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        c = F.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat([noise, c], dim=-1)

        x = self.fc(x)
        x = x.view(-1, 128, 45)

        signal = self.conv(x)
        return signal