import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self,
                 num_classes=4,
                 input_channels=12,
                 gru_dropout=0.2,
                 gru_hidden=64,
                 gru_layers=1,
                 gru_bidirectional=True,
                 fc_dropout=0.3,
                 fc_features=[64]):
        super().__init__()

        self.gru_bidirectional = gru_bidirectional
        self.input_dropout = nn.Dropout1d(p=gru_dropout)

        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            bidirectional=gru_bidirectional,
            batch_first=True
        )

        gru_out_dim = gru_hidden * 2 if gru_bidirectional else gru_hidden

        classifier_layers = []
        in_features = gru_out_dim
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
        x = self.input_dropout(x)
        x = x.permute(0, 2, 1)
        
        _, h_n = self.gru(x)

        if self.gru_bidirectional:
            out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            out = h_n[-1, :, :]

        logits = self.classifier(out)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)