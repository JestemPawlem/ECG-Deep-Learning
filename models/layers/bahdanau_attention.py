import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, ht):
        score = self.v(torch.tanh(self.Wh(ht)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = torch.sum(ht * attention_weights, dim=1)

        return context_vector