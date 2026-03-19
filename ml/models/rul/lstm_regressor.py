from __future__ import annotations

import torch
import torch.nn as nn


class RULLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
