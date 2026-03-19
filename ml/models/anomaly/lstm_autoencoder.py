from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


class GRUAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=n_features, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


def resolve_device(preferred: str = "auto") -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    num_epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "auto",
) -> list[float]:
    if x_train.size == 0:
        raise ValueError("x_train is empty; cannot train autoencoder.")

    run_device = resolve_device(device)
    model.to(run_device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history: list[float] = []
    for _ in range(num_epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for (xb,) in loader:
            xb = xb.to(run_device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            optimizer.step()

            batch_n = xb.shape[0]
            loss_sum += float(loss.item()) * batch_n
            seen += batch_n
        history.append(loss_sum / max(seen, 1))
    return history


def reconstruct(model: nn.Module, x: np.ndarray, device: str = "auto") -> np.ndarray:
    run_device = resolve_device(device)
    model.eval()
    model.to(run_device)
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=run_device)
        out = model(xt)
    return out.cpu().numpy()


def reconstruction_error(model: nn.Module, x: np.ndarray, device: str = "auto") -> np.ndarray:
    run_device = resolve_device(device)
    model.eval()
    model.to(run_device)
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=run_device)
        out = model(xt)
        err = torch.mean((out - xt) ** 2, dim=(1, 2))
    return err.cpu().numpy()
