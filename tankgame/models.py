"""Neural network trajectory predictors (PyTorch).

* RNNPredictorNet - GRU or LSTM encoder over the observed history followed by an
  MLP head that outputs the whole future (HORIZON x 2) in one shot.
* MLPPredictorNet - feed-forward baseline that sees the same flattened history,
  used to show what the recurrent memory adds.
"""

from __future__ import annotations

import torch
from torch import nn

from . import config as C


class RNNPredictorNet(nn.Module):
    def __init__(self, cell: str = "gru", in_dim: int = C.N_FEATURES, hidden: int = 128,
                 layers: int = 2, horizon: int = C.HORIZON, dropout: float = 0.1):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.cell = cell
        self.horizon = horizon
        self.rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, H, F)
        out, _ = self.rnn(x)
        z = torch.cat([out[:, -1], x[:, -1]], dim=-1)
        return self.head(z).view(x.shape[0], self.horizon, 2)


class MLPPredictorNet(nn.Module):
    def __init__(self, in_dim: int = C.N_FEATURES, history: int = C.HISTORY,
                 horizon: int = C.HORIZON, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim * history, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], self.horizon, 2)


def build_model(kind: str, **kw) -> nn.Module:
    if kind in ("gru", "lstm"):
        return RNNPredictorNet(cell=kind, **kw)
    if kind == "mlp":
        return MLPPredictorNet(**kw)
    raise ValueError(f"unknown model kind {kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
