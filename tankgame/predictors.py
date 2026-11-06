"""Unified interface over all trajectory predictors (learned and physics baselines).

Every predictor receives the opponent history sampled at the model rate
(positions, headings, ray distances) and returns HORIZON future absolute
positions, one per MODEL_DT step.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from . import config as C
from .features import build_features, decode
from .models import build_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

LABELS = {
    "static": "Static (no prediction)",
    "cv": "Constant velocity",
    "ctrv": "Constant turn rate",
    "mlp": "MLP (feed-forward NN)",
    "lstm": "LSTM (RNN)",
    "gru": "GRU (RNN)",
}


def _clamp(pred: np.ndarray) -> np.ndarray:
    pred[..., 0] = np.clip(pred[..., 0], C.TANK_RADIUS, C.ARENA_W - C.TANK_RADIUS)
    pred[..., 1] = np.clip(pred[..., 1], C.TANK_RADIUS, C.ARENA_H - C.TANK_RADIUS)
    return pred


class Predictor:
    name = "base"
    min_history = 3

    @property
    def label(self) -> str:
        return LABELS.get(self.name, self.name)

    def _predict_batch(self, pos, head, rays) -> np.ndarray:
        raise NotImplementedError

    def predict_batch(self, pos: np.ndarray, head: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """pos (B,T,2), head (B,T), rays (B,T,3) with T >= min_history  ->  (B,HORIZON,2)."""
        pos = np.asarray(pos, dtype=np.float64)
        head = np.asarray(head, dtype=np.float64)
        rays = np.asarray(rays, dtype=np.float64)
        return _clamp(self._predict_batch(pos, head, rays))

    def predict(self, pos: np.ndarray, head: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """Single history (T,2), (T,), (T,3) of any length >= 1  ->  (HORIZON,2)."""
        pos = np.asarray(pos, dtype=np.float64)
        n = len(pos)
        if n < self.min_history:
            if n >= 3:
                return CVPredictor().predict(pos, head, rays)
            return np.repeat(pos[-1:], C.HORIZON, axis=0).copy()
        return self.predict_batch(pos[None], np.asarray(head)[None], np.asarray(rays)[None])[0]


class StaticPredictor(Predictor):
    """Assumes the target stays where it is (= aiming directly at it)."""
    name = "static"
    min_history = 1

    def _predict_batch(self, pos, head, rays):
        return np.repeat(pos[:, -1:, :], C.HORIZON, axis=1).copy()


class CVPredictor(Predictor):
    """Constant velocity extrapolation from the last two steps."""
    name = "cv"

    def _predict_batch(self, pos, head, rays):
        v = (pos[:, -1] - pos[:, -3]) / 2.0
        k = np.arange(1, C.HORIZON + 1, dtype=np.float64)[None, :, None]
        return pos[:, -1:, :] + v[:, None, :] * k


class CTRVPredictor(Predictor):
    """Constant turn rate and velocity model (classic tracking baseline)."""
    name = "ctrv"
    min_history = 4

    def _predict_batch(self, pos, head, rays):
        d = pos[:, -1] - pos[:, -3]
        speed = np.hypot(d[:, 0], d[:, 1]) / 2.0
        theta = np.arctan2(d[:, 1], d[:, 0])
        dh = np.angle(np.exp(1j * (head[:, -1] - head[:, -4])))  # wrapped difference
        omega = dh / 3.0
        k = np.arange(1, C.HORIZON + 1, dtype=np.float64)[None, :]
        # the chord p[-3]->p[-1] points along the tangent at t-1; step k follows the tangent at t+k-0.5
        ang = theta[:, None] + omega[:, None] * (k + 0.5)
        steps = np.stack([np.cos(ang), np.sin(ang)], axis=-1) * speed[:, None, None]
        return pos[:, -1:, :] + np.cumsum(steps, axis=1)


class NeuralPredictor(Predictor):
    """Wraps a trained GRU / LSTM / MLP checkpoint."""
    min_history = C.HISTORY + 1

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.name = ckpt["kind"]
        self.model = build_model(ckpt["kind"], **ckpt.get("model_kwargs", {}))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.metrics = ckpt.get("metrics", {})

    @torch.no_grad()
    def _predict_batch(self, pos, head, rays):
        h = C.HISTORY + 1
        pos_t = torch.as_tensor(pos[:, -h:], dtype=torch.float32, device=self.device)
        head_t = torch.as_tensor(head[:, -h:], dtype=torch.float32, device=self.device)
        rays_t = torch.as_tensor(rays[:, -h:], dtype=torch.float32, device=self.device)
        out = self.model(build_features(pos_t, head_t, rays_t))
        pred = decode(out, pos_t[:, -1], head_t[:, -1])
        return pred.cpu().numpy().astype(np.float64)


def load_predictor(name: str, device: str = "cpu", models_dir: str | Path = MODELS_DIR) -> Predictor:
    name = name.lower()
    if name == "static":
        return StaticPredictor()
    if name == "cv":
        return CVPredictor()
    if name == "ctrv":
        return CTRVPredictor()
    path = Path(models_dir) / f"{name}.pt"
    if not path.exists():
        warnings.warn(f"model {path} not found - falling back to the constant turn rate baseline")
        return CTRVPredictor()
    return NeuralPredictor(path, device=device)


AVAILABLE = tuple(LABELS)
