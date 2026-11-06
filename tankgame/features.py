"""Feature engineering shared by training and in-game inference.

Input of the network is the last HISTORY observations of the opponent
(HISTORY+1 raw samples are needed to compute HISTORY velocities). Everything
that describes motion is expressed in the *ego frame* of the opponent at the
last observed time step (x axis = its current heading). This makes the
problem rotation invariant: a left turn looks the same in every part of the
arena. Absolute position, absolute heading and three ray-sensor distances
are added so that the network can still anticipate walls and obstacles.

Per time step features (N_FEATURES = 11):
    0-1  velocity in the ego frame / STEP_SCALE
    2-3  cos/sin of heading relative to the last heading
    4-6  ray distances (front, front-left, front-right) / RAY_MAX
    7-8  absolute position scaled to [-1, 1]
    9-10 cos/sin of absolute heading

Output of the network is HORIZON per-step displacements in the same ego frame
divided by STEP_SCALE; `decode` integrates and rotates them back to world
coordinates.
"""

from __future__ import annotations

import torch

from . import config as C


def _rot(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, s: torch.Tensor):
    """Rotate vectors (x, y) by the angle whose cos/sin are c/s."""
    return c * x - s * y, s * x + c * y


def build_features(pos: torch.Tensor, head: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    """pos (B,H+1,2), head (B,H+1), rays (B,H+1,3)  ->  features (B,H,N_FEATURES)."""
    h_last = head[:, -1]
    c = torch.cos(h_last)[:, None]
    s = torch.sin(h_last)[:, None]
    vel = pos[:, 1:] - pos[:, :-1]
    vx, vy = _rot(vel[..., 0], vel[..., 1], c, -s)          # rotate by -h_last
    dh = head[:, 1:] - h_last[:, None]
    p = pos[:, 1:]
    h = head[:, 1:]
    feats = torch.stack([
        vx / C.STEP_SCALE,
        vy / C.STEP_SCALE,
        torch.cos(dh),
        torch.sin(dh),
        rays[:, 1:, 0] / C.RAY_MAX,
        rays[:, 1:, 1] / C.RAY_MAX,
        rays[:, 1:, 2] / C.RAY_MAX,
        p[..., 0] / C.ARENA_W * 2.0 - 1.0,
        p[..., 1] / C.ARENA_H * 2.0 - 1.0,
        torch.cos(h),
        torch.sin(h),
    ], dim=-1)
    return feats


def ego_targets(future: torch.Tensor, pos_last: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    """Future absolute positions (B,K,2) -> cumulative displacement in the ego frame (B,K,2)."""
    d = future - pos_last[:, None, :]
    c = torch.cos(h_last)[:, None]
    s = torch.sin(h_last)[:, None]
    ex, ey = _rot(d[..., 0], d[..., 1], c, -s)
    return torch.stack([ex, ey], dim=-1)


def integrate(out: torch.Tensor) -> torch.Tensor:
    """Normalised per-step ego displacements (B,K,2) -> cumulative ego displacement in units."""
    return torch.cumsum(out * C.STEP_SCALE, dim=1)


def decode(out: torch.Tensor, pos_last: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
    """Network output (B,K,2) -> predicted absolute world positions (B,K,2)."""
    cum = integrate(out)
    c = torch.cos(h_last)[:, None]
    s = torch.sin(h_last)[:, None]
    wx, wy = _rot(cum[..., 0], cum[..., 1], c, s)
    return torch.stack([wx, wy], dim=-1) + pos_last[:, None, :]
