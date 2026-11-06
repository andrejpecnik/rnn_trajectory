"""Aiming: turn a predicted trajectory into a firing direction.

A bullet fired now from the tank centre g travels MUZZLE_LENGTH + BULLET_SPEED * t
after t seconds. We look for the first predicted time t_k at which the target
is closer to g than the bullet would have travelled - that is where the bullet
and the target meet. Between two prediction samples we interpolate linearly;
beyond the prediction horizon we extrapolate with constant velocity.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C


def solve_cv_intercept(g, p, v, speed: float = C.BULLET_SPEED, muzzle: float = 0.0) -> float | None:
    """Smallest t >= 0 with |p + v t - g| = muzzle + speed * t (target moving with constant v)."""
    dx, dy = p[0] - g[0], p[1] - g[1]
    a = v[0] * v[0] + v[1] * v[1] - speed * speed
    b = 2.0 * (dx * v[0] + dy * v[1] - muzzle * speed)
    c = dx * dx + dy * dy - muzzle * muzzle
    if c <= 0.0:
        return 0.0
    if abs(a) < 1e-9:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t >= 0.0 else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    roots = sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a)))
    for t in roots:
        if t >= 0.0:
            return t
    return None


def intercept_point(gun, current, future: np.ndarray, dt: float = C.MODEL_DT,
                    speed: float = C.BULLET_SPEED, muzzle: float = C.MUZZLE_LENGTH):
    """Return (aim_point, time_to_impact) for a target whose future positions are `future` (K,2)."""
    g = np.asarray(gun, dtype=np.float64)
    pts = np.vstack([np.asarray(current, dtype=np.float64)[None], np.asarray(future, dtype=np.float64)])
    t = np.arange(len(pts)) * dt
    f = np.hypot(pts[:, 0] - g[0], pts[:, 1] - g[1]) - (muzzle + speed * t)
    if f[0] <= 0.0:
        return pts[0], 0.0
    hit = np.nonzero(f <= 0.0)[0]
    if len(hit):
        i = int(hit[0])
        a = f[i - 1] / (f[i - 1] - f[i])
        point = pts[i - 1] + a * (pts[i] - pts[i - 1])
        return point, float(t[i - 1] + a * dt)
    # target still out of reach at the end of the horizon: constant velocity extrapolation
    v = (pts[-1] - pts[-2]) / dt
    t_end = t[-1]
    tt = solve_cv_intercept(g, pts[-1], v, speed, muzzle + speed * t_end)
    if tt is None:
        return pts[-1], float(t_end)
    point = pts[-1] + v * tt
    point[0] = min(max(point[0], C.TANK_RADIUS), C.ARENA_W - C.TANK_RADIUS)
    point[1] = min(max(point[1], C.TANK_RADIUS), C.ARENA_H - C.TANK_RADIUS)
    return point, float(t_end + tt)


def aim_angle(gun, point) -> float:
    return math.atan2(point[1] - gun[1], point[0] - gun[0])
