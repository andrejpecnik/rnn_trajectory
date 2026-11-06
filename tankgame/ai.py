"""Opponent observation (trajectory tracking) and the computer controlled tank."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import config as C
from .aiming import aim_angle, intercept_point
from .behaviors import Behavior
from .predictors import Predictor
from .sim import Tank, World


class TrajectoryTracker:
    """Observes one tank every physics tick and keeps its history at the model rate.

    It also remembers past predictions about this tank so that the real
    prediction error can be measured live (shown in the game HUD).
    """

    def __init__(self, keep: int = C.HISTORY + 1 + C.HORIZON + 5):
        self.pos: deque = deque(maxlen=keep)
        self.head: deque = deque(maxlen=keep)
        self.rays: deque = deque(maxlen=keep)
        self.tick = 0
        self.samples = 0
        self.recording: list | None = None
        self._pending: deque = deque(maxlen=C.HORIZON + 2)
        self.errors: deque = deque(maxlen=90)

    def reset(self):
        self.pos.clear()
        self.head.clear()
        self.rays.clear()
        self._pending.clear()
        self.tick = 0
        if self.recording is not None and self.recording[-1]:
            self.recording.append([])   # every round is stored as a separate episode

    def start_recording(self):
        self.recording = [[]]

    def observe(self, tank: Tank, world: World) -> bool:
        """Call once per physics tick. Returns True when a new model-rate sample was taken."""
        new = self.tick % C.MODEL_STRIDE == 0
        if new:
            r = world.rays(tank.x, tank.y, tank.heading)
            self.pos.append((tank.x, tank.y))
            self.head.append(tank.heading)
            self.rays.append(r)
            self.samples += 1
            if self.recording is not None:
                self.recording[-1].append((tank.x, tank.y, tank.heading, r[0], r[1], r[2], 0.0))
            self._score_predictions()
        self.tick += 1
        return new

    def history(self, length: int = C.HISTORY + 1):
        n = min(length, len(self.pos))
        pos = np.asarray(list(self.pos)[-n:], dtype=np.float64)
        head = np.asarray(list(self.head)[-n:], dtype=np.float64)
        rays = np.asarray(list(self.rays)[-n:], dtype=np.float64)
        return pos, head, rays

    # -- live error measurement ------------------------------------------------
    def register_prediction(self, prediction: np.ndarray):
        self._pending.append((self.samples, prediction))

    def _score_predictions(self):
        while self._pending and self.samples - self._pending[0][0] >= C.HORIZON:
            made_at, pred = self._pending.popleft()
            back = self.samples - made_at        # samples observed since the prediction
            if back > len(self.pos):
                continue
            actual = np.asarray(list(self.pos)[-back:][:C.HORIZON])
            self.errors.append(float(np.linalg.norm(pred[:len(actual)] - actual, axis=1).mean()))

    @property
    def mean_error(self) -> float | None:
        return float(np.mean(self.errors)) if self.errors else None


class AIController:
    """Computer tank: moves with a behaviour, aims with a trajectory predictor."""

    def __init__(self, behavior: Behavior, predictor: Predictor, fire_tolerance: float = 0.035,
                 aim_noise: float = 0.0, reload_time: float | None = None, seed: int = 0):
        self.behavior = behavior
        self.predictor = predictor
        self.fire_tolerance = fire_tolerance
        self.aim_noise = aim_noise          # std of a random aim error (radians), used for difficulty
        self.reload_time = reload_time      # None = default reload time
        self._rng = np.random.default_rng(seed)
        self._offset = 0.0
        self.prediction: np.ndarray | None = None
        self.aim_point: np.ndarray | None = None
        self._last_sample = -1

    def reset(self):
        self.prediction = None
        self.aim_point = None
        self._last_sample = -1

    def plan_aim(self, tank: Tank, target_tracker: TrajectoryTracker, world: World):
        """Re-plan whenever the tracker has a new sample of the target (30 Hz)."""
        if target_tracker.samples == self._last_sample or not target_tracker.pos:
            return
        self._last_sample = target_tracker.samples
        pos, head, rays = target_tracker.history()
        self.prediction = self.predictor.predict(pos, head, rays)
        target_tracker.register_prediction(self.prediction)
        self.aim_point, _ = intercept_point((tank.x, tank.y), pos[-1], self.prediction)

    def control(self, tank: Tank, target_tracker: TrajectoryTracker, world: World, dt: float = C.SIM_DT):
        """Returns (throttle, turn, turret_angle, fire)."""
        if self.reload_time is not None:
            tank.reload_time = self.reload_time
        throttle, turn = self.behavior.act(tank, world, dt)
        self.plan_aim(tank, target_tracker, world)
        if self.aim_point is None:
            return throttle, turn, None, False
        angle = aim_angle((tank.x, tank.y), self.aim_point) + self._offset
        err = abs(math.atan2(math.sin(angle - tank.turret), math.cos(angle - tank.turret)))
        fire = False
        if err < self.fire_tolerance and tank.reload <= 0.0:
            mx, my = tank.muzzle()
            dist = math.hypot(self.aim_point[0] - mx, self.aim_point[1] - my)
            in_range = dist < C.BULLET_SPEED * C.BULLET_LIFETIME * 0.8
            fire = in_range and not world.segment_blocked(mx, my, *self.aim_point)
            if fire and self.aim_noise > 0.0:
                self._offset = float(self._rng.normal(0.0, self.aim_noise))  # new error for next shot
        return throttle, turn, angle, fire
