"""Movement policies ("drivers") for computer controlled tanks.

Every behaviour turns the current tank state into (throttle, turn) inputs.
They are deliberately diverse - circling, zig-zagging, waypoint driving,
sudden reversals, stop-and-go - so that the trajectory dataset contains
patterns that a simple constant-velocity extrapolation cannot predict, but a
recurrent network that remembers the recent past can.
"""

from __future__ import annotations

import math

import numpy as np

from . import config as C
from .sim import Tank, World, wrap_angle


class Behavior:
    name = "base"
    behavior_id = 0

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.t = 0.0
        self.noise = 0.0
        self._avoid_cache = (0.0, 0.0, 1.0)
        self._avoid_tick = 0

    # -- helpers -------------------------------------------------------------
    def _ou_noise(self, dt: float, sigma: float = 0.25, tau: float = 0.6) -> float:
        """Smooth (Ornstein-Uhlenbeck) noise added to the steering input."""
        self.noise += -self.noise * dt / tau + sigma * math.sqrt(2.0 * dt / tau) * self.rng.normal()
        return self.noise

    def _avoid(self, tank: Tank, world: World, throttle: float, turn: float):
        """Blend steering with obstacle avoidance based on three look-ahead rays."""
        self._avoid_tick += 1
        if self._avoid_tick % 3 == 1:  # rays are recomputed at 20 Hz
            direction = tank.heading if tank.speed >= 0.0 else tank.heading + math.pi
            look = 60.0 + abs(tank.speed) * 0.7
            front = world.ray_distance(tank.x, tank.y, direction, look, inflate=C.TANK_RADIUS)
            if front < look:
                left = world.ray_distance(tank.x, tank.y, direction + 0.7, look * 1.5, inflate=C.TANK_RADIUS)
                right = world.ray_distance(tank.x, tank.y, direction - 0.7, look * 1.5, inflate=C.TANK_RADIUS)
                steer = 1.0 if left > right else -1.0
                urgency = 1.0 - front / look
                self._avoid_cache = (steer, urgency, front)
            else:
                self._avoid_cache = (0.0, 0.0, front)
        steer, urgency, _ = self._avoid_cache
        if urgency > 0.0:
            turn = (1.0 - urgency) * turn + urgency * steer
            throttle *= 1.0 - 0.6 * urgency
        return throttle, turn

    # -- interface -----------------------------------------------------------
    def control(self, tank: Tank, world: World, dt: float) -> tuple[float, float]:
        raise NotImplementedError

    def act(self, tank: Tank, world: World, dt: float = C.SIM_DT) -> tuple[float, float]:
        self.t += dt
        throttle, turn = self.control(tank, world, dt)
        return self._avoid(tank, world, throttle, turn)


class CircleBehavior(Behavior):
    """Drives in circles of random radius, occasionally changing direction."""
    name, behavior_id = "circle", 1

    def __init__(self, rng):
        super().__init__(rng)
        self._new()

    def _new(self):
        self.throttle = self.rng.uniform(0.5, 1.0)
        self.turn = self.rng.choice([-1, 1]) * self.rng.uniform(0.25, 0.9)
        self.until = self.t + self.rng.uniform(3.0, 8.0)

    def control(self, tank, world, dt):
        if self.t > self.until:
            self._new()
        return self.throttle, self.turn + 0.3 * self._ou_noise(dt, 0.2)


class ZigZagBehavior(Behavior):
    """Alternates hard left/right turns with a fixed period."""
    name, behavior_id = "zigzag", 2

    def __init__(self, rng):
        super().__init__(rng)
        self.period = rng.uniform(0.5, 1.6)
        self.amp = rng.uniform(0.6, 1.0)
        self.throttle = rng.uniform(0.7, 1.0)
        self.sign = rng.choice([-1, 1])
        self.next_flip = self.period * rng.uniform(0.3, 1.0)

    def control(self, tank, world, dt):
        if self.t > self.next_flip:
            self.sign = -self.sign
            self.next_flip = self.t + self.period * self.rng.uniform(0.85, 1.15)
        return self.throttle, self.sign * self.amp + 0.2 * self._ou_noise(dt, 0.2)


class SineBehavior(Behavior):
    """Smooth weaving: steering follows a sine wave."""
    name, behavior_id = "sine", 3

    def __init__(self, rng):
        super().__init__(rng)
        self.freq = rng.uniform(0.25, 0.9)
        self.amp = rng.uniform(0.5, 1.0)
        self.phase = rng.uniform(0, 2 * math.pi)
        self.throttle = rng.uniform(0.6, 1.0)

    def control(self, tank, world, dt):
        turn = self.amp * math.sin(2 * math.pi * self.freq * self.t + self.phase)
        return self.throttle, turn + 0.2 * self._ou_noise(dt, 0.2)


class WaypointBehavior(Behavior):
    """Drives to random points in the arena with a proportional steering controller."""
    name, behavior_id = "waypoint", 4

    def __init__(self, rng):
        super().__init__(rng)
        self.target = None
        self.gain = rng.uniform(1.5, 3.0)

    def control(self, tank, world, dt):
        if self.target is None or math.hypot(self.target[0] - tank.x, self.target[1] - tank.y) < 40.0:
            self.target = world.random_free_position(self.rng, margin=30.0)
        desired = math.atan2(self.target[1] - tank.y, self.target[0] - tank.x)
        err = wrap_angle(desired - tank.heading)
        turn = max(-1.0, min(1.0, self.gain * err))
        throttle = 1.0 if abs(err) < 0.6 else 0.35
        return throttle, turn + 0.15 * self._ou_noise(dt)


class DodgeBehavior(Behavior):
    """Evasive driving: sudden switches between forward and reverse with small turns."""
    name, behavior_id = "dodge", 5

    def __init__(self, rng):
        super().__init__(rng)
        self.dir = rng.choice([-1.0, 1.0])
        self.turn = 0.0
        self.next_switch = rng.uniform(0.4, 1.5)

    def control(self, tank, world, dt):
        if self.t > self.next_switch:
            self.dir = -self.dir
            self.turn = self.rng.uniform(-0.6, 0.6)
            self.next_switch = self.t + self.rng.uniform(0.5, 1.8)
        return self.dir, self.turn + 0.2 * self._ou_noise(dt)


class StopAndGoBehavior(Behavior):
    """Accelerates, brakes to a stop, waits, then goes again in a new direction."""
    name, behavior_id = "stopgo", 6

    def __init__(self, rng):
        super().__init__(rng)
        self.phase = "go"
        self.until = rng.uniform(0.8, 2.0)
        self.turn = rng.uniform(-0.5, 0.5)

    def control(self, tank, world, dt):
        if self.t > self.until:
            if self.phase == "go":
                self.phase = "stop"
                self.until = self.t + self.rng.uniform(0.3, 2.5)
            else:
                self.phase = "go"
                self.until = self.t + self.rng.uniform(0.8, 2.2)
                self.turn = self.rng.uniform(-0.7, 0.7)
        if self.phase == "go":
            return 1.0, self.turn + 0.2 * self._ou_noise(dt)
        return 0.0, 0.0


class IdleBehavior(Behavior):
    """Mostly standing still, sometimes rotating in place or creeping slowly.

    Human players often stop to aim; without such data the network would assume
    that a standing tank is about to drive off.
    """
    name, behavior_id = "idle", 8

    def __init__(self, rng):
        super().__init__(rng)
        self._new()

    def _new(self):
        r = self.rng.random()
        if r < 0.55:
            self.cmd = (0.0, 0.0)                                         # stand still
            self.until = self.t + self.rng.uniform(1.5, 5.0)
        elif r < 0.8:
            self.cmd = (0.0, self.rng.choice([-1, 1]) * self.rng.uniform(0.4, 1.0))  # rotate in place
            self.until = self.t + self.rng.uniform(0.4, 1.5)
        else:
            self.cmd = (self.rng.uniform(-0.4, 0.4), self.rng.uniform(-0.3, 0.3))   # creep
            self.until = self.t + self.rng.uniform(0.5, 2.0)

    def control(self, tank, world, dt):
        if self.t > self.until:
            self._new()
        return self.cmd


BASE_BEHAVIORS = (CircleBehavior, ZigZagBehavior, SineBehavior, WaypointBehavior,
                  DodgeBehavior, StopAndGoBehavior, IdleBehavior)
# opponents in the game should keep moving - no idle behaviour for them
GAME_BEHAVIORS = BASE_BEHAVIORS[:-1]


class MixedBehavior(Behavior):
    """Switches between the base behaviours every few seconds (most realistic opponent)."""
    name, behavior_id = "mixed", 7

    def __init__(self, rng, min_time: float = 2.5, max_time: float = 7.0, pool=BASE_BEHAVIORS):
        super().__init__(rng)
        self.min_time, self.max_time = min_time, max_time
        self.pool = pool
        self._switch()

    def _switch(self):
        cls = self.pool[self.rng.integers(len(self.pool))]
        self.current = cls(self.rng)
        self.until = self.t + self.rng.uniform(self.min_time, self.max_time)

    @property
    def active_id(self) -> int:
        return self.current.behavior_id

    def control(self, tank, world, dt):
        if self.t > self.until:
            self._switch()
        self.current.t += dt
        return self.current.control(tank, world, dt)


BEHAVIORS = {cls.name: cls for cls in (*BASE_BEHAVIORS, MixedBehavior)}
BEHAVIOR_NAMES = {cls.behavior_id: cls.name for cls in BEHAVIORS.values()}


def active_behavior_id(b: Behavior) -> int:
    return b.active_id if isinstance(b, MixedBehavior) else b.behavior_id


def make_behavior(name: str, rng: np.random.Generator) -> Behavior:
    return BEHAVIORS[name](rng)


def random_behavior(rng: np.random.Generator, p_mixed: float = 0.6) -> Behavior:
    if rng.random() < p_mixed:
        return MixedBehavior(rng)
    cls = BASE_BEHAVIORS[rng.integers(len(BASE_BEHAVIORS))]
    return cls(rng)
