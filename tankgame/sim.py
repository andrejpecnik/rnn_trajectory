"""Deterministic 2D physics simulator of the tank arena (no graphics).

The same code is used for data generation, evaluation and the playable game,
so trajectories the network sees during training follow exactly the physics of
the game.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config as C

INF = float("inf")


def wrap_angle(a: float) -> float:
    """Wrap an angle to the interval (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        return (self.x - margin <= px <= self.x1 + margin
                and self.y - margin <= py <= self.y1 + margin)


def ray_rect(ox, oy, dx, dy, x0, y0, x1, y1) -> float:
    """Distance along a unit ray (ox,oy)+t*(dx,dy) to an axis aligned box, INF if missed."""
    tmin, tmax = -INF, INF
    for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return INF
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return INF
    if tmax < 0.0:
        return INF
    return max(tmin, 0.0)


@dataclass
class Tank:
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    turret: float = 0.0
    hp: int = C.TANK_HP
    reload: float = 0.0
    team: int = 0
    reload_time: float = C.RELOAD_TIME

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def muzzle(self) -> tuple[float, float]:
        return (self.x + math.cos(self.turret) * C.MUZZLE_LENGTH,
                self.y + math.sin(self.turret) * C.MUZZLE_LENGTH)


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    owner: int
    age: float = 0.0


@dataclass
class World:
    width: float = C.ARENA_W
    height: float = C.ARENA_H
    obstacles: tuple = field(default_factory=lambda: tuple(Rect(*r) for r in C.OBSTACLES))

    # ------------------------------------------------------------------ geometry
    def collide_circle(self, x: float, y: float, r: float) -> tuple[float, float, bool]:
        """Push a circle out of walls and obstacles. Returns new position and hit flag."""
        hit = False
        if x < r:
            x, hit = r, True
        elif x > self.width - r:
            x, hit = self.width - r, True
        if y < r:
            y, hit = r, True
        elif y > self.height - r:
            y, hit = self.height - r, True
        for o in self.obstacles:
            cx = min(max(x, o.x), o.x1)
            cy = min(max(y, o.y), o.y1)
            dx, dy = x - cx, y - cy
            d2 = dx * dx + dy * dy
            if d2 >= r * r:
                continue
            hit = True
            if d2 > 1e-12:
                d = math.sqrt(d2)
                x += dx / d * (r - d)
                y += dy / d * (r - d)
            else:  # centre inside the box: leave through the closest side
                pens = ((x - o.x + r, -1, 0), (o.x1 - x + r, 1, 0),
                        (y - o.y + r, 0, -1), (o.y1 - y + r, 0, 1))
                p, sx, sy = min(pens)
                x += sx * p
                y += sy * p
        return x, y, hit

    def is_free(self, x: float, y: float, r: float) -> bool:
        if not (r <= x <= self.width - r and r <= y <= self.height - r):
            return False
        return not any(o.contains(x, y, r) for o in self.obstacles)

    def ray_distance(self, x: float, y: float, angle: float, max_dist: float = C.RAY_MAX,
                     inflate: float = 0.0) -> float:
        """Free distance from (x, y) in direction `angle` to a wall or (inflated) obstacle."""
        dx, dy = math.cos(angle), math.sin(angle)
        best = max_dist
        lo_x, hi_x = inflate, self.width - inflate
        lo_y, hi_y = inflate, self.height - inflate
        if dx > 1e-12:
            best = min(best, (hi_x - x) / dx)
        elif dx < -1e-12:
            best = min(best, (lo_x - x) / dx)
        if dy > 1e-12:
            best = min(best, (hi_y - y) / dy)
        elif dy < -1e-12:
            best = min(best, (lo_y - y) / dy)
        for o in self.obstacles:
            t = ray_rect(x, y, dx, dy, o.x - inflate, o.y - inflate, o.x1 + inflate, o.y1 + inflate)
            if t < best:
                best = t
        return max(best, 0.0)

    def rays(self, x: float, y: float, heading: float) -> tuple[float, float, float]:
        return tuple(self.ray_distance(x, y, heading + a) for a in C.RAY_ANGLES)

    def segment_blocked(self, x0: float, y0: float, x1: float, y1: float,
                        inflate: float = C.BULLET_RADIUS) -> bool:
        """True if the straight segment hits an obstacle (used for line of sight)."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return False
        dx, dy = dx / length, dy / length
        for o in self.obstacles:
            t = ray_rect(x0, y0, dx, dy, o.x - inflate, o.y - inflate, o.x1 + inflate, o.y1 + inflate)
            if t <= length:
                return True
        return False

    def random_free_position(self, rng: np.random.Generator, r: float = C.TANK_RADIUS,
                             margin: float = 20.0) -> tuple[float, float]:
        for _ in range(1000):
            x = rng.uniform(r + margin, self.width - r - margin)
            y = rng.uniform(r + margin, self.height - r - margin)
            if self.is_free(x, y, r + margin):
                return x, y
        raise RuntimeError("no free position found")

    # ------------------------------------------------------------------ dynamics
    def step_tank(self, tank: Tank, throttle: float, turn: float, dt: float = C.SIM_DT) -> bool:
        """Advance one tank by dt using throttle/turn inputs in [-1, 1]. Returns collision flag."""
        throttle = max(-1.0, min(1.0, throttle))
        turn = max(-1.0, min(1.0, turn))
        target = throttle * (C.TANK_MAX_SPEED if throttle >= 0.0 else C.TANK_MAX_REVERSE)
        speeding_up = (target * tank.speed >= 0.0) and abs(target) > abs(tank.speed)
        a = C.TANK_ACCEL if speeding_up else C.TANK_BRAKE
        dv = max(-a * dt, min(a * dt, target - tank.speed))
        tank.speed += dv
        tank.heading = wrap_angle(tank.heading + turn * C.TANK_TURN_RATE * dt)
        nx = tank.x + math.cos(tank.heading) * tank.speed * dt
        ny = tank.y + math.sin(tank.heading) * tank.speed * dt
        nx, ny, hit = self.collide_circle(nx, ny, C.TANK_RADIUS)
        if hit:
            tank.speed *= 0.5
        tank.x, tank.y = nx, ny
        if tank.reload > 0.0:
            tank.reload = max(0.0, tank.reload - dt)
        return hit

    def rotate_turret(self, tank: Tank, target_angle: float, dt: float = C.SIM_DT) -> float:
        """Rotate the turret towards target_angle with limited speed. Returns remaining error."""
        err = wrap_angle(target_angle - tank.turret)
        max_step = C.TURRET_TURN_RATE * dt
        tank.turret = wrap_angle(tank.turret + max(-max_step, min(max_step, err)))
        return abs(wrap_angle(target_angle - tank.turret))

    def fire(self, tank: Tank) -> Bullet | None:
        if tank.reload > 0.0 or not tank.alive:
            return None
        tank.reload = tank.reload_time
        mx, my = tank.muzzle()
        return Bullet(mx, my, math.cos(tank.turret) * C.BULLET_SPEED,
                      math.sin(tank.turret) * C.BULLET_SPEED, owner=tank.team)

    def step_bullet(self, b: Bullet, dt: float = C.SIM_DT, obstacles: bool = True) -> bool:
        """Move a bullet. Returns False when the bullet should be removed."""
        b.x += b.vx * dt
        b.y += b.vy * dt
        b.age += dt
        if b.age > C.BULLET_LIFETIME:
            return False
        if not (0.0 <= b.x <= self.width and 0.0 <= b.y <= self.height):
            return False
        if obstacles and any(o.contains(b.x, b.y, C.BULLET_RADIUS) for o in self.obstacles):
            return False
        return True
