"""Game logic of a 1-vs-1 match, independent of any rendering.

The playable game (game.py) and the headless AI tournament both drive a
`Match`. A controller is any object with
    reset()
    control(tank, target_tracker, world, dt) -> (throttle, turn, turret_angle | None, fire)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config as C
from .ai import TrajectoryTracker
from .sim import Bullet, Tank, World


@dataclass
class Explosion:
    x: float
    y: float
    t: float = 0.0
    size: float = 1.0
    duration: float = 0.5


class Match:
    ROUND_PAUSE = 2.0       # seconds between rounds
    ROUND_TIMEOUT = 90.0    # draw after this many seconds

    def __init__(self, controllers: list, seed: int = 0, world: World | None = None):
        self.world = world or World()
        self.rng = np.random.default_rng(seed)
        self.controllers = controllers
        self.tanks = [Tank(0, 0, team=0), Tank(0, 0, team=1)]
        self.trackers = [TrajectoryTracker(), TrajectoryTracker()]
        self.bullets: list[Bullet] = []
        self.explosions: list[Explosion] = []
        self.score = [0, 0]
        self.draws = 0
        self.shots = [0, 0]
        self.hits = [0, 0]
        self.rounds = 0
        self.round_time = 0.0
        self.time = 0.0
        self.pause_timer = 0.0
        self.winner: int | None = None
        self.reset_round()

    # ------------------------------------------------------------------ rounds
    def reset_round(self):
        a, b = self.tanks
        for _ in range(200):
            pa = self.world.random_free_position(self.rng, margin=30.0)
            pb = self.world.random_free_position(self.rng, margin=30.0)
            if math.dist(pa, pb) > 450.0:
                break
        for t, p, other in ((a, pa, pb), (b, pb, pa)):
            t.x, t.y = p
            t.heading = math.atan2(other[1] - p[1], other[0] - p[0]) + self.rng.uniform(-1.0, 1.0)
            t.turret = math.atan2(other[1] - p[1], other[0] - p[0])
            t.speed = 0.0
            t.hp = C.TANK_HP
            t.reload = C.RELOAD_TIME
        self.bullets.clear()
        for tr in self.trackers:
            tr.reset()
        for c in self.controllers:
            c.reset()
        self.round_time = 0.0
        self.pause_timer = 0.0
        self.winner = None

    @property
    def round_over(self) -> bool:
        return self.pause_timer > 0.0

    def accuracy(self, team: int) -> float:
        return self.hits[team] / self.shots[team] if self.shots[team] else 0.0

    # ------------------------------------------------------------------ simulation
    def step(self, dt: float = C.SIM_DT) -> None:
        self.time += dt
        if self.round_over:
            self.pause_timer -= dt
            self._step_effects(dt)
            for b in list(self.bullets):
                if not self.world.step_bullet(b, dt):
                    self.bullets.remove(b)
            if self.pause_timer <= 0.0:
                self.reset_round()
            return

        self.round_time += dt
        for tank, tracker in zip(self.tanks, self.trackers):
            tracker.observe(tank, self.world)

        for i, (tank, ctrl) in enumerate(zip(self.tanks, self.controllers)):
            throttle, turn, turret, fire = ctrl.control(tank, self.trackers[1 - i], self.world, dt)
            self.world.step_tank(tank, throttle, turn, dt)
            if turret is not None:
                self.world.rotate_turret(tank, turret, dt)
            if fire:
                b = self.world.fire(tank)
                if b is not None:
                    self.bullets.append(b)
                    self.shots[i] += 1

        for b in list(self.bullets):
            alive = self.world.step_bullet(b, dt)
            hit_tank = False
            if alive:
                for j, t in enumerate(self.tanks):
                    if j != b.owner and t.alive and math.hypot(b.x - t.x, b.y - t.y) < C.TANK_RADIUS + C.BULLET_RADIUS:
                        t.hp -= 1
                        self.hits[b.owner] += 1
                        self.explosions.append(Explosion(b.x, b.y, size=1.0 if t.alive else 2.5,
                                                         duration=0.4 if t.alive else 1.0))
                        hit_tank = True
                        break
            if hit_tank or not alive:
                self.bullets.remove(b)
                if not hit_tank:
                    self.explosions.append(Explosion(b.x, b.y, size=0.35, duration=0.25))
        self._step_effects(dt)

        dead = [i for i, t in enumerate(self.tanks) if not t.alive]
        if dead or self.round_time > self.ROUND_TIMEOUT:
            self.rounds += 1
            if len(dead) == 1:
                self.winner = 1 - dead[0]
                self.score[self.winner] += 1
            else:
                self.winner = None
                self.draws += 1
            self.pause_timer = self.ROUND_PAUSE

    def _step_effects(self, dt: float):
        for e in list(self.explosions):
            e.t += dt
            if e.t > e.duration:
                self.explosions.remove(e)


def run_tournament(controller_a, controller_b, rounds: int = 50, seed: int = 0,
                   max_seconds: float = 1e9, log=None) -> dict:
    """Headless AI-vs-AI tournament. Returns score and accuracy for both sides."""
    match = Match([controller_a, controller_b], seed=seed)
    match.ROUND_PAUSE = C.SIM_DT  # no pause needed without graphics
    done = 0
    while match.rounds < rounds and match.time < max_seconds:
        match.step()
        if log and match.rounds != done:
            done = match.rounds
            log(f"round {done}: score {match.score[0]}:{match.score[1]} draws={match.draws} "
                f"accuracy {match.accuracy(0):.1%} / {match.accuracy(1):.1%}")
    return {"rounds": match.rounds, "score": list(match.score), "draws": match.draws,
            "shots": list(match.shots), "hits": list(match.hits),
            "accuracy": [match.accuracy(0), match.accuracy(1)], "game_seconds": round(match.time, 1)}
