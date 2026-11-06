import math

import numpy as np
import pytest

from tankgame import config as C
from tankgame.sim import Tank, World, wrap_angle


def run(world, tank, throttle, turn, seconds):
    for _ in range(int(seconds * C.SIM_HZ)):
        world.step_tank(tank, throttle, turn)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi) or wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(0.5) == pytest.approx(0.5)
    assert wrap_angle(-7.0) == pytest.approx(-7.0 + 2 * math.pi)


def test_tank_accelerates_to_max_speed_and_drives_straight():
    w = World(obstacles=())
    t = Tank(100, 350, heading=0.0)
    run(w, t, 1.0, 0.0, 1.5)
    assert t.speed == pytest.approx(C.TANK_MAX_SPEED)
    assert t.y == pytest.approx(350)
    assert t.x > 100 + 150


def test_reverse_speed_is_limited():
    w = World(obstacles=())
    t = Tank(500, 350)
    run(w, t, -1.0, 0.0, 2.0)
    assert t.speed == pytest.approx(-C.TANK_MAX_REVERSE)
    assert t.x < 500


def test_turning_rate():
    w = World(obstacles=())
    t = Tank(500, 350)
    run(w, t, 0.0, 1.0, 0.5)
    assert t.heading == pytest.approx(C.TANK_TURN_RATE * 0.5, abs=1e-6)


def test_walls_keep_tank_inside():
    w = World(obstacles=())
    t = Tank(500, 350, heading=0.0)
    run(w, t, 1.0, 0.0, 10.0)
    assert t.x == pytest.approx(C.ARENA_W - C.TANK_RADIUS)
    assert w.is_free(t.x, t.y, C.TANK_RADIUS - 1e-6)


def test_obstacle_collision_pushes_out():
    w = World()
    o = w.obstacles[4]  # central block
    t = Tank(o.x - 60, o.y + o.h / 2, heading=0.0)
    run(w, t, 1.0, 0.0, 3.0)
    assert t.x <= o.x - C.TANK_RADIUS + 1e-6
    x, y, hit = w.collide_circle(o.x + o.w / 2, o.y + o.h / 2, C.TANK_RADIUS)
    assert hit and not o.contains(x, y)


def test_ray_distance():
    w = World(obstacles=())
    assert w.ray_distance(100, 350, 0.0, max_dist=5000) == pytest.approx(C.ARENA_W - 100)
    assert w.ray_distance(100, 350, math.pi, max_dist=5000) == pytest.approx(100)
    assert w.ray_distance(100, 350, 0.0) == pytest.approx(C.RAY_MAX)
    w2 = World()
    o = w2.obstacles[4]
    d = w2.ray_distance(o.x - 50, o.y + 10, 0.0)
    assert d == pytest.approx(50)


def test_segment_blocked():
    w = World()
    o = w.obstacles[4]
    cy = o.y + o.h / 2
    assert w.segment_blocked(o.x - 100, cy, o.x1 + 100, cy)
    assert not w.segment_blocked(50, 20, 950, 20)


def test_fire_and_reload():
    w = World(obstacles=())
    t = Tank(100, 350, turret=0.0)
    b = w.fire(t)
    assert b is not None and b.vx == pytest.approx(C.BULLET_SPEED)
    assert w.fire(t) is None            # reloading
    run(w, t, 0.0, 0.0, C.RELOAD_TIME + 0.05)
    assert w.fire(t) is not None


def test_bullet_lifecycle():
    w = World()
    t = Tank(30, 30, turret=0.0)
    b = w.fire(t)
    steps = 0
    while w.step_bullet(b):
        steps += 1
    assert 0 < steps < C.BULLET_LIFETIME * C.SIM_HZ + 1


def test_turret_rotation_is_rate_limited():
    w = World()
    t = Tank(100, 100, turret=0.0)
    err = w.rotate_turret(t, math.pi / 2)
    assert t.turret == pytest.approx(C.TURRET_TURN_RATE * C.SIM_DT)
    assert err == pytest.approx(math.pi / 2 - C.TURRET_TURN_RATE * C.SIM_DT)


def test_random_free_position():
    w = World()
    rng = np.random.default_rng(0)
    for _ in range(100):
        x, y = w.random_free_position(rng)
        assert w.is_free(x, y, C.TANK_RADIUS)
