import math

import numpy as np
import pytest
import torch

from tankgame import config as C
from tankgame.aiming import intercept_point, solve_cv_intercept
from tankgame.features import build_features, decode, ego_targets, integrate
from tankgame.models import build_model
from tankgame.predictors import (MODELS_DIR, CTRVPredictor, CVPredictor, NeuralPredictor, StaticPredictor,
                                 load_predictor)

H1 = C.HISTORY + 1


def straight_line(n, x0=200.0, y0=300.0, angle=0.3, step=4.0):
    k = np.arange(n)[:, None]
    d = np.array([math.cos(angle), math.sin(angle)])
    pos = np.array([x0, y0]) + k * step * d
    return pos, np.full(n, angle), np.full((n, 3), C.RAY_MAX)


def circle(n, r=150.0, cx=500.0, cy=350.0, w=0.04):
    a = np.arange(n) * w
    pos = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    head = a + math.pi / 2
    return pos, head, np.full((n, 3), C.RAY_MAX)


# ------------------------------------------------------------------ features
def test_feature_shapes_and_ego_frame():
    pos, head, rays = straight_line(H1)
    x = build_features(torch.tensor(pos[None]), torch.tensor(head[None]), torch.tensor(rays[None]))
    assert x.shape == (1, C.HISTORY, C.N_FEATURES)
    # moving straight along the heading -> velocity is purely along ego x
    assert torch.allclose(x[0, :, 0], torch.full((C.HISTORY,), 4.0 / C.STEP_SCALE, dtype=x.dtype))
    assert torch.allclose(x[0, :, 1], torch.zeros(C.HISTORY, dtype=x.dtype), atol=1e-9)


def test_decode_inverts_ego_targets():
    rng = np.random.default_rng(0)
    pos_last = torch.tensor(rng.uniform(100, 600, (5, 2)))
    h_last = torch.tensor(rng.uniform(-3, 3, 5))
    future = pos_last[:, None, :] + torch.tensor(rng.normal(0, 30, (5, C.HORIZON, 2))).cumsum(1)
    ego = ego_targets(future, pos_last, h_last)
    # build the per-step network output that would produce exactly this future
    steps = torch.diff(ego, dim=1, prepend=torch.zeros(5, 1, 2, dtype=ego.dtype)) / C.STEP_SCALE
    assert torch.allclose(integrate(steps), ego)
    assert torch.allclose(decode(steps, pos_last, h_last), future, atol=1e-8)


@pytest.mark.parametrize("kind", ["gru", "lstm", "mlp"])
def test_models_forward(kind):
    m = build_model(kind)
    out = m(torch.zeros(7, C.HISTORY, C.N_FEATURES))
    assert out.shape == (7, C.HORIZON, 2)


# ------------------------------------------------------------------ baselines
def test_static_predictor():
    pos, head, rays = straight_line(10)
    p = StaticPredictor().predict(pos, head, rays)
    assert p.shape == (C.HORIZON, 2)
    assert np.allclose(p, pos[-1])


def test_cv_is_exact_on_straight_line():
    pos, head, rays = straight_line(H1, step=3.0)
    p = CVPredictor().predict(pos, head, rays)
    truth = straight_line(H1 + C.HORIZON, step=3.0)[0][H1:]
    assert np.allclose(p, truth)


def test_ctrv_is_exact_on_circle_and_beats_cv():
    pos, head, rays = circle(H1 + C.HORIZON)
    hist = slice(0, H1)
    p_ctrv = CTRVPredictor().predict(pos[hist], head[hist], rays[hist])
    p_cv = CVPredictor().predict(pos[hist], head[hist], rays[hist])
    truth = pos[H1:]
    e_ctrv = np.linalg.norm(p_ctrv - truth, axis=1).mean()
    e_cv = np.linalg.norm(p_cv - truth, axis=1).mean()
    assert e_ctrv < 2.0
    assert e_ctrv < e_cv


def test_predictions_are_clamped_to_arena():
    pos, head, rays = straight_line(H1, x0=900, y0=300, angle=0.0, step=5.0)
    p = CVPredictor().predict(pos, head, rays)
    assert p[:, 0].max() <= C.ARENA_W - C.TANK_RADIUS + 1e-9


def test_short_history_fallback():
    pos, head, rays = straight_line(2)
    gru = load_predictor("gru")
    p = gru.predict(pos, head, rays)
    assert p.shape == (C.HORIZON, 2) and np.isfinite(p).all()


# ------------------------------------------------------------------ trained networks
@pytest.mark.skipif(not (MODELS_DIR / "gru.pt").exists(), reason="trained model not available")
def test_trained_gru_loads_and_predicts():
    p = load_predictor("gru")
    assert isinstance(p, NeuralPredictor)
    pos, head, rays = straight_line(H1)
    out = p.predict(pos, head, rays)
    assert out.shape == (C.HORIZON, 2) and np.isfinite(out).all()
    # on a perfectly straight line far from walls the GRU continues roughly straight ahead
    truth = straight_line(H1 + C.HORIZON)[0][H1:]
    assert np.linalg.norm(out[9] - truth[9]) < 25.0


@pytest.mark.skipif(not (MODELS_DIR / "gru.pt").exists(), reason="trained model not available")
def test_trained_gru_keeps_a_standing_tank_in_place():
    from tankgame.sim import World
    w = World()
    p = load_predictor("gru")
    for x, y, h in ((300, 300, 0.0), (600, 450, 2.0), (150, 600, -1.0)):
        pos = np.tile([[x, y]], (H1, 1))
        rays = np.tile(w.rays(x, y, h), (H1, 1))
        out = p.predict(pos, np.full(H1, h), rays)
        assert np.linalg.norm(out[14] - (x, y)) < 15.0      # 0.5 s ahead
        assert np.linalg.norm(out[-1] - (x, y)) < 60.0      # 1.5 s ahead


@pytest.mark.skipif(not (MODELS_DIR / "gru.pt").exists(), reason="trained model not available")
def test_trained_gru_beats_baselines_on_fresh_data():
    from tankgame.data import WINDOW, generate_episode, split_windows, window_starts
    eps = [generate_episode(seed=900_000 + i, seconds=20.0) for i in range(12)]
    traj = np.concatenate(eps)
    offsets = np.concatenate([[0], np.cumsum([len(e) for e in eps])])
    starts = window_starts(offsets, stride=5)
    win = traj[starts[:, None] + np.arange(WINDOW)]
    pos, head, rays, fut, _ = split_windows(win)
    errs = {}
    for name in ("cv", "ctrv", "gru"):
        pred = load_predictor(name).predict_batch(pos, head, rays)
        errs[name] = np.linalg.norm(pred - fut, axis=-1).mean()
    assert errs["gru"] < 0.85 * errs["cv"], errs
    assert errs["gru"] < 0.85 * errs["ctrv"], errs


# ------------------------------------------------------------------ aiming
def test_intercept_static_target():
    point, t = intercept_point((0, 0), (300, 400), np.tile([[300.0, 400.0]], (C.HORIZON, 1)))
    assert np.allclose(point, (300, 400))
    assert t == pytest.approx((500 - C.MUZZLE_LENGTH) / C.BULLET_SPEED, abs=C.MODEL_DT)


def test_cv_intercept_formula():
    g, p, v = (0.0, 0.0), (400.0, 0.0), (0.0, 100.0)
    t = solve_cv_intercept(g, p, v, speed=C.BULLET_SPEED)
    hit = np.array(p) + np.array(v) * t
    assert np.hypot(*hit) == pytest.approx(C.BULLET_SPEED * t)


def test_intercept_moving_target_actually_hits():
    """Simulate the bullet: with an exact prediction the shot must hit a moving target."""
    pos, head, rays = straight_line(H1 + 400, x0=300, y0=150, angle=0.6, step=4.0)
    shooter = np.array([700.0, 600.0])
    hist, fut = pos[:H1], pos[H1:H1 + C.HORIZON]
    point, _ = intercept_point(shooter, hist[-1], fut)
    a = math.atan2(point[1] - shooter[1], point[0] - shooter[0])
    d = np.array([math.cos(a), math.sin(a)])
    hit = False
    for k in range(1, 200):  # physics ticks (2 per model step)
        b = shooter + d * (C.MUZZLE_LENGTH + C.BULLET_SPEED * k * C.SIM_DT)
        target = hist[-1] + (pos[H1] - pos[H1 - 1]) * (k / C.MODEL_STRIDE)
        if np.linalg.norm(b - target) < C.TANK_RADIUS + C.BULLET_RADIUS:
            hit = True
            break
    assert hit
