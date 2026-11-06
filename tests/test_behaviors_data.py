import numpy as np
import pytest

from tankgame import config as C
from tankgame.behaviors import BEHAVIORS
from tankgame.data import (WINDOW, generate_dataset, generate_episode, load_dataset, merge_datasets,
                           save_recording, split_windows, window_starts)
from tankgame.sim import World


@pytest.mark.parametrize("name", list(BEHAVIORS))
def test_behavior_moves_and_stays_in_free_space(name):
    ep = generate_episode(seed=3, seconds=30.0, behavior=name)
    assert ep.shape == (30 * C.SIM_HZ // C.MODEL_STRIDE, 7)
    w = World()
    assert all(w.is_free(x, y, C.TANK_RADIUS - 0.5) for x, y in ep[:, :2])
    path = np.hypot(*np.diff(ep[:, :2], axis=0).T).sum()
    if name == "idle":
        assert path < 800, "idle behaviour should mostly stand still"
    else:
        assert path > 800, f"{name} barely moved ({path:.0f})"
    step = np.hypot(*np.diff(ep[:, :2], axis=0).T).max()
    assert step <= C.STEP_SCALE + 1e-3          # never faster than the max speed


def test_generation_is_deterministic():
    a = generate_episode(seed=42, seconds=5.0)
    b = generate_episode(seed=42, seconds=5.0)
    np.testing.assert_array_equal(a, b)
    c = generate_episode(seed=43, seconds=5.0)
    assert not np.array_equal(a, c)


def test_dataset_roundtrip(tmp_path):
    meta = generate_dataset(4, 6.0, seed=7, out=tmp_path / "d.npz", workers=1)
    assert meta["n_episodes"] == 4
    d = load_dataset(tmp_path / "d.npz")
    assert d["traj"].shape == (4 * 6 * C.SIM_HZ // C.MODEL_STRIDE, 7)
    assert len(d["offsets"]) == 5
    # every window lies inside a single episode
    for s in d["starts"]:
        e = np.searchsorted(d["offsets"], s, side="right") - 1
        assert s + WINDOW <= d["offsets"][e + 1]
    win = d["traj"][d["starts"][:8][:, None] + np.arange(WINDOW)]
    pos, head, rays, fut, beh = split_windows(win)
    assert pos.shape == (8, C.HISTORY + 1, 2) and fut.shape == (8, C.HORIZON, 2)
    assert rays.shape == (8, C.HISTORY + 1, 3) and head.shape == (8, C.HISTORY + 1)


def test_window_starts_skip_short_episodes():
    offsets = np.array([0, WINDOW - 1, 2 * WINDOW + 10])
    s = window_starts(offsets)
    assert s.min() == WINDOW - 1 and s.max() == 2 * WINDOW + 10 - WINDOW


def test_recording_and_merge(tmp_path):
    rows = [[(1.0 * i, 2.0, 0.0, 1, 1, 1, 0)] * 1 for i in range(3)]
    seg1 = [r[0] for r in rows] * 40
    n = save_recording(tmp_path / "r.npz", [seg1, [], seg1])
    assert n == 2 * len(seg1)
    rec = load_dataset(tmp_path / "r.npz")
    assert len(rec["offsets"]) == 3
    generate_dataset(2, 4.0, seed=1, out=tmp_path / "g.npz", workers=1)
    merged = merge_datasets([load_dataset(tmp_path / "g.npz"), rec])
    assert len(merged["traj"]) == len(rec["traj"]) + 2 * 4 * C.SIM_HZ // C.MODEL_STRIDE
    assert merged["offsets"][-1] == len(merged["traj"])
