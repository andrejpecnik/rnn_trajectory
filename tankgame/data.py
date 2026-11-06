"""Generation and loading of the trajectory dataset.

A dataset file (.npz) contains:
    traj     float32 (N, 7)  rows [x, y, heading, ray_front, ray_left, ray_right, behavior_id]
                              sampled at the model rate (every MODEL_STRIDE physics ticks)
    offsets  int64  (E+1,)   episode boundaries in `traj` (episode e = traj[offsets[e]:offsets[e+1]])
    meta     str             JSON with generation parameters

Training windows are cut on the fly: HISTORY+1 samples of the past and
HORIZON samples of the future, always inside one episode.
"""

from __future__ import annotations

import json
import math
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import config as C
from .behaviors import active_behavior_id, make_behavior, random_behavior
from .sim import Tank, World

WINDOW = C.HISTORY + 1 + C.HORIZON


def generate_episode(seed: int, seconds: float = 40.0, behavior: str | None = None) -> np.ndarray:
    """Simulate one tank driven by a random behaviour and record its trajectory."""
    rng = np.random.default_rng(seed)
    world = World()
    x, y = world.random_free_position(rng)
    tank = Tank(x, y, heading=rng.uniform(-math.pi, math.pi),
                speed=rng.uniform(0.0, C.TANK_MAX_SPEED))
    beh = make_behavior(behavior, rng) if behavior else random_behavior(rng)
    n_ticks = int(round(seconds * C.SIM_HZ))
    rows = []
    for k in range(n_ticks):
        if k % C.MODEL_STRIDE == 0:
            r = world.rays(tank.x, tank.y, tank.heading)
            rows.append((tank.x, tank.y, tank.heading, r[0], r[1], r[2], active_behavior_id(beh)))
        throttle, turn = beh.act(tank, world)
        world.step_tank(tank, throttle, turn)
    return np.asarray(rows, dtype=np.float32)


def _worker(args):
    return generate_episode(*args)


def generate_dataset(n_episodes: int, seconds: float, seed: int, out: str | os.PathLike,
                     workers: int | None = None, behavior: str | None = None) -> dict:
    t0 = time.time()
    jobs = [(seed * 100_003 + i, seconds, behavior) for i in range(n_episodes)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1 and n_episodes > 8:
        with Pool(workers) as pool:
            episodes = pool.map(_worker, jobs, chunksize=8)
    else:
        episodes = [_worker(j) for j in jobs]
    offsets = np.zeros(len(episodes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(e) for e in episodes])
    traj = np.concatenate(episodes, axis=0)
    meta = dict(n_episodes=n_episodes, seconds=seconds, seed=seed, behavior=behavior,
                sim_hz=C.SIM_HZ, model_stride=C.MODEL_STRIDE, samples=int(len(traj)),
                generated_in_s=round(time.time() - t0, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, traj=traj, offsets=offsets, meta=json.dumps(meta))
    return meta


def window_starts(offsets: np.ndarray, stride: int = 1) -> np.ndarray:
    """All indices i such that traj[i : i+WINDOW] lies inside a single episode."""
    starts = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b - a >= WINDOW:
            starts.append(np.arange(a, b - WINDOW + 1, stride, dtype=np.int64))
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def load_dataset(path: str | os.PathLike) -> dict:
    d = np.load(path, allow_pickle=False)
    traj, offsets = d["traj"], d["offsets"]
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return dict(traj=traj, offsets=offsets, meta=meta, starts=window_starts(offsets))


def merge_datasets(datasets: list[dict]) -> dict:
    trajs, offs, base = [], [np.zeros(1, dtype=np.int64)], 0
    for d in datasets:
        trajs.append(d["traj"])
        offs.append(d["offsets"][1:] + base)
        base += len(d["traj"])
    traj = np.concatenate(trajs)
    offsets = np.concatenate(offs)
    return dict(traj=traj, offsets=offsets, meta={"merged": len(datasets)}, starts=window_starts(offsets))


def split_windows(window: np.ndarray):
    """window (..., WINDOW, 7) -> pos (...,H+1,2), head (...,H+1), rays (...,H+1,3), future (...,K,2), beh."""
    h = C.HISTORY + 1
    past, fut = window[..., :h, :], window[..., h:, :]
    return past[..., 0:2], past[..., 2], past[..., 3:6], fut[..., 0:2], past[..., -1, 6]


def save_recording(path: str | os.PathLike, segments: list[list]) -> int:
    """Store recorded (e.g. human player) trajectories in the dataset format.

    `segments` is a list of episodes, each a list of rows. Returns the number of samples saved.
    """
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return 0
    traj = np.concatenate([np.asarray(s, dtype=np.float32) for s in segments])
    offsets = np.zeros(len(segments) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in segments])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, traj=traj, offsets=offsets, meta=json.dumps({"recording": True}))
    return int(len(traj))
