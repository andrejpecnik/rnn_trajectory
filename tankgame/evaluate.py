"""Evaluation of all predictors and generation of the figures used in the documentation.

Three experiments:
1. Prediction accuracy on unseen test trajectories (ADE / FDE / error over horizon / per behaviour).
2. Shooting range - a static shooter fires at a moving target using each predictor
   (only the aim differs, the target trajectories are identical) -> hit rate.
3. Tournaments - full games of GRU-aiming tank vs. every other predictor.
"""

from __future__ import annotations

import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import config as C
from .aiming import aim_angle, intercept_point
from .behaviors import BEHAVIOR_NAMES, active_behavior_id, random_behavior
from .data import WINDOW, load_dataset, split_windows
from .predictors import LABELS, load_predictor
from .sim import Tank, World

ORDER = ("gru", "lstm", "mlp", "ctrv", "cv", "static")
# fixed categorical slots (validated reference palette) - colour follows the predictor
COLORS = {"gru": "#2a78d6", "lstm": "#eb6834", "mlp": "#1baf7a",
          "ctrv": "#eda100", "cv": "#e87ba4", "static": "#008300"}
SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def available_predictors(device: str = "cpu") -> dict:
    preds = {}
    for name in ORDER:
        p = load_predictor(name, device=device)
        if p.name == name:      # skip neural models that are not trained (fallback)
            preds[name] = p
    return preds


# --------------------------------------------------------------------------- experiment 1
def prediction_metrics(test_path: str, predictors: dict, n_windows: int = 30000, seed: int = 0) -> dict:
    d = load_dataset(test_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["starts"]), size=min(n_windows, len(d["starts"])), replace=False)
    win = d["traj"][d["starts"][idx][:, None] + np.arange(WINDOW)]
    pos, head, rays, fut, beh = split_windows(win)
    res = {}
    for name, p in predictors.items():
        preds = np.concatenate([p.predict_batch(pos[i:i + 4096], head[i:i + 4096], rays[i:i + 4096])
                                for i in range(0, len(pos), 4096)])
        err = np.linalg.norm(preds - fut, axis=-1)          # (N, K)
        per_beh = {}
        for b in np.unique(beh).astype(int):
            m = beh == b
            per_beh[BEHAVIOR_NAMES.get(b, str(b))] = float(err[m].mean())
        res[name] = {"ade": float(err.mean()), "fde": float(err[:, -1].mean()),
                     "err_1s": float(err[:, int(1.0 / C.MODEL_DT) - 1].mean()),
                     "horizon_curve": err.mean(axis=0).tolist(), "per_behavior": per_beh}
    res["_examples"] = {"pos": pos[:400].tolist(), "fut": fut[:400].tolist(),
                        "head": head[:400].tolist(), "rays": rays[:400].tolist()}
    res["_n_windows"] = int(len(idx))
    return res


# --------------------------------------------------------------------------- experiment 2
def _simulate_target(seed: int, seconds: float):
    rng = np.random.default_rng(10_000 + seed)
    world = World()
    x, y = world.random_free_position(rng)
    tank = Tank(x, y, heading=rng.uniform(-math.pi, math.pi), speed=rng.uniform(0, C.TANK_MAX_SPEED))
    beh = random_behavior(rng)
    n = int(seconds * C.SIM_HZ)
    pos = np.zeros((n, 2))
    head = np.zeros(n)
    rays = np.zeros((n, 3))
    bid = np.zeros(n, dtype=int)
    for k in range(n):
        pos[k] = (tank.x, tank.y)
        head[k] = tank.heading
        bid[k] = active_behavior_id(beh)
        if k % C.MODEL_STRIDE == 0:
            rays[k] = world.rays(tank.x, tank.y, tank.heading)
        world.step_tank(tank, *beh.act(tank, world))
    # shooter: a free spot 300-600 units away from the target's mean position
    centre = pos.mean(axis=0)
    for _ in range(500):
        sx, sy = world.random_free_position(rng)
        if 300 <= math.dist((sx, sy), centre) <= 600:
            break
    return pos, head, rays, bid, (sx, sy)


def shooting_range(predictors: dict, n_scenarios: int = 400, seconds: float = 12.0) -> dict:
    h = C.HISTORY + 1
    shots = []      # (scenario, tick)
    scen = []
    for s in range(n_scenarios):
        pos, head, rays, bid, shooter = _simulate_target(s, seconds)
        scen.append((pos, shooter))
        first = h * C.MODEL_STRIDE
        for tick in range(first, len(pos) - int(C.BULLET_LIFETIME * C.SIM_HZ) - 1, int(C.RELOAD_TIME * C.SIM_HZ)):
            tick -= tick % C.MODEL_STRIDE
            sel = np.arange(tick - (h - 1) * C.MODEL_STRIDE, tick + 1, C.MODEL_STRIDE)
            shots.append((s, tick, pos[sel], head[sel], rays[sel], bid[tick]))
    hist_pos = np.stack([x[2] for x in shots])
    hist_head = np.stack([x[3] for x in shots])
    hist_rays = np.stack([x[4] for x in shots])
    beh = np.array([x[5] for x in shots])
    life = int(C.BULLET_LIFETIME * C.SIM_HZ)
    j = np.arange(1, life + 1)
    res = {}
    for name, p in predictors.items():
        preds = np.concatenate([p.predict_batch(hist_pos[i:i + 4096], hist_head[i:i + 4096], hist_rays[i:i + 4096])
                                for i in range(0, len(hist_pos), 4096)])
        hits = np.zeros(len(shots), dtype=bool)
        for k, (s, tick, *_rest) in enumerate(shots):
            tpos, shooter = scen[s]
            point, _ = intercept_point(shooter, tpos[tick], preds[k])
            a = aim_angle(shooter, point)
            d = np.array([math.cos(a), math.sin(a)])
            muzzle = np.asarray(shooter) + d * C.MUZZLE_LENGTH
            bullet = muzzle[None] + d[None] * (C.BULLET_SPEED * C.SIM_DT) * j[:, None]
            target = tpos[tick + j]
            hits[k] = bool((np.linalg.norm(bullet - target, axis=1) < C.TANK_RADIUS + C.BULLET_RADIUS).any())
        per_beh = {BEHAVIOR_NAMES.get(int(b), str(b)): float(hits[beh == b].mean()) for b in np.unique(beh)}
        res[name] = {"hit_rate": float(hits.mean()), "shots": int(len(hits)), "per_behavior": per_beh}
    return res


# --------------------------------------------------------------------------- experiment 3
def _tournament_job(args):
    a, b, rounds, seed = args
    import torch
    torch.set_num_threads(1)
    from .ai import AIController
    from .arena import run_tournament
    from .behaviors import GAME_BEHAVIORS, MixedBehavior
    rng = np.random.default_rng(seed)
    r = run_tournament(AIController(MixedBehavior(rng, pool=GAME_BEHAVIORS), load_predictor(a)),
                       AIController(MixedBehavior(rng, pool=GAME_BEHAVIORS), load_predictor(b)), rounds=rounds, seed=seed)
    r["a"], r["b"] = a, b
    return r


def tournaments(opponents: list[str], rounds: int = 60, champion: str = "gru") -> list[dict]:
    jobs = [(champion, o, rounds, 100 + i) for i, o in enumerate(opponents)]
    with Pool(min(len(jobs), 8)) as pool:
        return pool.map(_tournament_job, jobs)


# --------------------------------------------------------------------------- figures
def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRIDC, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _bars(ax, names, values, fmt):
    x = np.arange(len(names))
    bars = ax.bar(x, values, width=0.62, color=[COLORS[n] for n in names], edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, [LABELS[n].replace(" (", "\n(") for n in names], fontsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt(v), ha="center", va="bottom",
                fontsize=9, color=INK)


def plot_training_curves(baselines: dict, img: Path):
    """Validation ADE per epoch for the neural models, baselines as dashed reference lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for n in ("gru", "lstm", "mlp"):
        f = Path("models") / f"{n}_history.json"
        if f.exists():
            hist = json.loads(f.read_text())["history"]
            ade = [r["val_ade"] for r in hist]
            ax.plot([r["epoch"] for r in hist], ade, color=COLORS[n], linewidth=2,
                    label=f"{LABELS[n]} - best {min(ade):.2f} px")
    for n, v in baselines.items():
        ax.axhline(v, color=COLORS[n], linewidth=2, linestyle="--", label=f"{LABELS[n]} (baseline) - {v:.1f} px")
    _style(ax, "Validation ADE during training (lower is better)", "epoch", "ADE [px]")
    ax.legend(frameon=False, fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.6))
    fig.tight_layout()
    fig.savefig(img / "training_curves.png", dpi=130)
    plt.close(fig)


def make_figures(pred: dict, shoot: dict, tours: list, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    names = [n for n in ORDER if n in pred]
    plt.rcParams.update({"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9})

    # 0) one sample trajectory per behaviour
    from .behaviors import BEHAVIORS
    from .data import generate_episode
    world = World()
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6))
    for ax, name in zip(axs.flat, BEHAVIORS):
        e = generate_episode(seed=11, seconds=20.0, behavior=name)
        for o in world.obstacles:
            ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, color="#c9c8c2"))
        ax.plot(e[:, 0], e[:, 1], color=COLORS["gru"], linewidth=1.6)
        ax.plot(e[0, 0], e[0, 1], "o", color=INK, markersize=5)
        ax.set_xlim(0, C.ARENA_W)
        ax.set_ylim(C.ARENA_H, 0)
        ax.set_aspect("equal")
        _style(ax, f"{name} (20 s)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sample trajectories of each behaviour (dot = start)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(img / "behaviors.png", dpi=110)
    plt.close(fig)

    # 1) training curves
    plot_training_curves({n: pred[n]["ade"] for n in ("cv", "ctrv") if n in pred}, img)

    # 2) error vs prediction horizon
    fig, ax = plt.subplots(figsize=(7.5, 4))
    t = (np.arange(C.HORIZON) + 1) * C.MODEL_DT
    for n in names:
        curve = pred[n]["horizon_curve"]
        ax.plot(t, curve, color=COLORS[n], linewidth=2, label=LABELS[n])
    _style(ax, "Mean position error vs. how far ahead we predict", "prediction horizon [s]", "error [px]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(img / "error_vs_horizon.png", dpi=130)
    plt.close(fig)

    # 3) ADE + FDE bars
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    _bars(axs[0], names, [pred[n]["ade"] for n in names], lambda v: f"{v:.1f}")
    _style(axs[0], "ADE - average error over 1.5 s [px]")
    _bars(axs[1], names, [pred[n]["fde"] for n in names], lambda v: f"{v:.1f}")
    _style(axs[1], "FDE - error at 1.5 s [px]")
    fig.tight_layout()
    fig.savefig(img / "ade_fde.png", dpi=130)
    plt.close(fig)

    # 4) hit rate
    if shoot:
        sn = [n for n in ORDER if n in shoot]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _bars(ax, sn, [100 * shoot[n]["hit_rate"] for n in sn], lambda v: f"{v:.1f} %")
        _style(ax, "Shooting range - hit rate (higher is better)", None, "hits [%]")
        fig.tight_layout()
        fig.savefig(img / "hit_rate.png", dpi=130)
        plt.close(fig)

    # 5) per-behaviour ADE (small multiples, one panel per behaviour)
    behs = [b for b in ("circle", "zigzag", "sine", "waypoint", "dodge", "stopgo") if b in pred[names[0]]["per_behavior"]]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for ax, b in zip(axs.flat, behs):
        vals = [pred[n]["per_behavior"].get(b, np.nan) for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, width=0.62, color=[COLORS[n] for n in names], edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(x, [n.upper() for n in names], fontsize=8)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        _style(ax, b)
    axs[0, 0].set_ylabel("ADE [px]", color=INK2)
    axs[1, 0].set_ylabel("ADE [px]", color=INK2)
    fig.suptitle("Prediction error by type of opponent movement", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(img / "per_behavior.png", dpi=130)
    plt.close(fig)

    # 6) example predictions
    ex = pred["_examples"]
    pos, fut, head, rays = (np.asarray(ex[k]) for k in ("pos", "fut", "head", "rays"))
    rng = np.random.default_rng(3)
    moving = [i for i in range(len(pos)) if np.linalg.norm(fut[i, -1] - pos[i, -1]) > 120]
    pick = rng.choice(moving, size=6, replace=False)
    world = World()
    show = [n for n in ("gru", "cv", "ctrv") if n in pred]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, i in zip(axs.flat, pick):
        allp = np.vstack([pos[i], fut[i]])
        cx, cy = allp.mean(axis=0)
        half = max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])) / 2 + 60
        for o in world.obstacles:
            ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, color="#c9c8c2"))
        ax.plot(pos[i, :, 0], pos[i, :, 1], color=INK2, linewidth=2, label="observed past (1 s)")
        ax.plot(fut[i, :, 0], fut[i, :, 1], color=INK, linewidth=2, linestyle=":", label="true future (1.5 s)")
        for n in show:
            p = load_predictor(n).predict(pos[i], head[i], rays[i])
            ax.plot(p[:, 0], p[:, 1], color=COLORS[n], linewidth=2, label=LABELS[n])
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.set_aspect("equal")
        _style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Example predictions on unseen test trajectories", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(img / "examples.png", dpi=130)
    plt.close(fig)

    # 7) tournament results
    if tours:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        opp = [r["b"] for r in tours]
        x = np.arange(len(opp))
        win = [100 * r["score"][0] / max(1, r["rounds"]) for r in tours]
        bars = ax.bar(x, win, width=0.62, color=COLORS["gru"], edgecolor=SURFACE, linewidth=2)
        for b, r in zip(bars, tours):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['score'][0]}:{r['score'][1]}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(50, color=INK2, linewidth=1, linestyle="--")
        ax.set_xticks(x, [f"vs {LABELS[o]}".replace(" (", "\n(") for o in opp], fontsize=8)
        _style(ax, "Tournaments - share of rounds won by the GRU tank", None, "rounds won [%]")
        fig.tight_layout()
        fig.savefig(img / "tournaments.png", dpi=130)
        plt.close(fig)


def results_markdown(pred: dict, shoot: dict, tours: list) -> str:
    names = [n for n in ORDER if n in pred]
    lines = ["| Predictor | ADE [px] | Error at 1 s [px] | FDE 1.5 s [px] | Hit rate |",
             "|---|---:|---:|---:|---:|"]
    for n in names:
        hr = f"{100 * shoot[n]['hit_rate']:.1f} %" if n in shoot else "-"
        lines.append(f"| {LABELS[n]} | {pred[n]['ade']:.1f} | {pred[n]['err_1s']:.1f} | {pred[n]['fde']:.1f} | {hr} |")
    md = "\n".join(lines)
    if tours:
        md += "\n\n| GRU tank vs. | Rounds | GRU wins | Opponent wins | Draws | GRU accuracy | Opponent accuracy |\n"
        md += "|---|---:|---:|---:|---:|---:|---:|\n"
        for r in tours:
            md += (f"| {LABELS[r['b']]} | {r['rounds']} | {r['score'][0]} | {r['score'][1]} | {r['draws']} | "
                   f"{100 * r['accuracy'][0]:.1f} % | {100 * r['accuracy'][1]:.1f} % |\n")
    return md


def run_all(test_path: str = "data/test.npz", n_windows: int = 30000, n_scenarios: int = 400,
            rounds: int = 60, out_dir: str = "docs") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = _device()
    preds = available_predictors(dev)
    print(f"predictors: {', '.join(preds)}  (device {dev})")
    pred = prediction_metrics(test_path, preds, n_windows)
    for n in preds:
        print(f"  {LABELS[n]:24s} ADE {pred[n]['ade']:6.2f}  1s {pred[n]['err_1s']:6.2f}  FDE {pred[n]['fde']:6.2f}")
    shoot = shooting_range(preds, n_scenarios) if n_scenarios else {}
    for n in shoot:
        print(f"  {LABELS[n]:24s} hit rate {100 * shoot[n]['hit_rate']:5.1f} %  ({shoot[n]['shots']} shots)")
    tours = []
    if rounds:
        tours = tournaments([n for n in ORDER if n != "gru" and n in preds], rounds)
        for r in tours:
            print(f"  GRU vs {LABELS[r['b']]:24s} {r['score'][0]}:{r['score'][1]} (draws {r['draws']})  "
                  f"accuracy {100 * r['accuracy'][0]:.1f} % vs {100 * r['accuracy'][1]:.1f} %")
    make_figures(pred, shoot, tours, out)
    summary = {k: v for k, v in pred.items() if not k.startswith("_")}
    result = {"prediction": summary, "n_windows": pred["_n_windows"], "shooting_range": shoot,
              "tournaments": tours, "seconds": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "results.md").write_text(results_markdown(pred, shoot, tours))
    print(f"results written to {out}/results.json, {out}/results.md and {out}/img/ ({result['seconds']} s)")
    return result
