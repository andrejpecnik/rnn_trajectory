"""Command line interface:  python -m tankgame <command> [options]

    play        play against the RNN-aiming computer tank
    watch       watch two AI tanks with different predictors (or run a headless tournament)
    generate    generate a trajectory dataset
    train       train a neural predictor (gru / lstm / mlp)
    evaluate    compare all predictors (prediction error, hit rate, tournaments) and draw plots
"""

from __future__ import annotations

import argparse
import json
import sys

PREDICTORS = ("static", "cv", "ctrv", "mlp", "lstm", "gru")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tankgame", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("play", help="play against the computer")
    sp.add_argument("--enemy", default="gru", choices=PREDICTORS, help="predictor used by the enemy to aim")
    sp.add_argument("--assist", default="gru", choices=PREDICTORS, help="predictor used for your aim assist (T)")
    sp.add_argument("--difficulty", default="normal", choices=("easy", "normal", "hard"),
                    help="enemy reload time and aim error (hard = no handicap)")
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--record", action="store_true", help="record your trajectories to data/recorded/")
    sp.add_argument("--autopilot", action="store_true", help="let an AI drive your tank (demo)")
    sp.add_argument("--headless", action="store_true", help="no window (for tests/screenshots)")
    sp.add_argument("--frames", type=int, default=None, help="stop after N frames")
    sp.add_argument("--screenshot", default=None, help="save the last frame to this PNG")

    sw = sub.add_parser("watch", help="AI vs AI")
    sw.add_argument("--blue", default="gru", choices=PREDICTORS)
    sw.add_argument("--red", default="cv", choices=PREDICTORS)
    sw.add_argument("--seed", type=int, default=None)
    sw.add_argument("--speed", type=int, default=1)
    sw.add_argument("--tournament", type=int, default=0, metavar="ROUNDS",
                    help="run ROUNDS rounds without graphics as fast as possible and print the result")
    sw.add_argument("--headless", action="store_true")
    sw.add_argument("--frames", type=int, default=None)
    sw.add_argument("--screenshot", default=None)

    sg = sub.add_parser("generate", help="generate a dataset")
    sg.add_argument("--episodes", type=int, default=1600)
    sg.add_argument("--seconds", type=float, default=40.0)
    sg.add_argument("--seed", type=int, default=1)
    sg.add_argument("--behavior", default=None, help="force one behaviour (default: random mix)")
    sg.add_argument("--out", default="data/train.npz")
    sg.add_argument("--workers", type=int, default=None)

    st = sub.add_parser("train", help="train a neural predictor")
    st.add_argument("--model", default="gru", choices=("gru", "lstm", "mlp"))
    st.add_argument("--data", default="data/train.npz")
    st.add_argument("--val", default="data/val.npz")
    st.add_argument("--extra", nargs="*", default=[], help="additional datasets, e.g. data/recorded/*.npz")
    st.add_argument("--epochs", type=int, default=40)
    st.add_argument("--steps", type=int, default=1000, help="optimisation steps per epoch")
    st.add_argument("--batch", type=int, default=512)
    st.add_argument("--lr", type=float, default=2e-3)
    st.add_argument("--hidden", type=int, default=128)
    st.add_argument("--layers", type=int, default=2)
    st.add_argument("--device", default="auto")
    st.add_argument("--out", default=None, help="output directory (default: models/)")

    se = sub.add_parser("evaluate", help="evaluate and compare all predictors")
    se.add_argument("--test", default="data/test.npz")
    se.add_argument("--windows", type=int, default=30000)
    se.add_argument("--scenarios", type=int, default=400, help="shooting-range scenarios")
    se.add_argument("--rounds", type=int, default=60, help="rounds per tournament (0 = skip)")
    se.add_argument("--out", default="docs")

    a = p.parse_args(argv)

    if a.cmd == "play":
        from .game import play
        res = play(mode="play", enemy=a.enemy, assist=a.assist, difficulty=a.difficulty, seed=a.seed,
                   record=a.record,
                   autopilot=a.autopilot, headless=a.headless, max_frames=a.frames, screenshot=a.screenshot)
        print(json.dumps(res))
    elif a.cmd == "watch":
        if a.tournament:
            import numpy as np
            from .ai import AIController
            from .arena import run_tournament
            from .behaviors import GAME_BEHAVIORS, MixedBehavior
            from .predictors import load_predictor
            seed = a.seed or 0
            rng = np.random.default_rng(seed)
            res = run_tournament(AIController(MixedBehavior(rng, pool=GAME_BEHAVIORS), load_predictor(a.blue)),
                                 AIController(MixedBehavior(rng, pool=GAME_BEHAVIORS), load_predictor(a.red)),
                                 rounds=a.tournament, seed=seed, log=print)
            res["blue"], res["red"] = a.blue, a.red
            print(json.dumps(res, indent=2))
        else:
            from .game import play
            res = play(mode="watch", blue=a.blue, red=a.red, seed=a.seed, speed=a.speed,
                       headless=a.headless, max_frames=a.frames, screenshot=a.screenshot)
            print(json.dumps(res))
    elif a.cmd == "generate":
        from .data import generate_dataset
        print(json.dumps(generate_dataset(a.episodes, a.seconds, a.seed, a.out, a.workers, a.behavior)))
    elif a.cmd == "train":
        from .predictors import MODELS_DIR
        from .train import train
        s = train(a.model, a.data, a.val, a.extra, a.out or MODELS_DIR, a.epochs, a.steps, a.batch, a.lr,
                  a.hidden, a.layers, a.device)
        print(f"best validation ADE: {s['best_val_ade']:.2f} px")
    elif a.cmd == "evaluate":
        from .evaluate import run_all
        run_all(test_path=a.test, n_windows=a.windows, n_scenarios=a.scenarios, rounds=a.rounds, out_dir=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
