import json

import numpy as np
import pytest

from tankgame import config as C
from tankgame.ai import AIController, TrajectoryTracker
from tankgame.arena import Match, run_tournament
from tankgame.behaviors import MixedBehavior
from tankgame.predictors import load_predictor
from tankgame.sim import Tank, World


class Idle:
    def reset(self):
        pass

    def control(self, tank, tracker, world, dt):
        return 0.0, 0.0, None, False


def test_tracker_samples_at_model_rate_and_scores_predictions():
    w = World()
    t = Tank(200, 200, heading=0.0)
    tr = TrajectoryTracker()
    taken = 0
    for k in range(200):
        taken += tr.observe(t, w)
        if k == 70:
            # a perfect "prediction" of a static tank -> zero error
            tr.register_prediction(np.tile([[t.x, t.y]], (C.HORIZON, 1)))
    assert taken == 100
    pos, head, rays = tr.history()
    assert pos.shape == (C.HISTORY + 1, 2) and rays.shape == (C.HISTORY + 1, 3)
    assert tr.mean_error == pytest.approx(0.0)


def test_ai_hits_a_stationary_target():
    m = Match([AIController(MixedBehavior(np.random.default_rng(0)), load_predictor("gru")), Idle()], seed=1)
    for _ in range(int(20 * C.SIM_HZ)):
        m.step()
        if m.rounds:
            break
    assert m.shots[0] > 0
    assert m.hits[0] >= 3
    assert m.accuracy(0) > 0.5


def test_match_round_flow():
    m = Match([Idle(), Idle()], seed=3)
    m.tanks[1].hp = 0
    m.step()
    assert m.round_over and m.score == [1, 0] and m.winner == 0
    for _ in range(int(m.ROUND_PAUSE * C.SIM_HZ) + 2):
        m.step()
    assert not m.round_over
    assert all(t.hp == C.TANK_HP for t in m.tanks)
    assert np.hypot(m.tanks[0].x - m.tanks[1].x, m.tanks[0].y - m.tanks[1].y) > 400


def test_headless_tournament_runs():
    rng = np.random.default_rng(5)
    r = run_tournament(AIController(MixedBehavior(rng), load_predictor("gru")),
                       AIController(MixedBehavior(rng), load_predictor("cv")), rounds=3, seed=5)
    assert r["rounds"] == 3
    assert sum(r["score"]) + r["draws"] == 3
    assert r["shots"][0] > 0 and r["shots"][1] > 0


@pytest.mark.parametrize("mode", ["play", "watch"])
def test_pygame_frontend_headless(mode, tmp_path):
    from tankgame.game import GameApp
    shot = tmp_path / f"{mode}.png"
    app = GameApp(mode=mode, autopilot=True, headless=True, seed=11)
    app.show_assist = True
    app.help_timer = 1e9          # also render the help overlay
    res = app.run(max_frames=420, screenshot=str(shot))
    assert res["frames"] == 420
    assert shot.exists() and shot.stat().st_size > 10_000
    assert sum(res["shots"]) > 0


def test_human_controller_input_path():
    from tankgame.game import GameApp
    app = GameApp(mode="play", headless=True, seed=2)
    assert app.human is not None
    app.human.throttle, app.human.turn, app.human.aim, app.human.fire = 1.0, 0.5, 0.0, True
    x0 = app.match.tanks[0].x, app.match.tanks[0].y
    for _ in range(90):
        app.update(C.SIM_DT)
        app.draw()
    moved = np.hypot(app.match.tanks[0].x - x0[0], app.match.tanks[0].y - x0[1])
    assert moved > 50
    assert app.match.shots[0] >= 1
    app.pg.quit()


def test_recording_is_saved(tmp_path, monkeypatch):
    from tankgame.game import GameApp
    monkeypatch.chdir(tmp_path)
    app = GameApp(mode="play", headless=True, seed=4, record=True)
    app.human.throttle = 1.0
    res = app.run(max_frames=200)
    assert res["recording"] is not None
    d = np.load(res["recording"])
    assert d["traj"].shape[1] == 7 and len(d["traj"]) > 50


def test_cli_help_and_watch_tournament(capsys):
    from tankgame.__main__ import main
    assert main(["watch", "--tournament", "1", "--blue", "ctrv", "--red", "cv", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    res = json.loads(out[out.index("{"):])
    assert res["rounds"] == 1
