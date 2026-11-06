import json

import numpy as np

from tankgame.data import generate_dataset
from tankgame.predictors import NeuralPredictor
from tankgame.train import train


def test_training_smoke_reduces_error_and_saves_checkpoint(tmp_path):
    generate_dataset(30, 12.0, seed=11, out=tmp_path / "train.npz", workers=1)
    generate_dataset(6, 12.0, seed=12, out=tmp_path / "val.npz", workers=1)
    logs = []
    s = train("gru", str(tmp_path / "train.npz"), str(tmp_path / "val.npz"), out_dir=tmp_path,
              epochs=3, steps_per_epoch=40, batch_size=128, hidden=32, layers=1, log=logs.append)
    hist = s["history"]
    assert len(hist) == 3
    assert hist[-1]["val_ade"] < hist[0]["val_ade"] * 1.05
    assert (tmp_path / "gru.pt").exists()
    assert json.loads((tmp_path / "gru_history.json").read_text())["kind"] == "gru"
    p = NeuralPredictor(tmp_path / "gru.pt")
    out = p.predict(np.tile([[300.0, 300.0]], (31, 1)) + np.arange(31)[:, None] * [3.0, 0.0],
                    np.zeros(31), np.full((31, 3), 300.0))
    assert out.shape == (45, 2) and np.isfinite(out).all()
