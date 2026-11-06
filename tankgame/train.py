"""Training of the neural trajectory predictors.

The whole dataset is copied to the GPU once; every training step samples a
random batch of window start indices and cuts the windows directly on the
device, so there is no data-loader bottleneck.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .data import WINDOW, load_dataset, merge_datasets, split_windows
from .features import build_features, ego_targets, integrate
from .models import build_model, count_parameters
from .predictors import MODELS_DIR


def pick_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class WindowSampler:
    """Cuts (features, targets) batches out of a dataset that lives on the device."""

    def __init__(self, dataset: dict, device: str):
        self.traj = torch.as_tensor(dataset["traj"], dtype=torch.float32, device=device)
        self.starts = torch.as_tensor(dataset["starts"], dtype=torch.long, device=device)
        self.offsets = torch.arange(WINDOW, device=device)
        self.device = device

    def __len__(self) -> int:
        return len(self.starts)

    def batch(self, idx: torch.Tensor):
        win = self.traj[self.starts[idx][:, None] + self.offsets]
        pos, head, rays, fut, beh = split_windows(win)
        x = build_features(pos, head, rays)
        y = ego_targets(fut, pos[:, -1], head[:, -1])
        return x, y, beh

    def random_batch(self, size: int, generator: torch.Generator):
        idx = torch.randint(len(self.starts), (size,), device=self.device, generator=generator)
        return self.batch(idx)


@torch.no_grad()
def evaluate_model(model, sampler: WindowSampler, idx: torch.Tensor, batch: int = 4096) -> dict:
    model.eval()
    ade_sum, fde_sum, n = 0.0, 0.0, 0
    for i in range(0, len(idx), batch):
        x, y, _ = sampler.batch(idx[i:i + batch])
        err = torch.linalg.norm(integrate(model(x)) - y, dim=-1)
        ade_sum += err.mean(dim=1).sum().item()
        fde_sum += err[:, -1].sum().item()
        n += len(x)
    model.train()
    return {"ade": ade_sum / n, "fde": fde_sum / n}


def train(kind: str = "gru", train_path: str = "data/train.npz", val_path: str = "data/val.npz",
          extra: list[str] | None = None, out_dir: str | Path = MODELS_DIR, epochs: int = 25,
          steps_per_epoch: int = 500, batch_size: int = 512, lr: float = 2e-3,
          hidden: int = 128, layers: int = 2, device: str | None = None, seed: int = 0,
          val_windows: int = 20000, log=print) -> dict:
    device = pick_device(device)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    datasets = [load_dataset(train_path)] + [load_dataset(p) for p in (extra or [])]
    train_ds = merge_datasets(datasets) if len(datasets) > 1 else datasets[0]
    val_ds = load_dataset(val_path)
    tr = WindowSampler(train_ds, device)
    va = WindowSampler(val_ds, device)
    rng = np.random.default_rng(seed)
    val_idx = torch.as_tensor(rng.choice(len(va), size=min(val_windows, len(va)), replace=False),
                              device=device)

    model_kwargs = {"hidden": hidden, "layers": layers} if kind in ("gru", "lstm") else {}
    model = build_model(kind, **model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.15)
    log(f"[{kind}] device={device} params={count_parameters(model):,} "
        f"train_windows={len(tr):,} val_windows={len(val_idx):,}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        for _ in range(steps_per_epoch):
            x, y, _ = tr.random_batch(batch_size, gen)
            pred = integrate(model(x))
            loss = F.smooth_l1_loss(pred / C.LOSS_SCALE, y / C.LOSS_SCALE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        metrics = evaluate_model(model, va, val_idx)
        row = {"epoch": epoch, "train_loss": running / steps_per_epoch, "val_ade": metrics["ade"],
               "val_fde": metrics["fde"], "lr": sched.get_last_lr()[0], "seconds": time.time() - t0}
        history.append(row)
        improved = metrics["ade"] < best
        if improved:
            best = metrics["ade"]
            torch.save({"kind": kind, "model_kwargs": model_kwargs,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "metrics": {"val_ade": metrics["ade"], "val_fde": metrics["fde"], "epoch": epoch},
                        "config": {"history": C.HISTORY, "horizon": C.HORIZON, "model_dt": C.MODEL_DT,
                                   "n_features": C.N_FEATURES}},
                       out_dir / f"{kind}.pt")
        log(f"[{kind}] epoch {epoch:3d}/{epochs} loss={row['train_loss']:.4f} "
            f"val ADE={metrics['ade']:6.2f} FDE={metrics['fde']:6.2f} ({row['seconds']:.1f}s)"
            + (" *" if improved else ""))

    summary = {"kind": kind, "best_val_ade": best, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
               "batch_size": batch_size, "lr": lr, "params": count_parameters(model),
               "train_seconds": round(time.time() - t_start, 1), "device": device,
               "train_windows": len(tr), "history": history}
    (out_dir / f"{kind}_history.json").write_text(json.dumps(summary, indent=2))
    return summary
