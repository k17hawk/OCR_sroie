"""
Checkpoint save/load + a small EMA wrapper.

Every checkpoint carries step, epoch, the full Config and any extra metrics you pass, so a run can be
fully reconstructed from the file alone.

    save_checkpoint("runs/rec/last.pt", model=model, optimizer=opt, scheduler=sched, ema=ema,
                    step=step, epoch=epoch, config=cfg, metrics={"cer": 0.043})
    state = load_checkpoint("runs/rec/last.pt", model=model, optimizer=opt, scheduler=sched, ema=ema)
    print(state["step"], state["metrics"])
"""
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Exponential moving average of a model's parameters. Update after every optimiser step."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.n_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        # Bias-correct early steps so EMA doesn't lag the very first updates.
        d = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src, alpha=1 - d)
            else:
                v.copy_(src)


def save_checkpoint(path: str,
                    *,
                    model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    ema: Optional[ModelEMA] = None,
                    step: int = 0,
                    epoch: int = 0,
                    config: Optional[Any] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    charset: Optional[list] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict() if hasattr(scheduler, "state_dict") else None
    if ema is not None:
        state["ema"] = ema.module.state_dict()
        state["ema_n_updates"] = ema.n_updates
    if config is not None:
        state["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    if charset is not None:
        state["charset"] = list(charset)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str,
                    *,
                    model: Optional[nn.Module] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    ema: Optional[ModelEMA] = None,
                    strict: bool = True,
                    map_location: str = "cpu",
                    prefer_ema: bool = False) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    if model is not None:
        key = "ema" if (prefer_ema and "ema" in state) else "model"
        missing, unexpected = model.load_state_dict(state[key], strict=strict)
        if missing or unexpected:
            print(f"[load_checkpoint] missing={list(missing)} unexpected={list(unexpected)}")
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and "ema" in state:
        ema.module.load_state_dict(state["ema"])
        ema.n_updates = int(state.get("ema_n_updates", 0))
    return {
        "step": int(state.get("step", 0)),
        "epoch": int(state.get("epoch", 0)),
        "metrics": dict(state.get("metrics", {})),
        "config": dict(state.get("config", {})),
        "charset": state.get("charset"),
    }


def find_latest(out_dir: str) -> Optional[str]:
    """Return the newest checkpoint in a run directory (by mtime). None if there is none."""
    p = Path(out_dir)
    if not p.is_dir():
        return None
    ckpts = [f for f in p.glob("*.pt") if not f.name.endswith(".tmp")]
    if not ckpts:
        return None
    return str(max(ckpts, key=lambda f: f.stat().st_mtime))