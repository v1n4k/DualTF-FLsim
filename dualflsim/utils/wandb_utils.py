"""Weighs & Biases helpers with safe no-op if disabled.

Usage:
    from utils.wandb_utils import maybe_init_wandb, log_metrics, finish

    run = maybe_init_wandb(cfg)
    log_metrics({"train/loss": 0.5}, step=1)
    finish()
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore


_run = None  # type: ignore


def maybe_init_wandb(cfg: Dict[str, Any]) -> Optional[Any]:
    global _run
    wb = cfg.get("wandb", {}) if isinstance(cfg, dict) else {}
    if not wb or not bool(wb.get("enabled", False)):
        return None
    if not _WANDB_AVAILABLE:
        print("[wandb] Not installed; skipping tracking. Install 'wandb' to enable.")
        return None
    proj = wb.get("project", "DualTF-FLSim")
    entity = wb.get("entity", None)
    run_name = wb.get("run_name", None)
    tags = wb.get("tags", None)
    notes = wb.get("notes", None)
    # Include key sections of cfg for provenance (flatten minimal)
    config_to_log = {
        "simulation": cfg.get("simulation", {}),
        "training": cfg.get("training", {}),
        "model": cfg.get("model", {}),
        "data": cfg.get("data", {}),
    }
    _run = wandb.init(project=proj, entity=entity, name=run_name, tags=tags, notes=notes, config=config_to_log)
    return _run


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if _run is None or not _WANDB_AVAILABLE:
        return
    wandb.log(metrics, step=step)


def log_artifact(path: str, name: Optional[str] = None, type_: str = "artifact") -> None:
    if _run is None or not _WANDB_AVAILABLE:
        return
    art = wandb.Artifact(name or os.path.basename(path), type=type_)
    art.add_file(path)
    _run.log_artifact(art)


def finish() -> None:
    global _run
    if _run is None or not _WANDB_AVAILABLE:
        return
    try:
        _run.finish()
    except Exception:
        pass
    _run = None
