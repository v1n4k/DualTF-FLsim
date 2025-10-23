"""Configuration loader for DualTF-FLSim.

Loads YAML configuration from a provided path, environment variable, or the
default file at <project_root>/configs/default.yaml. Provides sensible
defaults and merges them with the loaded file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required. Please add 'pyyaml' to requirements and install it."
    ) from e


def _project_root() -> Path:
    # utils/config.py -> utils -> project root
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _project_root() / "configs" / "default.yaml"


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def default_config() -> Dict[str, Any]:
    return {
        "ray": {"num_cpus": 32, "num_gpus": 4, "ignore_reinit_error": True},
        "resources": {"client": {"num_cpus": 4, "num_gpus": 1}},
        "simulation": {
            "num_clients": 24,
            "total_rounds": 10,
            "fraction_fit": 0.25,
            "fraction_evaluate": 0.0,
            "min_available_clients": 6,
            "min_fit_clients": 6,
        },
        "training": {"local_epochs": 5, "lr": 1e-4, "proximal_mu": 0.0},
        "model": {
            "seq_len": 100,
            "time": {"enc_in": 25, "c_out": 25, "e_layers": 3, "n_heads": 8, "d_model": 512},
            "freq": {"nest_length": 25, "enc_in": 25, "c_out": 25, "e_layers": 3, "n_heads": 4, "d_model": 512},
        },
        "data": {
            "dataset": "PSM",
            "partition_mode": "sequential",
            "cache_clients": None,
            "time_batch_size": 64,
            "freq_batch_size": 16,
            "seq_length": 100,
            "step": 1,
        },
        "evaluation": {
            "enabled": False,
            "min_consecutive": 10,
            "num_thresholds_periodic": 256,
            "num_thresholds_final": 1000,
            "seq_length": 50,
            "step": 5,
            "threshold_quantile_range": [0.01, 0.99],
            "prefer_gpu_sweep": True,
        },
        "scaffold": {"damping": 0.1, "max_corr_norm": 1.0, "grad_clip": 1.0},
        "post_training": {
            "generate_arrays": True,
            "dataset": "PSM",
            "data_num": 0,
            "seq_length": 100,
            "nest_length": 25,
        },
    }


def load_config(path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    """Load configuration dictionary.

    Order of precedence:
    1) Explicit path argument
    2) Environment variable DUALFLSIM_CONFIG
    3) Default file at <project_root>/configs/default.yaml
    If file is missing, returns the internal defaults.
    """
    defaults = default_config()

    # Resolve target path
    cfg_env = os.environ.get("DUALFLSIM_CONFIG")
    candidate = Path(path) if path else (Path(cfg_env) if cfg_env else _default_config_path())

    if candidate and candidate.exists():
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Configuration file must parse to a dictionary")
            return _deep_update(defaults, loaded)
        except Exception as e:
            # Fallback to defaults on parse errors
            print(f"[Config] Failed to load {candidate}: {e}. Using defaults.")
            return defaults
    else:
        # No file found: return defaults
        return defaults


__all__ = ["load_config", "default_config"]
