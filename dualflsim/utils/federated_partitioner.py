from __future__ import annotations
from typing import List
import numpy as np


def partition_indices_dirichlet_quantity(n_samples: int, num_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet(np.repeat(alpha, num_clients))
    counts = np.round(proportions * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    parts = []
    ptr = 0
    for c in counts:
        parts.append(idx[ptr:ptr + int(c)])
        ptr += int(c)
    return parts


def partition_indices_iid(n_samples: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    # Approximately equal IID: shuffle then split; first remainder clients take +1
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples)
    rng.shuffle(idx)
    base = n_samples // num_clients
    rem = n_samples % num_clients
    parts = []
    ptr = 0
    for i in range(num_clients):
        take = base + (1 if i < rem else 0)
        parts.append(idx[ptr:ptr + take])
        ptr += take
    return parts
