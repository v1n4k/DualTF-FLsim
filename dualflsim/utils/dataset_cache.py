import os
import json
import numpy as np
import gc
from typing import Dict, Any
from .data_loader import load_dataset_by_name

CACHE_META = "cache_meta.json"


def build_central_cache(dataset: str, seq_length: int, stride: int, cache_dir: str) -> Dict[str, Any]:
    """Build a centralized cache of dataset partitions.

    Writes per-client index arrays and full test/label arrays to disk.
    Returns metadata dict.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_dict = load_dataset_by_name(dataset, seq_length=seq_length, stride=stride)
    # Using first (only) element lists
    x_train = data_dict['x_train'][0]  # [N_train, seq, feat]
    x_test = data_dict['x_test'][0]
    y_test = data_dict['y_test'][0]

    # Compute global mean/std for standardization (per feature)
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0) + 1e-6

    meta = {
        'dataset': dataset,
        'seq_length': seq_length,
        'stride': stride,
        'train_shape': list(x_train.shape),
        'test_shape': list(x_test.shape),
        'label_shape': list(y_test.shape),
        'has_mean_std': True
    }

    # Persist arrays
    np.save(os.path.join(cache_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(cache_dir, 'y_test.npy'), y_test)
    # Train windows stored whole; client slices will reference indices
    np.save(os.path.join(cache_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(cache_dir, 'train_mean.npy'), train_mean)
    np.save(os.path.join(cache_dir, 'train_std.npy'), train_std)

    # Write meta
    with open(os.path.join(cache_dir, CACHE_META), 'w') as f:
        json.dump(meta, f, indent=2)

    # Free memory in driver
    del x_train, x_test, y_test, data_dict
    gc.collect()
    return meta


def load_cache(cache_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(cache_dir, CACHE_META)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cache metadata not found at {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta
