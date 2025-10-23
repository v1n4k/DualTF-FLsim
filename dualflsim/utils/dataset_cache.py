import os
import json
import numpy as np
import gc
from typing import Dict, Any
from .data_loader import load_dataset_by_name, sequential_partition

CACHE_META = "cache_meta.json"


def build_central_cache(
    dataset: str,
    seq_length: int,
    stride: int,
    cache_dir: str,
    num_clients: int = None,
) -> Dict[str, Any]:
    """Persist raw dataset segments for reuse across simulation runs."""

    os.makedirs(cache_dir, exist_ok=True)
    data_dict = load_dataset_by_name(dataset)
    train_segments = data_dict['train_segments']
    test_segments = data_dict['test_segments']
    label_segments = data_dict['label_segments']

    # Persist raw arrays (object arrays of np.ndarrays)
    np.save(os.path.join(cache_dir, 'train_segments.npy'), np.array(train_segments, dtype=object), allow_pickle=True)
    np.save(os.path.join(cache_dir, 'test_segments.npy'), np.array(test_segments, dtype=object), allow_pickle=True)
    np.save(os.path.join(cache_dir, 'label_segments.npy'), np.array(label_segments, dtype=object), allow_pickle=True)

    client_slices = []
    if num_clients is not None and num_clients > 0:
        train_concat = np.concatenate(train_segments, axis=0)
        test_concat = np.concatenate(test_segments, axis=0) if len(test_segments) > 1 else test_segments[0]
        label_concat = np.concatenate(label_segments, axis=0) if len(label_segments) > 1 else label_segments[0]

        np.save(os.path.join(cache_dir, 'train_concat.npy'), train_concat)
        np.save(os.path.join(cache_dir, 'test_concat.npy'), test_concat)
        np.save(os.path.join(cache_dir, 'label_concat.npy'), label_concat)

        lengths = [seg.shape[0] for seg in train_segments]
        slices = sequential_partition(lengths, num_clients)
        client_slices = [(int(sl.start or 0), int(sl.stop or 0)) for sl in slices]
        np.save(os.path.join(cache_dir, 'client_slices.npy'), np.array(client_slices, dtype=np.int64))

    meta = {
        'dataset': dataset,
        'seq_length': seq_length,
        'stride': stride,
        'num_train_segments': len(train_segments),
        'train_segment_lengths': [int(seg.shape[0]) for seg in train_segments],
        'num_test_segments': len(test_segments),
        'test_segment_lengths': [int(seg.shape[0]) for seg in test_segments],
        'feature_dim': int(train_segments[0].shape[1]) if train_segments else 0,
        'has_client_slices': bool(client_slices),
    }

    with open(os.path.join(cache_dir, CACHE_META), 'w') as f:
        json.dump(meta, f, indent=2)

    del data_dict
    gc.collect()
    return meta


def load_cache(cache_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(cache_dir, CACHE_META)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cache metadata not found at {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


# ---- Federated partition indices (IID/Dirichlet) build/reuse ----
from typing import Any, Dict
