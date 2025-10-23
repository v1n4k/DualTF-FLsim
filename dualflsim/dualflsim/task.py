"""DualFLSim: A Flower / PyTorch app."""

from collections import OrderedDict
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast, GradScaler

# Import the model definitions from the copied directory
from model.TimeTransformer import AnomalyTransformer
from model.FrequencyTransformer import FrequencyTransformer

# Add the project root to the Python path to allow imports from `utils`
import sys
from pathlib import Path
# Assumes task.py is in dualflsim/dualflsim/
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the data loading and processing logic from the copied utils
from utils.data_loader import load_tods, TrainingLoader, GeneralLoader, generate_frequency_grandwindow, _create_sequences, load_dataset_by_name
from utils.dataset_cache import load_cache
from utils.config import load_config


class FederatedDualTF(nn.Module):
    """A unified model container for the time and frequency transformers."""
    def __init__(self, time_model_args, freq_model_args):
        super().__init__()
        self.time_model = AnomalyTransformer(**time_model_args)
        self.freq_model = FrequencyTransformer(**freq_model_args)


def load_data(
    partition_id: int,
    num_partitions: int,
    time_batch_size: int = 64,
    freq_batch_size: int = 16,
    seq_length: int = 50,
    nest_length: Optional[int] = None,
):
    """Load and partition the time-series data per plan."""

    cfg = load_config()
    data_cfg = cfg.get('data', {})
    dataset_name = data_cfg.get('dataset', 'PSM')
    client_load_test = bool(data_cfg.get('client_load_test', False))
    partition_mode = str(data_cfg.get('partition_mode', 'sequential')).lower()

    cache_dir = os.environ.get('DUALFLSIM_CACHE_DIR')
    cache_enabled = os.environ.get('DUALFLSIM_CACHE_ENABLED') == '1'
    cached_dataset = os.environ.get('DUALFLSIM_DATASET')

    use_cache = (
        cache_enabled
        and cache_dir
        and os.path.isdir(cache_dir)
        and cached_dataset
        and cached_dataset.upper() == dataset_name.upper()
        and partition_mode == 'sequential'
    )

    if use_cache:
        train_concat = np.load(os.path.join(cache_dir, 'train_concat.npy'), mmap_mode='r')
        test_concat = np.load(os.path.join(cache_dir, 'test_concat.npy'), mmap_mode='r')
        label_concat = np.load(os.path.join(cache_dir, 'label_concat.npy'), mmap_mode='r')
        slices_path = os.path.join(cache_dir, 'client_slices.npy')
        if os.path.exists(slices_path):
            slices = np.load(slices_path)
            if partition_id >= len(slices):
                raise IndexError(f"partition_id {partition_id} out of range for cached slices ({len(slices)})")
            start, stop = map(int, slices[partition_id])
        else:
            meta = load_cache(cache_dir)
            lengths = meta.get('train_segment_lengths', [])
            if partition_id >= len(lengths):
                raise IndexError(f"partition_id {partition_id} out of range for cached segment lengths ({len(lengths)})")
            computed = sequential_partition(lengths, num_partitions)
            start = int((computed[partition_id].start or 0))
            stop = int((computed[partition_id].stop or 0))
        train_segment = train_concat[start:stop]
        test_segment = test_concat
        label_segment = label_concat
    else:
        data_dict = load_dataset_by_name(dataset_name)

        if dataset_name.upper() == 'SMD' and partition_mode == 'by_machine':
            if partition_id >= len(data_dict['train_segments']):
                raise IndexError("partition_id out of range for SMD machines")
            train_segment = data_dict['train_segments'][partition_id]
            test_segment = data_dict['test_segments'][partition_id]
            label_segment = data_dict['label_segments'][partition_id]
        elif dataset_name.upper() in ['SMAP', 'MSL'] and partition_mode == 'by_machine':
            if partition_id >= len(data_dict['train_segments']):
                raise IndexError("partition_id out of range for SMAP/MSL channels")
            train_segment = data_dict['train_segments'][partition_id]
            test_segment = data_dict['test_segments'][partition_id]
            label_segment = data_dict['label_segments'][partition_id]
        else:
            lengths = [seg.shape[0] for seg in data_dict['train_segments']]
            slices = sequential_partition(lengths, num_partitions)
            concat_train = np.concatenate(data_dict['train_segments'], axis=0)
            current_slice = slices[partition_id]
            train_segment = concat_train[current_slice]
            test_segment = np.concatenate(data_dict['test_segments'], axis=0) if len(data_dict['test_segments']) > 1 else data_dict['test_segments'][0]
            label_segment = np.concatenate(data_dict['label_segments'], axis=0) if len(data_dict['label_segments']) > 1 else data_dict['label_segments'][0]

    feature_dim = train_segment.shape[1]
    x_train_partition = _create_sequences(train_segment, seq_length, 1)
    if client_load_test and test_segment.shape[0] >= seq_length:
        x_test = _create_sequences(test_segment, seq_length, 1)
        y_test = _create_sequences(label_segment, seq_length, 1)
    else:
        x_test = np.empty((0, seq_length, feature_dim))
        y_test = np.empty((0, seq_length, feature_dim))

    # Validate feature dims vs config (applies in both modes)
    time_cfg = cfg.get('model', {}).get('time', {})
    freq_cfg = cfg.get('model', {}).get('freq', {})
    expected_time = (time_cfg.get('enc_in'), time_cfg.get('c_out'))
    expected_freq = (freq_cfg.get('enc_in'), freq_cfg.get('c_out'))
    mismatches = []
    if expected_time[0] not in (feature_dim, None) or expected_time[1] not in (feature_dim, None):
        mismatches.append(f"time.enc_in/time.c_out={expected_time} vs dataset feature_dim={feature_dim}")
    if expected_freq[0] not in (feature_dim, None) or expected_freq[1] not in (feature_dim, None):
        mismatches.append(f"freq.enc_in/freq.c_out={expected_freq} vs dataset feature_dim={feature_dim}")
    if mismatches:
        raise ValueError(
            "Configuration feature mismatch for dataset '" + dataset_name + "':\n  " +
            "\n  ".join(mismatches) +
            "\nFix: Set all of model.time.enc_in, model.time.c_out, model.freq.enc_in, model.freq.c_out to " + str(feature_dim) +
            f" (current dataset '{dataset_name}' inferred feature_dim={feature_dim}). See README consistency rules."
        )

    print(f"Client {partition_id}: Loading {len(x_train_partition)} time-domain training samples. mode={partition_mode}")

    # Create Time-Domain DataLoaders
    train_dataset_time = TrainingLoader(x_train_partition)
    trainloader_time = DataLoader(dataset=train_dataset_time, batch_size=time_batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    if client_load_test and x_test.size > 0:
        test_dataset_time = GeneralLoader(x_test, y_test)
        testloader_time = DataLoader(dataset=test_dataset_time, batch_size=time_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    else:
        empty_time = TrainingLoader(np.empty((0, seq_length, feature_dim)))
        testloader_time = DataLoader(dataset=empty_time, batch_size=time_batch_size)

    # --- Federated Partitioning (Frequency Domain) ---
    # Resolve nest_length from argument or configuration (single source of truth)
    if nest_length is None:
        cfg = load_config()
        nest_length = int(cfg.get('model', {}).get('freq', {}).get('nest_length', 25))
    # Generate frequency data for the client's partition
    freq_dict = generate_frequency_grandwindow(x_train_partition, x_test if client_load_test else np.empty((0, seq_length, feature_dim)),
                                              y_test if client_load_test else np.empty((0, seq_length, feature_dim)), nest_length, step=1)
    x_train_freq_partition = freq_dict['grand_train_reshaped']
    x_test_freq = freq_dict['grand_test_reshaped']
    y_test_freq = freq_dict['grand_label_reshaped']

    print(f"Client {partition_id}: Loading {len(x_train_freq_partition)} frequency-domain training samples.")

    # Create Frequency-Domain DataLoaders
    train_dataset_freq = TrainingLoader(x_train_freq_partition)
    trainloader_freq = DataLoader(dataset=train_dataset_freq, batch_size=freq_batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    if client_load_test and x_test_freq.size > 0:
        test_dataset_freq = GeneralLoader(x_test_freq, y_test_freq)
        testloader_freq = DataLoader(dataset=test_dataset_freq, batch_size=freq_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    else:
        # Determine freq feature dimension safely
        if freq_dict['grand_train_reshaped'].size > 0:
            freq_feat = freq_dict['grand_train_reshaped'].shape[-1]
        else:
            freq_feat = feature_dim
        empty_freq = TrainingLoader(np.empty((0, 0, freq_feat)))
        testloader_freq = DataLoader(dataset=empty_freq, batch_size=freq_batch_size)
    
    return trainloader_time, testloader_time, trainloader_freq, testloader_freq


def load_centralized_test_data(
    time_batch_size: int = 64,
    freq_batch_size: int = 16,
    seq_length: int = 50,
    nest_length: Optional[int] = None,
):
    """Load the centralized (full) test dataset."""
    cfg = load_config()
    dataset_name = cfg.get('data', {}).get('dataset', 'PSM')
    data_dict = load_dataset_by_name(dataset_name)
    sample_segment = data_dict['train_segments'][0] if 'train_segments' in data_dict else data_dict['train_sequences'][0]
    x_train_sample = _create_sequences(sample_segment, seq_length, 1)
    feature_dim = x_train_sample.shape[-1]
    time_cfg = cfg.get('model', {}).get('time', {})
    freq_cfg = cfg.get('model', {}).get('freq', {})
    expected_time = (time_cfg.get('enc_in'), time_cfg.get('c_out'))
    expected_freq = (freq_cfg.get('enc_in'), freq_cfg.get('c_out'))
    if (expected_time[0] != feature_dim or expected_time[1] != feature_dim or 
        expected_freq[0] != feature_dim or expected_freq[1] != feature_dim):
        raise ValueError(
            f"Model configuration enc_in/c_out mismatch for dataset '{dataset_name}'. Expected all four to be {feature_dim}, got time={expected_time}, freq={expected_freq}."
        )
    test_concat = np.concatenate(data_dict['test_segments'], axis=0) if len(data_dict['test_segments']) > 1 else data_dict['test_segments'][0]
    label_concat = np.concatenate(data_dict['label_segments'], axis=0) if len(data_dict['label_segments']) > 1 else data_dict['label_segments'][0]

    time_windows = _create_sequences(test_concat, seq_length, 1)
    label_windows = _create_sequences(label_concat, seq_length, 1)
    test_dataset_time = GeneralLoader(time_windows, label_windows)
    testloader_time = DataLoader(dataset=test_dataset_time, batch_size=time_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    # Create Frequency-Domain DataLoader
    if nest_length is None:
        cfg = load_config()
        nest_length = int(cfg.get('model', {}).get('freq', {}).get('nest_length', 25))
    # Note: We pass dummy empty arrays for train data as we only need the test set
    freq_dict = generate_frequency_grandwindow(np.array([]), time_windows, label_windows, nest_length, step=1)
    x_test_freq = freq_dict['grand_test_reshaped']
    y_test_freq = freq_dict['grand_label_reshaped']

    test_dataset_freq = GeneralLoader(x_test_freq, y_test_freq)
    testloader_freq = DataLoader(dataset=test_dataset_freq, batch_size=freq_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    return testloader_time, testloader_freq


def my_kl_loss(p, q):
    """Numerically stable KL(p||q) per position.

    - Upcast to float32 to avoid FP16 log underflow/overflow under AMP.
    - Renormalize along the last dim to ensure valid distributions.
    - Clamp and sanitize before log to avoid NaN/Inf.
    Returns tensor of shape [B, L] when inputs are [B, H, L, L].
    """
    eps = 1e-12
    # Upcast for stability
    p = p.to(dtype=torch.float32)
    q = q.to(dtype=torch.float32)
    # Renormalize to probabilities along the last dim
    p_sum = torch.sum(p, dim=-1, keepdim=True)
    q_sum = torch.sum(q, dim=-1, keepdim=True)
    p = p / (p_sum + eps)
    q = q / (q_sum + eps)
    # Clamp and sanitize
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    # KL computation
    res = p * (torch.log(p) - torch.log(q))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def train(net, trainloader_time, trainloader_freq, epochs, device, proximal_mu, k=3.0, lr=1e-4, control_c=None, control_ci=None, log_step_recon_stats: bool = False):
    """Train the complete DualTF model.

    If control_c and control_ci are provided (SCAFFOLD), apply gradient
    correction g <- g - ci + c before each optimizer step. Set proximal_mu=0
    to disable FedProx when using SCAFFOLD.
    Returns: (train_loss_placeholder, total_optimizer_steps)
    """
    
    # Note: Clone initial weights per-submodel on the target device for FedProx

    # Enable mixed precision to reduce memory
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Prepare control variates if provided
    if control_c is not None and control_ci is not None:
        c_tensors_all = [torch.tensor(a, device=device, dtype=torch.float32) for a in control_c]
        ci_tensors_all = [torch.tensor(a, device=device, dtype=torch.float32) for a in control_ci]
    else:
        c_tensors_all = None
        ci_tensors_all = None

    total_steps = 0
    k_actual = 0
    # Track cumulative losses for average
    sum_loss_time = 0.0
    count_time = 0
    sum_rec_time = 0.0  # pure reconstruction (MSE) component average
    step_rec_time_vals = [] if log_step_recon_stats else None

    # --- Train Time Model ---
    time_model = net.time_model
    time_model.to(device)
    # Clone global params for time model on the same device as local params
    global_params_time = [p.detach().clone().to(device) for p in time_model.parameters()]
    time_model.train()
    time_optimizer = torch.optim.Adam(time_model.parameters(), lr=lr)
    time_criterion = nn.MSELoss()
    
    print("Training Time-Domain Model...")
    params_time = list(time_model.parameters())
    n_time = len(params_time)
    if c_tensors_all is not None:
        c_time = c_tensors_all[:n_time]
        ci_time = ci_tensors_all[:n_time]
    else:
        c_time = ci_time = None
    for epoch in range(epochs):
        for input_data in trainloader_time:
            input_data = input_data.float().to(device)
            time_optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                output, series, prior, _ = time_model(input_data)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    # Stable normalization with epsilon to avoid NaNs/Infs
                    den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                    prior_norm = prior[u] / den  # [B, H, L, L]
                    # Sanitize inputs to KL to avoid propagating NaNs/Infs
                    prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                    series_loss += (
                        torch.mean(
                            my_kl_loss(
                                series_u,
                                prior_norm.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                prior_norm.detach(),
                                series_u,
                            )
                        )
                    )
                    prior_loss += (
                        torch.mean(
                            my_kl_loss(
                                prior_norm,
                                series_u.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                series_u.detach(),
                                prior_norm,
                            )
                        )
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)
                rec_loss = time_criterion(output, input_data)
                loss1 = rec_loss - k * series_loss
                loss2 = rec_loss + k * prior_loss

                # Add FedProx proximal term
                proximal_term = 0.0
                for local_param, global_param in zip(time_model.parameters(), global_params_time):
                    proximal_term += (local_param - global_param).pow(2).sum()

                total_loss = loss1 + loss2 + (proximal_mu / 2) * proximal_term

            scaler.scale(total_loss).backward()
            # Apply SCAFFOLD correction if provided
            scaler.unscale_(time_optimizer)
            # Guard: skip if any gradient is non-finite
            if any(p.grad is not None and (not torch.isfinite(p.grad).all()) for p in time_model.parameters()):
                time_optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            # Clip base gradients
            torch.nn.utils.clip_grad_norm_(time_model.parameters(), max_norm=1.0)
            if c_time is not None and ci_time is not None:
                # Clip correction to avoid exploding updates
                max_corr_norm = 1.0
                for p, c_k, ci_k in zip(params_time, c_time, ci_time):
                    if p.grad is not None:
                        diff = (c_k - ci_k)
                        # global-norm clipping per tensor
                        n = torch.linalg.vector_norm(diff).item()
                        if n > max_corr_norm:
                            diff = diff * (max_corr_norm / (n + 1e-12))
                        p.grad.add_(diff)
            # Use scaler.step to respect found_inf and count successful steps
            _s_before = scaler.get_scale()
            scaler.step(time_optimizer)
            scaler.update()
            if scaler.get_scale() >= _s_before:
                k_actual += 1
                # accumulate averages only for successful optimizer steps
                sum_loss_time += float(total_loss.detach().cpu())
                rec_scalar_t = float(rec_loss.detach().mean().cpu())
                sum_rec_time += rec_scalar_t
                if step_rec_time_vals is not None:
                    step_rec_time_vals.append(rec_scalar_t)
                count_time += 1
            total_steps += 1

    # --- Train Frequency Model ---
    freq_model = net.freq_model
    freq_model.to(device)
    # Clone global params for freq model on the same device as local params
    global_params_freq = [p.detach().clone().to(device) for p in freq_model.parameters()]
    freq_model.train()
    freq_optimizer = torch.optim.Adam(freq_model.parameters(), lr=lr)
    freq_criterion = nn.MSELoss()

    print("Training Frequency-Domain Model...")
    sum_loss_freq = 0.0
    count_freq = 0
    sum_rec_freq = 0.0
    step_rec_freq_vals = [] if log_step_recon_stats else None
    for epoch in range(epochs):
        for input_data in trainloader_freq:
            input_data = input_data.float().to(device)
            freq_optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                output, series, prior, _ = freq_model(input_data)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                    prior_norm = prior[u] / den
                    prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                    series_loss += (
                        torch.mean(
                            my_kl_loss(
                                series_u,
                                prior_norm.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                prior_norm.detach(),
                                series_u,
                            )
                        )
                    )
                    prior_loss += (
                        torch.mean(
                            my_kl_loss(
                                prior_norm,
                                series_u.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                series_u.detach(),
                                prior_norm,
                            )
                        )
                    )
                series_loss /= len(prior)
                prior_loss /= len(prior)
                rec_loss = freq_criterion(output, input_data)
                loss1 = rec_loss - k * series_loss
                loss2 = rec_loss + k * prior_loss

                # Add FedProx proximal term
                proximal_term = 0.0
                for local_param, global_param in zip(freq_model.parameters(), global_params_freq):
                    proximal_term += (local_param - global_param).pow(2).sum()

                total_loss = loss1 + loss2 + (proximal_mu / 2) * proximal_term

            scaler.scale(total_loss).backward()
            # Apply SCAFFOLD correction if provided
            scaler.unscale_(freq_optimizer)
            # Guard: skip if any gradient is non-finite
            if any(p.grad is not None and (not torch.isfinite(p.grad).all()) for p in freq_model.parameters()):
                freq_optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            # Clip base gradients
            torch.nn.utils.clip_grad_norm_(freq_model.parameters(), max_norm=1.0)
            if (total_steps % 50) == 0:
                rec_m = float(rec_loss.mean().detach().cpu())
                ser_m = float(series_loss.mean().detach().cpu())
                pri_m = float(prior_loss.mean().detach().cpu())
                # grad norm before correction (freq)
                def _grad_global_norm(params):
                    sq = 0.0
                    for p in params:
                        if p.grad is not None and torch.isfinite(p.grad).all():
                            sq += float(torch.sum(p.grad * p.grad))
                    return (sq ** 0.5)
                # params_freq defined below; compute using current model params
                tmp_params = list(freq_model.parameters())
                gn_before_f = _grad_global_norm(tmp_params)
                print(f"[Client][freq] step={total_steps} rec={rec_m:.4f} series={ser_m:.4f} prior={pri_m:.4f} grad_norm_before={gn_before_f:.4f}")
            params_freq = list(freq_model.parameters())
            if c_tensors_all is not None:
                c_freq = c_tensors_all[n_time:]
                ci_freq = ci_tensors_all[n_time:]
            else:
                c_freq = ci_freq = None
            if c_freq is not None and ci_freq is not None:
                max_corr_norm = 1.0
                corr_sq = 0.0
                for p, c_k, ci_k in zip(params_freq, c_freq, ci_freq):
                    if p.grad is not None:
                        diff = (c_k - ci_k)
                        corr_sq += float(torch.sum(diff * diff))
                        n = torch.linalg.vector_norm(diff).item()
                        if n > max_corr_norm:
                            diff = diff * (max_corr_norm / (n + 1e-12))
                        p.grad.add_(diff)
                if (total_steps % 50) == 0:
                    print(f"[Client][freq] corr_norm_preclip={(corr_sq**0.5):.4f}")
            _s_before = scaler.get_scale()
            scaler.step(freq_optimizer)
            scaler.update()
            if scaler.get_scale() >= _s_before:
                k_actual += 1
                sum_loss_freq += float(total_loss.detach().cpu())
                rec_scalar_f = float(rec_loss.detach().mean().cpu())
                sum_rec_freq += rec_scalar_f
                if step_rec_freq_vals is not None:
                    step_rec_freq_vals.append(rec_scalar_f)
                count_freq += 1
            if (total_steps % 50) == 0:
                print(f"[Client][freq] grad_norm_after (approx) computed next step")
            total_steps += 1

    # Combine average losses (if any). Avoid division by zero.
    avg_time = (sum_loss_time / max(1, count_time)) if count_time > 0 else 0.0
    avg_freq = (sum_loss_freq / max(1, count_freq)) if count_freq > 0 else 0.0
    avg_rec_time = (sum_rec_time / max(1, count_time)) if count_time > 0 else 0.0
    avg_rec_freq = (sum_rec_freq / max(1, count_freq)) if count_freq > 0 else 0.0
    combined_avg = 0.5 * (avg_time + avg_freq)
    # Prepare optional step statistics
    def _compute_stats(arr):
        if arr is None or len(arr) == 0:
            return {}
        a = np.array(arr, dtype=float)
        return {
            'mean': float(a.mean()),
            'std': float(a.std()),
            'min': float(a.min()),
            'max': float(a.max()),
            'last': float(a[-1]),
            'count': int(a.size),
        }
    step_stats_time = _compute_stats(step_rec_time_vals)
    step_stats_freq = _compute_stats(step_rec_freq_vals)
    return float(combined_avg), k_actual, float(avg_rec_time), float(avg_rec_freq), step_stats_time, step_stats_freq


def get_anomaly_scores(net, dataloader, device, is_time_model=True):
    """Calculate anomaly scores for a given model and dataloader."""
    model = net.time_model if is_time_model else net.freq_model
    model.to(device)
    model.eval()
    
    criterion = nn.MSELoss(reduction='none')
    
    scores = []
    labels = []
    
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for input_data, target in dataloader:
            input_data = input_data.float().to(device)
            with autocast(enabled=use_amp):
                output, series, prior, _ = model(input_data)
            
            # Reconstruction loss for each sample in the batch
            rec_loss = criterion(output, input_data).mean(dim=(1, 2))
            
            # Association discrepancy for each sample (stable normalization)
            series_loss = torch.zeros(rec_loss.shape[0], device=device)
            for u in range(len(prior)):
                den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                prior_norm = prior[u] / den
                # Sanitize inputs to KL
                prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                kl_term_1 = my_kl_loss(series_u, prior_norm.detach())
                kl_term_2 = my_kl_loss(prior_norm.detach(), series_u)
                series_loss += (kl_term_1 + kl_term_2).mean(dim=1)
            
            series_loss /= len(prior)
            
            # Anomaly score for inference should be reconstruction + association discrepancy
            anomaly_score = rec_loss + 3.0 * series_loss  # Using k=3.0 as in train
            scores.append(anomaly_score.detach().cpu().numpy())
            # Reduce label to per-sample scalar: 1 if any anomaly in the window, else 0
            if isinstance(target, torch.Tensor):
                t = target
            else:
                t = torch.from_numpy(target)
            if t.dim() >= 2:
                lbl = (t.to(dtype=torch.float32).max(dim=1).values > 0.5).to(dtype=torch.int64)
            else:
                lbl = (t.to(dtype=torch.float32) > 0.5).to(dtype=torch.int64)
            labels.append(lbl.cpu().numpy())
            
    # After loop, concatenate collected arrays. If somehow empty, raise a clear error.
    if len(scores) == 0:
        raise RuntimeError("get_anomaly_scores: No scores were collected from the dataloader; check model forward and dataloader outputs.")
    return np.concatenate(scores), np.concatenate(labels)


def _smooth_pred_mask(mask_bool, min_consecutive: int):
    idx = np.flatnonzero(mask_bool)
    if idx.size == 0:
        return mask_bool
    splits = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, splits)
    keep = np.zeros_like(mask_bool, dtype=bool)
    for r in runs:
        if r.size >= min_consecutive:
            keep[r] = True
    return keep


def compute_seq_metrics(score_sequences, label_sequences, th: float, min_consecutive: int):
    tp, fp, fn = 0, 0, 0
    for i in range(len(score_sequences)):
        seq_scores = score_sequences[i]
        seq_labels = label_sequences[i]
        pred_mask = seq_scores > th
        pred_mask = _smooth_pred_mask(pred_mask, min_consecutive)
        pred_anomalies = set(np.where(pred_mask)[0])
        true_anomalies = set(np.where(seq_labels == 1)[0])
        if len(true_anomalies) > 0:
            if len(pred_anomalies.intersection(true_anomalies)) > 0:
                tp += 1
            else:
                fn += 1
        else:
            if len(pred_anomalies) > 0:
                fp += 1
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return tp, fp, fn, precision, recall, f1


def get_best_f1(
    scores,
    labels,
    seq_length=50,
    step=1,
    num_thresholds=256,
    min_consecutive=5,
    thresholds: np.ndarray = None,
    quantile_range: tuple = (0.01, 0.99),
):
    """Find the best F1 and associated stats by sweeping thresholds.

    Uses quantile-based candidate thresholds by default for speed and robustness.
    """
    score_sequences = _create_sequences(scores, seq_length, step)
    label_sequences = _create_sequences(labels, seq_length, step)
    # Collapse any extra label dims beyond [Nseq, L]
    if hasattr(label_sequences, "ndim") and label_sequences.ndim > 2:
        axes = tuple(range(2, label_sequences.ndim))
        label_sequences = np.any(label_sequences != 0, axis=axes)
    if thresholds is None:
        # Use quantiles to avoid wasting sweeps on flat tails
        lo, hi = quantile_range
        lo = max(0.0, min(1.0, float(lo)))
        hi = max(lo, min(1.0, float(hi)))
        qs = np.linspace(lo, hi, num=num_thresholds)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            finite_scores = np.array([0.0, 1.0])
        thresholds = np.quantile(finite_scores, qs)
    best = (0, 0, 0, float(thresholds[0]), 0, 0)  # f1, p, r, th, tp, fp, fn
    for th in thresholds:
        tp, fp, fn, p, r, f1 = compute_seq_metrics(score_sequences, label_sequences, th, min_consecutive)
        if f1 > best[0]:
            best = (f1, p, r, float(th), tp, fp, fn)
    return best  # f1, p, r, th, tp, fp, fn


def _smooth_mask_torch(pred_mask: torch.Tensor, k: int) -> torch.Tensor:
    """GPU-friendly smoothing: keep positions that belong to runs of >= k ones.

    pred_mask: [B, L] bool tensor.
    Returns: [B, L] bool tensor.
    """
    if k <= 1:
        return pred_mask
    B, L = pred_mask.shape
    if L < k:
        return torch.zeros_like(pred_mask)
    # Identify windows fully one
    windows_all = pred_mask.unfold(dimension=1, size=k, step=1).all(dim=2)  # [B, L-k+1]
    # Transposed conv to expand window indicators back to spans
    x = windows_all.to(dtype=torch.float32).unsqueeze(1)  # [B,1,L-k+1]
    kernel = torch.ones(1, 1, k, device=pred_mask.device, dtype=torch.float32)
    recovered = F.conv_transpose1d(x, kernel, stride=1)  # [B,1,L]
    return recovered.squeeze(1) >= 1.0


def get_best_f1_gpu(
    scores,
    labels,
    device: torch.device,
    seq_length=50,
    step=1,
    num_thresholds=256,
    min_consecutive=5,
    thresholds: np.ndarray = None,
    quantile_range: tuple = (0.01, 0.99),
    thr_chunk: int = 128,
):
    """GPU-accelerated threshold sweep using torch tensors.

    Returns: f1, precision, recall, best_th, tp, fp, fn
    """
    # Create sequences on CPU first, then move to GPU
    score_sequences = _create_sequences(scores, seq_length, step)
    label_sequences = _create_sequences(labels, seq_length, step)
    if hasattr(label_sequences, "ndim") and label_sequences.ndim > 2:
        axes = tuple(range(2, label_sequences.ndim))
        label_sequences = np.any(label_sequences != 0, axis=axes)
    scores_t = torch.from_numpy(score_sequences).to(device=device, dtype=torch.float32)
    labels_t = torch.from_numpy(label_sequences.astype(np.bool_)).to(device=device)
    Nseq, L = scores_t.shape
    # Candidate thresholds
    if thresholds is None:
        lo, hi = quantile_range
        lo = max(0.0, min(1.0, float(lo)))
        hi = max(lo, min(1.0, float(hi)))
        qs = np.linspace(lo, hi, num=num_thresholds)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            finite_scores = np.array([0.0, 1.0])
        thresholds = np.quantile(finite_scores, qs)
    thr_t = torch.as_tensor(thresholds, device=device, dtype=torch.float32)

    # Precompute per-sequence truth mask and has_true flags
    true_mask = labels_t  # [Nseq,L]
    has_true = true_mask.any(dim=1)  # [Nseq]

    best = {
        "f1": -1.0,
        "p": 0.0,
        "r": 0.0,
        "th": float(thr_t[0].item()),
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    eps = 1e-7
    T = thr_t.numel()
    for start in range(0, T, thr_chunk):
        end = min(T, start + thr_chunk)
        ths = thr_t[start:end]  # [Tc]
        # Compare and smooth
        pred = (scores_t.unsqueeze(0) > ths[:, None, None])  # [Tc,Nseq,L]
        Tc = pred.shape[0]
        pred_flat = pred.reshape(Tc * Nseq, L)
        sm_flat = _smooth_mask_torch(pred_flat, min_consecutive)
        sm = sm_flat.view(Tc, Nseq, L)
        # Metrics per threshold
        has_pred = sm.any(dim=2)  # [Tc,Nseq]
        overlap = (sm & true_mask.unsqueeze(0)).any(dim=2)  # [Tc,Nseq]
        has_true_b = has_true.unsqueeze(0).expand(Tc, -1)  # [Tc,Nseq]
        tp = (has_true_b & overlap).sum(dim=1).to(dtype=torch.float32)
        fn = (has_true_b & (~overlap)).sum(dim=1).to(dtype=torch.float32)
        fp = ((~has_true_b) & has_pred).sum(dim=1).to(dtype=torch.float32)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        # Pick best in this chunk
        idx = torch.argmax(f1)
        f1_b = f1[idx].item()
        if f1_b > best["f1"]:
            best.update({
                "f1": f1_b,
                "p": precision[idx].item(),
                "r": recall[idx].item(),
                "th": float(ths[idx].item()),
                "tp": int(tp[idx].item()),
                "fp": int(fp[idx].item()),
                "fn": int(fn[idx].item()),
            })
    return best["f1"], best["p"], best["r"], best["th"], best["tp"], best["fp"], best["fn"]


def get_f1_fixed_threshold(scores, labels, threshold: float, seq_length=50, step=1, min_consecutive=5):
    """Compute F1 at a fixed threshold."""
    score_sequences = _create_sequences(scores, seq_length, step)
    label_sequences = _create_sequences(labels, seq_length, step)
    if hasattr(label_sequences, "ndim") and label_sequences.ndim > 2:
        axes = tuple(range(2, label_sequences.ndim))
        label_sequences = np.any(label_sequences != 0, axis=axes)
    tp, fp, fn, p, r, f1 = compute_seq_metrics(score_sequences, label_sequences, threshold, min_consecutive)
    return f1, p, r, tp, fp, fn


def test(
    net,
    testloader_time,
    testloader_freq,
    device,
    eval_state=None,
    min_consecutive=10,
    num_thresholds=256,
    seq_length=50,
    step=1,
    threshold_quantile_range=(0.01, 0.99),
    prefer_gpu_sweep: bool = True,
):
    """Validate the complete DualTF model.

    Behavior:
    - Fit scalers only once (cache in eval_state) and reuse across rounds.
    - On first call (no threshold in eval_state), sweep thresholds and cache the best.
    - On subsequent calls, evaluate at the fixed cached threshold.
    """
    if eval_state is None:
        eval_state = {}
    
    # 1. Get anomaly scores from both models
    time_scores, time_labels = get_anomaly_scores(net, testloader_time, device, is_time_model=True)
    freq_scores, _ = get_anomaly_scores(net, testloader_freq, device, is_time_model=False)

    # Diagnostics: check for NaN/Inf and value ranges before scaling
    def _print_stats(name, arr):
        total = arr.size
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        finite_mask = np.isfinite(arr)
        finite_cnt = int(finite_mask.sum())
        if finite_cnt > 0:
            fin_min = float(np.min(arr[finite_mask]))
            fin_max = float(np.max(arr[finite_mask]))
        else:
            fin_min = np.nan
            fin_max = np.nan
        print(f"[Eval] {name}: size={total}, nan={nan}, inf={inf}, finite={finite_cnt}, min={fin_min}, max={fin_max}")

    _print_stats("time_scores", time_scores)
    _print_stats("freq_scores", freq_scores)

    # The frequency domain data is shorter due to windowing. We need to align them.
    # The original paper's evaluation logic loads pre-computed, aligned scores.
    # Here, we'll pad the shorter frequency scores to match the time scores length.
    if len(freq_scores) < len(time_scores):
        padding = len(time_scores) - len(freq_scores)
        freq_scores = np.pad(freq_scores, (0, padding), 'edge')

    # 2. Normalize and combine scores (sanitize to avoid NaN/Inf)
    time_scores = np.nan_to_num(time_scores, nan=0.0, posinf=0.0, neginf=0.0)
    freq_scores = np.nan_to_num(freq_scores, nan=0.0, posinf=0.0, neginf=0.0)
    # Fit scalers only once and reuse
    scaler_time = eval_state.get("scaler_time")
    scaler_freq = eval_state.get("scaler_freq")
    if scaler_time is None:
        scaler_time = MinMaxScaler()
        scaler_time.fit(time_scores.reshape(-1, 1))
        eval_state["scaler_time"] = scaler_time
    if scaler_freq is None:
        scaler_freq = MinMaxScaler()
        scaler_freq.fit(freq_scores.reshape(-1, 1))
        eval_state["scaler_freq"] = scaler_freq
    time_scores_norm = scaler_time.transform(time_scores.reshape(-1, 1)).flatten()
    freq_scores_norm = scaler_freq.transform(freq_scores.reshape(-1, 1)).flatten()
    
    # Fusion: Combine scores as per original project's logic (simple sum)
    final_scores = time_scores_norm + freq_scores_norm
    
    # 3. Thresholding: sweep once, then reuse the best threshold
    cached_th = eval_state.get("threshold")
    if cached_th is None:
        # Prefer GPU-accelerated sweep when available
        if prefer_gpu_sweep and (device.type == "cuda"):
            f1, precision, recall, best_th, tp, fp, fn = get_best_f1_gpu(
                final_scores,
                time_labels,
                device=device,
                seq_length=seq_length,
                step=step,
                num_thresholds=num_thresholds,
                min_consecutive=min_consecutive,
                thresholds=None,
                quantile_range=threshold_quantile_range,
                thr_chunk=128,
            )
        else:
            f1, precision, recall, best_th, tp, fp, fn = get_best_f1(
                final_scores,
                time_labels,
                seq_length=seq_length,
                step=step,
                num_thresholds=num_thresholds,
                min_consecutive=min_consecutive,
                thresholds=None,
                quantile_range=threshold_quantile_range,
            )
        eval_state["threshold"] = float(best_th)
    else:
        f1, precision, recall, tp, fp, fn = get_f1_fixed_threshold(
            final_scores,
            time_labels,
            threshold=float(cached_th),
            seq_length=seq_length,
            step=step,
            min_consecutive=min_consecutive,
        )
        best_th = float(cached_th)
    
    # The loss can be the average of the final anomaly scores
    loss = np.mean(final_scores)
    
    print(f"Test Results - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Threshold: {best_th:.6f}, TP: {tp}, FP: {fp}, FN: {fn}")

    return loss, {"f1": f1, "precision": precision, "recall": recall, "threshold": best_th, "tp": tp, "fp": fp, "fn": fn}


def get_weights(net):
    """Get model weights as a list of NumPy ndarrays."""
    # Return a list of all parameters in the model
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
