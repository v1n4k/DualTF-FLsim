"""DualFLSim: A Flower / PyTorch app."""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
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
from utils.data_loader import load_tods, TrainingLoader, GeneralLoader, generate_frequency_grandwindow, _create_sequences, load_PSM


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
    freq_batch_size: int = 8,
    seq_length: int = 100,
    dirichlet_alpha: float = 2.0,
):
    """Load and partition the time-series data."""
    # Load the full dataset using the function from the original project
    # The data_loader expects a './datasets' folder in the CWD.
    # The run_simulation.py script is in the parent directory, so this path should resolve correctly.
    data_dict = load_PSM(seq_length=seq_length, stride=1)
    # PSM loader returns lists, we take the first element
    x_train_full, x_test, y_test = data_dict['x_train'][0], data_dict['x_test'][0], data_dict['y_test'][0]

    # Standardization: z-score per feature before training
    mean = np.mean(x_train_full, axis=0)
    std = np.std(x_train_full, axis=0) + 1e-6  # Avoid division by zero
    x_train_full = (x_train_full - mean) / std
    x_test = (x_test - mean) / std

    # --- Federated Partitioning (Time Domain) ---
    # Partition the training data using a quantity-based Dirichlet distribution
    train_partitions = partition_data_dirichlet_quantity(
        x_train_full, num_partitions, alpha=dirichlet_alpha
    )
    x_train_partition = train_partitions[partition_id]
    
    print(f"Client {partition_id}: Loading {len(x_train_partition)} time-domain training samples.")

    # Create Time-Domain DataLoaders
    train_dataset_time = TrainingLoader(x_train_partition)
    trainloader_time = DataLoader(dataset=train_dataset_time, batch_size=time_batch_size, shuffle=True, num_workers=0)
    test_dataset_time = GeneralLoader(x_test, y_test)
    testloader_time = DataLoader(dataset=test_dataset_time, batch_size=time_batch_size, shuffle=False, num_workers=0)

    # --- Federated Partitioning (Frequency Domain) ---
    # Use a default nest_length from the original project
    nest_length = 25
    # Generate frequency data for the client's partition
    freq_dict = generate_frequency_grandwindow(x_train_partition, x_test, y_test, nest_length, step=1)
    x_train_freq_partition = freq_dict['grand_train_reshaped']
    x_test_freq = freq_dict['grand_test_reshaped']
    y_test_freq = freq_dict['grand_label_reshaped']

    print(f"Client {partition_id}: Loading {len(x_train_freq_partition)} frequency-domain training samples.")

    # Create Frequency-Domain DataLoaders
    train_dataset_freq = TrainingLoader(x_train_freq_partition)
    trainloader_freq = DataLoader(dataset=train_dataset_freq, batch_size=freq_batch_size, shuffle=True, num_workers=0)
    test_dataset_freq = GeneralLoader(x_test_freq, y_test_freq)
    testloader_freq = DataLoader(dataset=test_dataset_freq, batch_size=freq_batch_size, shuffle=False, num_workers=0)
    
    return trainloader_time, testloader_time, trainloader_freq, testloader_freq


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def train(net, trainloader_time, trainloader_freq, epochs, device, proximal_mu, k=3.0, lr=1e-5, control_c=None, control_ci=None):
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
                    series_loss += (
                        torch.mean(
                            my_kl_loss(
                                series[u],
                                prior_norm.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                prior_norm.detach(),
                                series[u],
                            )
                        )
                    )
                    prior_loss += (
                        torch.mean(
                            my_kl_loss(
                                prior_norm,
                                series[u].detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                series[u].detach(),
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
            if c_time is not None and ci_time is not None:
                for p, c_k, ci_k in zip(params_time, c_time, ci_time):
                    if p.grad is not None:
                        p.grad.add_(c_k - ci_k)
            time_optimizer.step()
            scaler.update()
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
                    series_loss += (
                        torch.mean(
                            my_kl_loss(
                                series[u],
                                prior_norm.detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                prior_norm.detach(),
                                series[u],
                            )
                        )
                    )
                    prior_loss += (
                        torch.mean(
                            my_kl_loss(
                                prior_norm,
                                series[u].detach(),
                            )
                        )
                        + torch.mean(
                            my_kl_loss(
                                series[u].detach(),
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
            params_freq = list(freq_model.parameters())
            if c_tensors_all is not None:
                c_freq = c_tensors_all[n_time:]
                ci_freq = ci_tensors_all[n_time:]
            else:
                c_freq = ci_freq = None
            if c_freq is not None and ci_freq is not None:
                for p, c_k, ci_k in zip(params_freq, c_freq, ci_freq):
                    if p.grad is not None:
                        p.grad.add_(c_k - ci_k)
            freq_optimizer.step()
            scaler.update()
            total_steps += 1

    # For simplicity, we'll just return 0.0 as a placeholder loss for now.
    # A more sophisticated approach would be to combine losses.
    return 0.0, total_steps


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
            
            # Association discrepancy for each sample
            series_loss = torch.zeros(rec_loss.shape[0], device=device)
            for u in range(len(prior)):
                win_size = model.win_size
                kl_term_1 = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach())
                kl_term_2 = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, win_size)).detach(), series[u])
                series_loss += (kl_term_1 + kl_term_2).mean(dim=1)
            
            series_loss /= len(prior)
            
            # Anomaly score (as defined in the paper's loss function)
            anomaly_score = rec_loss - 3.0 * series_loss # Using k=3.0 as in train
            
            scores.append(anomaly_score.cpu().numpy())
            labels.append(target.cpu().numpy())
            
    return np.concatenate(scores), np.concatenate(labels)


def get_best_f1(scores, labels, seq_length=100, step=5, num_thresholds=201):
    """Find the best F1 score by simulating thresholds."""
    
    # Create sequences for point-adjusted evaluation
    score_sequences = _create_sequences(scores, seq_length, step)
    label_sequences = _create_sequences(labels, seq_length, step)

    min_score, max_score = np.min(scores), np.max(scores)
    thresholds = np.linspace(min_score, max_score, num=num_thresholds)
    
    best_f1, best_p, best_r = 0, 0, 0

    for th in thresholds:
        tp, fp, fn = 0, 0, 0
        for i in range(len(score_sequences)):
            seq_scores = score_sequences[i]
            seq_labels = label_sequences[i]
            
            pred_anomalies = set(np.where(seq_scores > th)[0])
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
        
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, precision, recall
            
    return best_f1, best_p, best_r


def test(net, testloader_time, testloader_freq, device):
    """Validate the complete DualTF model and find the best F1 score."""
    
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
    scaler = MinMaxScaler()
    time_scores_norm = scaler.fit_transform(time_scores.reshape(-1, 1)).flatten()
    freq_scores_norm = scaler.fit_transform(freq_scores.reshape(-1, 1)).flatten()
    
    final_scores = time_scores_norm + freq_scores_norm
    
    # 3. Find best F1 score
    f1, precision, recall = get_best_f1(final_scores, time_labels)
    
    # The loss can be the average of the final anomaly scores
    loss = np.mean(final_scores)
    
    print(f"Test Results - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return loss, {"f1": f1, "precision": precision, "recall": recall}


def partition_data_dirichlet_quantity(data, num_partitions, alpha=0.5, seed=42):
    """Partitions data into different quantities based on a Dirichlet distribution."""
    np.random.seed(seed)  # Ensure deterministic partitioning across clients
    num_samples = len(data)

    # Get proportions for data quantity
    proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))

    # Calculate the number of samples for each partition
    samples_per_partition = np.round(proportions * num_samples).astype(int)

    # Ensure the sum is correct due to rounding
    samples_per_partition[-1] = num_samples - np.sum(samples_per_partition[:-1])

    # Shuffle data indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Create the partitions
    partitions = []
    start = 0
    for num_samples_in_partition in samples_per_partition:
        end = start + num_samples_in_partition
        partition_indices = indices[start:end]
        partitions.append(data[partition_indices])
        start = end

    return partitions


def get_weights(net):
    """Get model weights as a list of NumPy ndarrays."""
    # Return a list of all parameters in the model
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
