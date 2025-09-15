"""Array Generation Module for DualTF-FLSim.

This module extracts and adapts the array generation logic from the original
DualTF repository to work with the federated learning setup. It generates
time_evaluation_array.pkl and freq_evaluation_array.pkl files that are
compatible with the original evaluation workflow.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.cuda.amp import autocast

# Import FL components
from dualflsim.task import get_anomaly_scores
from utils.data_loader import _create_sequences


def my_kl_loss(p, q):
    """Numerically stable KL(p||q) per position (copied from task.py)."""
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


def generate_time_evaluation_array(model, testloader_time, device, dataset="PSM", data_num=0, seq_length=100):
    """Generate time domain evaluation array similar to TimeReconstructor.

    Returns array with shape (7, len(test_seq)) and indices:
    ['Normal', 'Anomaly', '#Seq', 'Pred(%)', 'Pred', 'GT', 'Avg(RE)']
    """
    print("Generating time domain evaluation array...")

    time_model = model.time_model
    time_model.to(device)
    time_model.eval()

    criterion = nn.MSELoss(reduction='none')

    # Get anomaly scores and labels from the test set
    test_scores = []
    test_labels = []

    with torch.no_grad():
        for input_data, labels in testloader_time:
            input_data = input_data.float().to(device)
            with autocast(enabled=device.type == 'cuda'):
                output, series, prior, _ = time_model(input_data)

            # Calculate reconstruction loss per sample
            rec_loss = criterion(output, input_data).mean(dim=(1, 2))

            # Calculate association discrepancy
            series_loss = torch.zeros(rec_loss.shape[0], device=device)
            for u in range(len(prior)):
                den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                prior_norm = prior[u] / den
                prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                kl_term_1 = my_kl_loss(series_u, prior_norm.detach())
                kl_term_2 = my_kl_loss(prior_norm.detach(), series_u)
                series_loss += (kl_term_1 + kl_term_2)

            series_loss /= len(prior)

            # Total anomaly score (reconstruction + association discrepancy)
            k = 3.0  # Using k=3.0 as in original code
            anomaly_scores = rec_loss + k * series_loss

            test_scores.append(anomaly_scores.detach().cpu().numpy())

            # Process labels (convert to per-sample labels)
            if isinstance(labels, torch.Tensor):
                lbl = labels
            else:
                lbl = torch.from_numpy(labels)
            if lbl.dim() >= 2:
                lbl = (lbl.to(dtype=torch.float32).max(dim=1).values > 0.5).to(dtype=torch.int64)
            else:
                lbl = (lbl.to(dtype=torch.float32) > 0.5).to(dtype=torch.int64)
            test_labels.append(lbl.cpu().numpy())

    # Concatenate all scores and labels
    all_scores = np.concatenate(test_scores)
    all_labels = np.concatenate(test_labels)

    # Create evaluation array
    test_seq_length = len(all_scores)
    evaluation_array = np.zeros((7, test_seq_length))

    # Create sliding windows for evaluation
    predicted_normal_array = np.zeros(test_seq_length)
    predicted_anomaly_array = np.zeros(test_seq_length)
    rec_error_array = np.zeros(test_seq_length)

    # Calculate threshold for anomaly detection (using percentile approach)
    threshold = np.percentile(all_scores, 100 - 1.0)  # top 1% as anomalies

    # Process each timestamp
    for ts in range(test_seq_length):
        # Number of sequences this timestamp belongs to
        if ts < seq_length - 1:
            num_context = ts + 1
        elif ts >= seq_length - 1 and ts < test_seq_length - seq_length + 1:
            num_context = seq_length
        elif ts >= test_seq_length - seq_length + 1:
            num_context = test_seq_length - ts
        else:
            num_context = seq_length

        evaluation_array[2][ts] = num_context  # '#Seq'
        rec_error_array[ts] = all_scores[ts]

        # Predict anomaly based on threshold
        if all_scores[ts] > threshold:
            predicted_anomaly_array[ts] += 1
        else:
            predicted_normal_array[ts] += 1

    # Fill evaluation array
    evaluation_array[0] = predicted_normal_array  # 'Normal'
    evaluation_array[1] = predicted_anomaly_array  # 'Anomaly'
    evaluation_array[6] = rec_error_array / evaluation_array[2]  # 'Avg(RE)'

    # Calculate prediction percentage and binary predictions
    for s in range(test_seq_length):
        if evaluation_array[2][s] > 0:
            evaluation_array[3][s] = evaluation_array[1][s] / evaluation_array[2][s]  # 'Pred(%)'
            if evaluation_array[3][s] > 0.5:
                evaluation_array[4][s] = 1  # 'Pred'

    evaluation_array[5] = all_labels  # 'GT'

    print(f'Time Evaluation Array Shape: {evaluation_array.shape}')

    # Convert to DataFrame
    df = pd.DataFrame(evaluation_array)
    df.index = ['Normal', 'Anomaly', '#Seq', 'Pred(%)', 'Pred', 'GT', 'Avg(RE)']
    df = df.astype('float')

    return df


def generate_freq_evaluation_array(model, testloader_freq, device, dataset="PSM", data_num=0, seq_length=100, nest_length=25):
    """Generate frequency domain evaluation array similar to FreqReconstructor.

    Returns array with shape (5, len(test_seq)) and indices:
    ['#SubSeq', '#GrandSeq', 'Avg(exp(RE))', 'Pred', 'GT']
    """
    print("Generating frequency domain evaluation array...")

    freq_model = model.freq_model
    freq_model.to(device)
    freq_model.eval()

    criterion = nn.MSELoss(reduction='none')

    # Get anomaly scores and labels from the test set
    test_scores = []
    test_labels = []

    with torch.no_grad():
        for input_data, labels in testloader_freq:
            input_data = input_data.float().to(device)
            with autocast(enabled=device.type == 'cuda'):
                output, series, prior, _ = freq_model(input_data)

            # Calculate reconstruction loss per sample
            rec_loss = criterion(output, input_data).mean(dim=(1, 2))

            # Calculate association discrepancy (frequency domain specific)
            series_loss = torch.zeros(rec_loss.shape[0], device=device)
            win_size = (seq_length - nest_length + 1) * (nest_length // 2)

            for u in range(len(prior)):
                den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                prior_norm = prior[u] / den
                prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                kl_term_1 = my_kl_loss(series_u, prior_norm.detach())
                kl_term_2 = my_kl_loss(prior_norm.detach(), series_u)
                series_loss += (kl_term_1 + kl_term_2)

            series_loss /= len(prior)

            # Use exponential of the anomaly score (as in original freq code)
            anomaly_scores = np.exp(rec_loss.detach().cpu().numpy())

            test_scores.append(anomaly_scores)

            # Process labels
            if isinstance(labels, torch.Tensor):
                lbl = labels
            else:
                lbl = torch.from_numpy(labels)
            if lbl.dim() >= 2:
                lbl = (lbl.to(dtype=torch.float32).max(dim=1).values > 0.5).to(dtype=torch.int64)
            else:
                lbl = (lbl.to(dtype=torch.float32) > 0.5).to(dtype=torch.int64)
            test_labels.append(lbl.cpu().numpy())

    # Concatenate all scores and labels
    all_scores = np.concatenate(test_scores)
    all_labels = np.concatenate(test_labels)

    # For frequency domain, we need to map back to the original sequence length
    # Since freq data is shorter, we'll interpolate/expand to match original length
    original_length = len(all_labels) * seq_length // len(all_scores) if len(all_scores) > 0 else seq_length

    # Create evaluation array for the mapped sequence
    grand_evaluation_array = np.zeros((5, original_length))

    # Map frequency scores to sequence positions
    for i in range(len(all_scores)):
        start_pos = i
        end_pos = min(start_pos + seq_length, original_length)
        grand_evaluation_array[2][start_pos:end_pos] += all_scores[i]  # 'Avg(exp(RE))'
        grand_evaluation_array[0][start_pos:end_pos] += nest_length  # '#SubSeq'

    # Calculate grand sequence counts
    for timestamp in range(original_length):
        if timestamp < seq_length - 1:
            grand_context = timestamp + 1
        elif timestamp >= seq_length - 1 and timestamp < original_length - seq_length + 1:
            grand_context = seq_length
        elif timestamp >= original_length - seq_length + 1:
            grand_context = original_length - timestamp
        else:
            grand_context = seq_length
        grand_evaluation_array[1][timestamp] = grand_context  # '#GrandSeq'

    # Normalize the scores
    for s in range(original_length):
        if grand_evaluation_array[1][s] > 0:
            grand_evaluation_array[2][s] = grand_evaluation_array[2][s] / grand_evaluation_array[1][s]

    # Generate predictions based on mean threshold
    mean_score = np.mean(grand_evaluation_array[2])
    for s in range(original_length):
        if grand_evaluation_array[2][s] > mean_score:
            grand_evaluation_array[3][s] = 1  # 'Pred'

    # Set ground truth (expand labels to match original length)
    if len(all_labels) > 0:
        expanded_labels = np.repeat(all_labels, original_length // len(all_labels))
        if len(expanded_labels) < original_length:
            expanded_labels = np.pad(expanded_labels, (0, original_length - len(expanded_labels)), 'edge')
        grand_evaluation_array[4] = expanded_labels[:original_length]  # 'GT'

    print(f'Freq Evaluation Array Shape: {grand_evaluation_array.shape}')

    # Convert to DataFrame
    df = pd.DataFrame(grand_evaluation_array)
    df.index = ['#SubSeq', '#GrandSeq', 'Avg(exp(RE))', 'Pred', 'GT']
    df = df.astype('float')

    return df


def save_evaluation_arrays(time_df, freq_df, dataset="PSM", form=None, data_num=0):
    """Save the evaluation arrays as pickle files in the same format as original."""

    # Create directories
    time_save_path = './time_arrays'
    freq_save_path = './freq_arrays'

    if not os.path.exists(time_save_path):
        os.makedirs(time_save_path)
    if not os.path.exists(freq_save_path):
        os.makedirs(freq_save_path)

    # Save time array
    if dataset == 'NeurIPSTS' and form is not None:
        time_file_path = f'{time_save_path}/{dataset}_{form}_time_evaluation_array.pkl'
        freq_file_path = f'{freq_save_path}/{dataset}_{form}_freq_evaluation_array.pkl'
    else:
        time_file_path = f'{time_save_path}/{dataset}_{data_num}_time_evaluation_array.pkl'
        freq_file_path = f'{freq_save_path}/{dataset}_{data_num}_freq_evaluation_array.pkl'

    print(f'Saving Time Array to: {time_file_path}')
    time_df.to_pickle(time_file_path)

    print(f'Saving Freq Array to: {freq_file_path}')
    freq_df.to_pickle(freq_file_path)

    print("Evaluation arrays saved successfully!")
    return time_file_path, freq_file_path


def generate_evaluation_arrays(model, testloader_time, testloader_freq, device,
                               dataset="PSM", form=None, data_num=0, seq_length=100, nest_length=25):
    """Main function to generate both time and frequency evaluation arrays."""

    print(f"=== Generating Evaluation Arrays for {dataset} ===")
    start_time = time.time()

    # Generate time domain array
    time_df = generate_time_evaluation_array(
        model=model,
        testloader_time=testloader_time,
        device=device,
        dataset=dataset,
        data_num=data_num,
        seq_length=seq_length
    )

    # Generate frequency domain array
    freq_df = generate_freq_evaluation_array(
        model=model,
        testloader_freq=testloader_freq,
        device=device,
        dataset=dataset,
        data_num=data_num,
        seq_length=seq_length,
        nest_length=nest_length
    )

    # Save both arrays
    time_path, freq_path = save_evaluation_arrays(
        time_df=time_df,
        freq_df=freq_df,
        dataset=dataset,
        form=form,
        data_num=data_num
    )

    print(f"=== Array Generation Complete in {time.time() - start_time:.2f}s ===")

    return time_path, freq_path, time_df, freq_df