"""CPU-Optimized Evaluation for DualTF-FLSim.

This file preserves the exact evaluation logic of `evaluation_fl.py`, but
optimizes the computation using NumPy vectorization and batched processing
to accelerate CPU performance. No metric definition or selection logic is
changed; only the way intermediate counts are computed is optimized.
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import math
import sklearn

from sklearn.metrics import auc
from sklearn.preprocessing import RobustScaler

# Add the project root to Python path for imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data_loader import load_PSM, load_SMD_raw, _load_smap_msl_combined, normalization, _create_sequences


# === TranAD-style adjusted prediction utilities (copied to preserve logic) ===
def adjust_predicts_from_tranad(label, score, threshold=None, pred=None, calc_latency=False):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred.copy()
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def get_threshold_tranad(labels, scores, print_or_not=True):
    auc_val = sklearn.metrics.roc_auc_score(labels, scores)
    thresholds_0 = np.asarray(scores).copy()
    thresholds_0.sort()

    thresholds = []
    for i in range(len(thresholds_0)):
        if i % 1000 == 0 or i == len(thresholds_0) - 1:
            thresholds.append(thresholds_0[i])

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_threshold = math.inf
    best_f1_adjusted = 0.0
    best_precision_adjusted = 0.0
    best_recall_adjusted = 0.0

    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores).astype(float)

    for threshold in thresholds:
        y_pred_from_threshold = (scores_np >= threshold).astype(int)
        precision = sklearn.metrics.precision_score(labels_np, y_pred_from_threshold, zero_division=0)
        recall = sklearn.metrics.recall_score(labels_np, y_pred_from_threshold, zero_division=0)
        f1 = sklearn.metrics.f1_score(labels_np, y_pred_from_threshold, zero_division=0)

        y_pred_adjusted = adjust_predicts_from_tranad(labels_np, scores_np, pred=y_pred_from_threshold, threshold=threshold)
        precision_adjusted = sklearn.metrics.precision_score(labels_np, y_pred_adjusted, zero_division=0)
        recall_adjusted = sklearn.metrics.recall_score(labels_np, y_pred_adjusted, zero_division=0)
        f1_adjusted = sklearn.metrics.f1_score(labels_np, y_pred_adjusted, zero_division=0)

        if f1_adjusted > best_f1_adjusted:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_f1_adjusted = f1_adjusted
            best_precision_adjusted = precision_adjusted
            best_recall_adjusted = recall_adjusted
            best_threshold = threshold

    if print_or_not:
        print('auc:', auc_val)
        print('precision_adjusted:', best_precision_adjusted)
        print('recall_adjusted:', best_recall_adjusted)
        print('f1:', best_f1)
        print('f1_adjusted:', best_f1_adjusted)
        print('threshold:', best_threshold)

    return (
        auc_val,
        best_precision,
        best_recall,
        best_f1,
        best_precision_adjusted,
        best_recall_adjusted,
        best_f1_adjusted,
        best_threshold,
    )


def _simulate_thresholds(rec_errors, n):
    thresholds, step_size = [], abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)

    print(f'Threshold Range: ({np.min(rec_errors)}, {np.max(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        th = th + step_size
        thresholds.append(th)

    return thresholds


def load_evaluation_arrays(dataset="PSM", form=None, data_num=0):
    # Time array
    print('Time Arrays Loading...')
    if dataset == 'NeurIPSTS' and form is not None:
        time_file_path = f'./time_arrays/{dataset}_{form}_time_evaluation_array.pkl'
    else:
        time_file_path = f'./time_arrays/{dataset}_{data_num}_time_evaluation_array.pkl'

    try:
        time_array = pd.read_pickle(time_file_path)
        print(f"Time array loaded: {time_array.shape}")
        print("Time array indices:", list(time_array.index))
    except FileNotFoundError:
        print(f"Error: Time array file not found at {time_file_path}")
        print("Please run FL training first to generate the arrays.")
        return None, None

    # Frequency array
    print('Frequency Arrays Loading...')
    if dataset == 'NeurIPSTS' and form is not None:
        freq_file_path = f'./freq_arrays/{dataset}_{form}_freq_evaluation_array.pkl'
    else:
        freq_file_path = f'./freq_arrays/{dataset}_{data_num}_freq_evaluation_array.pkl'

    try:
        freq_array = pd.read_pickle(freq_file_path)
        print(f"Freq array loaded: {freq_array.shape}")
        print("Freq array indices:", list(freq_array.index))
    except FileNotFoundError:
        print(f"Error: Freq array file not found at {freq_file_path}")
        print("Please run FL training first to generate the arrays.")
        return None, None

    return time_array, freq_array


def _reconstruct_1d_labels_from_windows(y_windows: np.ndarray, step: int) -> np.ndarray:
    """Reconstruct 1D label vector from windowed labels using max aggregation.

    This preserves the semantics of binary anomaly labels across overlapping
    windows. Assumes y_windows has shape (num_windows, seq_len) and step>=1.
    """
    if y_windows.ndim != 2:
        raise ValueError("y_windows must be 2D: (num_windows, seq_len)")
    num_windows, seq_len = y_windows.shape
    if step <= 0:
        step = 1

    # Original length for step=1 is num_windows + seq_len - 1
    out_len = (num_windows - 1) * step + seq_len
    y_1d = np.zeros(out_len, dtype=y_windows.dtype)

    # Max-aggregate overlapping regions
    for i in range(num_windows):
        start = i * step
        end = start + seq_len
        np.maximum(y_1d[start:end], y_windows[i], out=y_1d[start:end])

    return y_1d


def _batched_window_confusion_counts(s_seq, y_seq, thresholds, batch_size=256):
    """Compute TP/TN/FP/FN across window sequences for all thresholds in batches.

    Logic strictly matches the original: a window counts as TP if both predicted
    anomalies and true anomalies exist in the window and intersect; TN if neither
    exists; FP if predicted exists but true does not; FN if true exists but predicted
    does not.

    This implementation batches over both thresholds and windows to avoid
    constructing gigantic intermediate tensors, and uses explicit expansion to
    prevent unintended outer-product broadcasting.
    """
    eps = 1e-7
    num_win, seq_len = s_seq.shape

    # Total accumulators per-threshold across all window batches
    TP_tot = np.zeros(len(thresholds), dtype=np.int64)
    TN_tot = np.zeros(len(thresholds), dtype=np.int64)
    FP_tot = np.zeros(len(thresholds), dtype=np.int64)
    FN_tot = np.zeros(len(thresholds), dtype=np.int64)

    # Window batch size chosen to balance memory and speed
    window_batch_size = 87000

    for w_start in range(0, num_win, window_batch_size):
        w_end = min(w_start + window_batch_size, num_win)
        s_chunk = np.asarray(s_seq[w_start:w_end])               # expect (m, L)
        y_chunk = np.asarray(y_seq[w_start:w_end])               # expect (m, L)
        # Enforce 2D shapes to avoid accidental higher-rank broadcasting
        if s_chunk.ndim != 2:
            s_chunk = s_chunk.reshape(s_chunk.shape[0], s_chunk.shape[1])
        if y_chunk.ndim != 2:
            y_chunk = y_chunk.reshape(y_chunk.shape[0], y_chunk.shape[1])

        has_true = (y_chunk == 1).any(axis=1)        # (m,)
        y_bin = (y_chunk == 1)                       # (m, L)

        for t_start in range(0, len(thresholds), batch_size):
            t_end = min(t_start + batch_size, len(thresholds))
            th_batch = thresholds[t_start:t_end]     # (b,)

            # Allocate block accumulators (b,)
            b = th_batch.shape[0]
            TP_block = np.zeros(b, dtype=np.int64)
            TN_block = np.zeros(b, dtype=np.int64)
            FP_block = np.zeros(b, dtype=np.int64)
            FN_block = np.zeros(b, dtype=np.int64)

            # Loop over thresholds within this block to keep memory bounded
            for j, th in enumerate(th_batch):
                pred2d = (s_chunk > th)                     # (m, L)
                has_pred = pred2d.any(axis=1)               # (m,)
                inter_any = (pred2d & y_bin).any(axis=1)    # (m,)

                TP_block[j] = int(inter_any.sum())
                TN_block[j] = int(((~has_pred) & (~has_true)).sum())
                FP_block[j] = int((has_pred & (~has_true)).sum())
                FN_block[j] = int(((~has_pred) & has_true).sum())

            TP_tot[t_start:t_end] += TP_block
            TN_tot[t_start:t_end] += TN_block
            FP_tot[t_start:t_end] += FP_block
            FN_tot[t_start:t_end] += FN_block

    precision = TP_tot / (TP_tot + FP_tot + eps)
    recall = TP_tot / (TP_tot + FN_tot + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    fpr = FP_tot / (FP_tot + TN_tot + eps)

    return TP_tot, TN_tot, FP_tot, FN_tot, precision, recall, f1, fpr


def _vectorized_pointwise_counts(s, y, thresholds, batch_size=4096):
    """Vectorized point-wise confusion counts across thresholds in batches."""
    eps = 1e-8
    y_pos = (y == 1)
    y_neg = ~y_pos

    TP_all = []
    TN_all = []
    FP_all = []
    FN_all = []

    for start in range(0, len(thresholds), batch_size):
        end = min(start + batch_size, len(thresholds))
        th_batch = thresholds[start:end]  # (b,)
        # Predictions matrix: (n, b)
        y_pred = s[:, None] >= th_batch[None, :]

        tp = (y_pos[:, None] & y_pred).sum(axis=0)
        tn = (y_neg[:, None] & (~y_pred)).sum(axis=0)
        fp = (y_neg[:, None] & y_pred).sum(axis=0)
        fn = (y_pos[:, None] & (~y_pred)).sum(axis=0)

        TP_all.append(tp)
        TN_all.append(tn)
        FP_all.append(fp)
        FN_all.append(fn)

    TP = np.concatenate(TP_all)
    TN = np.concatenate(TN_all)
    FP = np.concatenate(FP_all)
    FN = np.concatenate(FN_all)

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    fpr = FP / (FP + TN + eps)

    return TP, TN, FP, FN, precision, recall, f1, fpr


def total_evaluation(opts):
    # Load arrays
    time_array, freq_array = load_evaluation_arrays(
        dataset=opts.dataset,
        form=getattr(opts, 'form', None),
        data_num=getattr(opts, 'data_num', 0)
    )

    if time_array is None or freq_array is None:
        return

    if opts.verbose:
        print("Time Array:")
        print(time_array)
        print("\nFreq Array:")
        print(freq_array)

    # Extract reconstruction errors for fusion
    time_rec = np.array(time_array.loc['Avg(RE)', :]).reshape(-1, 1)
    freq_rec = np.array(freq_array.loc['Avg(exp(RE))', :]).reshape(-1, 1)

    # Align lengths if mismatched
    if len(time_rec) != len(freq_rec):
        longer = len(time_rec) if len(time_rec) >= len(freq_rec) else len(freq_rec)
        shorter_name = 'freq' if len(freq_rec) < len(time_rec) else 'time'
        print(f"[WARN] Length mismatch between time ({len(time_rec)}) and freq ({len(freq_rec)}) arrays. Resampling {shorter_name} to length {longer} to align with original evaluation.")
        if len(freq_rec) < len(time_rec):
            x_old = np.linspace(0.0, 1.0, len(freq_rec))
            x_new = np.linspace(0.0, 1.0, len(time_rec))
            freq_rec = np.interp(x_new, x_old, freq_rec.ravel()).reshape(-1, 1)
        else:
            x_old = np.linspace(0.0, 1.0, len(time_rec))
            x_new = np.linspace(0.0, 1.0, len(freq_rec))
            time_rec = np.interp(x_new, x_old, time_rec.ravel()).reshape(-1, 1)

    # Normalize scores as in original
    scaler = RobustScaler(unit_variance=True)
    time_as = scaler.fit_transform(time_rec)
    freq_as = scaler.transform(freq_rec)

    time_as = normalization(time_rec)
    freq_as = normalization(freq_rec)

    final_as = (time_as + freq_as).flatten()  # fused
    time_only_as = time_as.flatten()
    freq_only_as = freq_as.flatten()

    # Load ground truth labels
    print("Loading ground truth labels...")
    ds_upper = opts.dataset.upper()
    if ds_upper == 'PSM':
        data_dict = load_PSM()
        label_segments = data_dict['label_segments']
    elif ds_upper == 'SMD':
        data_dict = load_SMD_raw()
        label_segments = data_dict['label_segments']
    elif ds_upper in {'SMAP', 'MSL'}:
        data_dict = _load_smap_msl_combined(dataset_name=ds_upper)
        label_segments = data_dict['label_segments']
    else:
        print(f"Dataset {opts.dataset} not implemented yet")
        return

    label = np.concatenate(label_segments, axis=0)

    # Helper to evaluate a single score vector with all methods
    def evaluate_scores(scores_1d: np.ndarray, labels_1d: np.ndarray):
        out = {}
        # Align lengths
        # labels_1d may come as windowed (num_win, 1) or (num_win, seq_len, 1) depending on loader
        y_input = labels_1d
        # Normalize to 1D raw labels before creating sequences below
        if y_input.ndim == 3 and y_input.shape[-1] == 1:
            # Likely (num_win, seq_len, 1) from some loaders
            y_flat = _reconstruct_1d_labels_from_windows(y_input.squeeze(-1), step=opts.step)
        elif y_input.ndim == 2 and (y_input.shape[0] > 1 and y_input.shape[1] > 1):
            # Likely (num_win, seq_len)
            y_flat = _reconstruct_1d_labels_from_windows(y_input, step=opts.step)
        else:
            y_flat = y_input.squeeze()

        n = min(len(scores_1d), len(y_flat))
        s = scores_1d[:n]
        y = y_flat[:n]

        # Prepare thresholds (same logic)
        thresholds_list = _simulate_thresholds(s, opts.thresh_num)
        thresholds = np.asarray(thresholds_list, dtype=float)

        # Point Adjusted (seq_length) - vectorized windows
        s_seq = _create_sequences(s, opts.seq_length, opts.step)
        y_seq = _create_sequences(y, opts.seq_length, opts.step)
        TP, TN, FP, FN, precision, recall, f1, fpr = _batched_window_confusion_counts(
            s_seq, y_seq, thresholds, batch_size=opts.thresh_batch
        )
        idx = int(np.argmax(f1))
        out['point_adjusted'] = {
            'threshold': float(thresholds[idx]),
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'pr_auc': float(auc(recall, precision)),
            'roc_auc': float(auc(fpr, recall)),
        }

        # Point-Wise - fully vectorized
        TP, TN, FP, FN, precision, recall, f1, fpr = _vectorized_pointwise_counts(
            s, y.astype(bool), thresholds, batch_size=opts.thresh_batch * 8
        )
        idx = int(np.argmax(f1))
        out['point_wise'] = {
            'threshold': float(thresholds[idx]),
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'pr_auc': float(auc(recall, precision)),
            'roc_auc': float(auc(fpr, recall)),
        }

        # Released Point-Wise (nest_length) - vectorized windows
        s_seq_rel = _create_sequences(s, opts.nest_length, opts.step)
        y_seq_rel = _create_sequences(y, opts.nest_length, opts.step)
        TP, TN, FP, FN, precision, recall, f1, fpr = _batched_window_confusion_counts(
            s_seq_rel, y_seq_rel, thresholds, batch_size=opts.thresh_batch
        )
        idx = int(np.argmax(f1))
        out['released_point_wise'] = {
            'threshold': float(thresholds[idx]),
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1': float(f1[idx]),
            'pr_auc': float(auc(recall, precision)),
            'roc_auc': float(auc(fpr, recall)),
        }

        # TranAD-adjusted (preserve original loop-based logic)
        auc_val, p_raw, r_raw, f1_raw, p_adj, r_adj, f1_adj, th_adj = get_threshold_tranad(
            labels=y.astype(int), scores=s, print_or_not=False
        )
        out['tranad_adjusted'] = {
            'threshold': float(th_adj),
            'precision': float(p_adj),
            'recall': float(r_adj),
            'f1': float(f1_adj),
            'pr_auc': float('nan'),
            'roc_auc': float(auc_val),
        }

        return out, n

    # Build labels once (aligned with loaded arrays): use GT row from time_array
    labels_full = np.asarray(time_array.loc['GT', :]).astype(int).reshape(-1)

    # Decide which modes to run
    wanted_modes = []
    mode = (opts.mode or 'all').lower()
    if mode == 'all':
        wanted_modes = ['time', 'freq', 'fused']
    else:
        if mode not in ['time', 'freq', 'fused']:
            print(f"Unknown mode {mode}, defaulting to fused")
            wanted_modes = ['fused']
        else:
            wanted_modes = [mode]

    # Prepare output accumulator
    rows = []
    print(f"\nStart CPU-optimized evaluation for {len(wanted_modes)} modes: {wanted_modes}")
    for m in tqdm(wanted_modes, desc="Evaluate Mode", unit="mode"):
        if m == 'time':
            scores = time_only_as
        elif m == 'freq':
            scores = freq_only_as
        else:
            scores = final_as

        res, n_used = evaluate_scores(scores, labels_full)

        print(f"\n=== Evaluate Mode: {m} (N={n_used}) ===")
        for k, v in res.items():
            print(f"-- {k} --")
            print("Threshold: {:.6f}".format(v['threshold']))
            print("Precision : {:0.4f}, Recall : {:0.4f}, F1 : {:0.4f}".format(v['precision'], v['recall'], v['f1']))
            if not (np.isnan(v['pr_auc'])):
                print("PR-AUC : {:0.4f}".format(v['pr_auc']))
            print("ROC-AUC : {:0.4f}".format(v['roc_auc']))

            rows.append({
                'dataset': opts.dataset,
                'data_num': opts.data_num,
                'mode': m,
                'method': k,
                'seq_length': opts.seq_length,
                'nest_length': opts.nest_length,
                'step': opts.step,
                'thresh_num': opts.thresh_num,
                'threshold': v['threshold'],
                'precision': v['precision'],
                'recall': v['recall'],
                'f1': v['f1'],
                'pr_auc': v['pr_auc'],
                'roc_auc': v['roc_auc'],
                'n_used': n_used,
            })

    # Optionally save CSV
    if not opts.no_save_csv:
        results_dir = Path(opts.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{opts.dataset}_{opts.data_num}_results_cpu_vectorization.csv"
        df_out = pd.DataFrame(rows)
        if out_path.exists():
            old = pd.read_csv(out_path)
            df_out = pd.concat([old, df_out], ignore_index=True)
        df_out.to_csv(out_path, index=False)
        print(f"\nSaved consolidated results to: {out_path}")

    print("\n=== FL CPU Evaluation Complete ===")


def main():
    parser = argparse.ArgumentParser(description='CPU-Optimized FL Evaluation for DualTF')

    # Settings (keep parity with evaluation_fl.py)
    parser.add_argument('--thresh_num', type=int, default=1000,
                        help='Number of thresholds for evaluation')
    parser.add_argument('--seq_length', type=int, default=75,
                        help='Sequence length used during training')
    parser.add_argument('--nest_length', type=int, default=25,
                        help='Nested sequence length')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size for sequence creation')
    parser.add_argument('--dataset', type=str, default='SMD',
                        help='Dataset name')
    parser.add_argument('--form', type=str, default=None,
                        help='Form for NeurIPSTS dataset')
    parser.add_argument('--data_num', type=int, default=1,
                        help='Data number')
    parser.add_argument('--mode', type=str, default='fused',
                        help='Which scores to evaluate: time|freq|fused|all')
    parser.add_argument('--results_dir', type=str, default='./eval_results',
                        help='Directory to save consolidated CSV results')
    parser.add_argument('--no_save_csv', action='store_true',
                        help='If set, do not write consolidated CSV')
    parser.add_argument('--verbose', action='store_true',
                        help='Print loaded arrays for inspection')

    # CPU-optimization parameter: batch size for threshold vectorization
    parser.add_argument('--thresh_batch', type=int, default=2048,
                        help='Batch size for threshold-wise vectorization to control memory')

    opts = parser.parse_args()

    if opts.dataset == 'NeurIPSTS' and opts.form:
        print(f"Dataset: {opts.dataset}\nForm: {opts.form}\nSeq_length: {opts.seq_length}\nNest_length: {opts.nest_length}")
    else:
        print(f"Dataset: {opts.dataset}\nNum: {opts.data_num}\nSeq_length: {opts.seq_length}\nNest_length: {opts.nest_length}")

    total_evaluation(opts)


if __name__ == '__main__':
    main()


