import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import groupby
from operator import itemgetter
from tqdm.notebook import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GeneralLoader(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, index): 
        x = torch.FloatTensor(self.x[index,:,:])
        y = torch.FloatTensor(self.y[index,:,:])
        return x, y

class TrainingLoader(Dataset):
    def __init__(self,x):
        self.x = x
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, index): 
        x = torch.FloatTensor(self.x[index,:,:])
        return x

def normalization(x):
    min_value = min(x)
    max_value = max(x)

    return np.array(list(map(lambda x: 1*(x-min_value)/(max_value-min_value), x)))

# Generated fast fourier transformed sequences.
def torch_fft_transform(seq):
    torch_seq = torch.from_numpy(seq)
    # freq length
    tp_cnt = seq.shape[1]
    tm_period = seq.shape[1]
    
    # FFT
    ft_ = torch.fft.fft(torch_seq, dim=1) / tm_period
    # Half
    ft_ = ft_[:, range(int(tm_period/2)), :]
    # index
    val_ = np.arange(int(tp_cnt/2))
    # freq axis
    freq = val_ / tm_period
    
    ffts_tensor = abs(ft_)
    ffts = ffts_tensor.numpy()
    return ffts, freq

# Generated training sequences for use in the model.
def _create_sequences(values, seq_length, stride, historical=False):
    seq = []
    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i-seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i : i + seq_length])
   
    return np.stack(seq)

def _count_anomaly_segments(values):
    values = np.where(values == 1)[0]
    anomaly_segments = []
    
    for k, g in groupby(enumerate(values), lambda ix : ix[0] - ix[1]):
        anomaly_segments.append(list(map(itemgetter(1), g)))
    return len(anomaly_segments), anomaly_segments

def generate_UCR_frequency_grandwindow(x_trains, x_tests, y_tests, nest_length, step):
    # x_trains: (250, N, seq_length, stride)
    g_grand_trains, g_grand_tests, g_grand_labels = [], [], []
    g_grand_train_reshaped, g_grand_test_reshaped, g_grand_label_reshaped = [], [], []

    for num in range(len(x_trains)):
        grand_trains, grand_tests, grand_labels = [], [], []
        for grand in range(len(x_trains[num])):
            sub_x_trains = _create_sequences(x_trains[num][grand], nest_length, step)
            train_sequences, freq = torch_fft_transform(sub_x_trains)
            grand_trains.append(train_sequences) 
        grand_train = np.array(grand_trains) # (grand_train: (301, 76, 25/2, 1) )
        grand_train_reshaped = grand_train.reshape(grand_train.shape[0], grand_train.shape[1]*grand_train.shape[2], grand_train.shape[3])
        
        for grand in range(len(x_tests[num])):
            sub_x_tests = _create_sequences(x_tests[num][grand], nest_length, step)
            test_sequences, freq = torch_fft_transform(sub_x_tests)
            grand_tests.append(test_sequences)
            
            sub_y_tests = _create_sequences(y_tests[num][grand], nest_length, step)
            grand_labels.append(sub_y_tests)
            
        grand_label = np.array(grand_labels)
        grand_test = np.array(grand_tests)
        grand_test_reshaped = grand_test.reshape(grand_test.shape[0], grand_test.shape[1]*grand_test.shape[2], grand_test.shape[3])
        grand_label_reshaped = grand_label.reshape(grand_label.shape[0], grand_label.shape[1]*grand_label.shape[2], grand_label.shape[3])

        g_grand_trains.append(grand_train)
        g_grand_tests.append(grand_test)
        g_grand_labels.append(grand_label)

        g_grand_train_reshaped.append(grand_train_reshaped)
        g_grand_test_reshaped.append(grand_test_reshaped)
        g_grand_label_reshaped.append(grand_label_reshaped)
    return {'grand_train': g_grand_trains, 'grand_test': g_grand_tests, 'grand_label': g_grand_labels,
            'grand_train_reshaped': g_grand_train_reshaped, 'grand_test_reshaped': g_grand_test_reshaped, 'grand_label_reshaped': g_grand_label_reshaped}


def generate_frequency_grandwindow(x_trains, x_tests, y_tests, nest_length, step):
    grand_trains, grand_tests, grand_labels = [], [], []
    # sub_evaluation_arrays = []
    # x_trains: (301, 100, 1)
    if x_trains.size > 0:
        for grand in range(len(x_trains)):
            sub_x_trains = _create_sequences(x_trains[grand], nest_length, step)
            train_sequences, freq = torch_fft_transform(sub_x_trains)
            grand_trains.append(train_sequences)
        grand_train = np.array(grand_trains) # (grand_train: (301, 76, 25/2, 1) )
        grand_train_reshaped = grand_train.reshape(grand_train.shape[0], grand_train.shape[1]*grand_train.shape[2], grand_train.shape[3])
    else:
        # If x_trains is empty, create empty arrays with appropriate dimension for reshaping later
        grand_train = np.array([])
        # The shape must be (0, 0, num_features) to be compatible.
        # We can get num_features from the test set.
        num_features = x_tests.shape[-1] if x_tests.ndim > 1 else 1
        grand_train_reshaped = np.empty((0, 0, num_features))

    # Early exit handling for empty test data (common when client_load_test=False)
    if x_tests.size == 0 or y_tests.size == 0:
        # Reuse inferred feature dimension if possible
        if x_trains.size > 0:
            feat_dim = x_trains.shape[-1]
        else:
            # Fallback to 1 if completely empty; callers typically won't use these
            feat_dim = 1
        grand_test = np.empty((0, 0, 0, feat_dim))  # maintain 4D for consistency
        grand_label = np.empty((0, 0, 0, feat_dim))
        grand_test_reshaped = np.empty((0, 0, feat_dim))
        grand_label_reshaped = np.empty((0, 0, feat_dim))
        return {'grand_train': grand_train, 'grand_test': grand_test, 'grand_label': grand_label,
                'grand_train_reshaped': grand_train_reshaped, 'grand_test_reshaped': grand_test_reshaped, 'grand_label_reshaped': grand_label_reshaped}

    for grand in range(len(x_tests)):
        sub_x_tests = _create_sequences(x_tests[grand], nest_length, step)
        test_sequences, freq = torch_fft_transform(sub_x_tests)
        grand_tests.append(test_sequences)
        
        sub_y_tests = _create_sequences(y_tests[grand], nest_length, step)
        grand_labels.append(sub_y_tests)
        
    grand_label = np.array(grand_labels)
    grand_test = np.array(grand_tests)
    grand_test_reshaped = grand_test.reshape(grand_test.shape[0], grand_test.shape[1]*grand_test.shape[2], grand_test.shape[3])
    grand_label_reshaped = grand_label.reshape(grand_label.shape[0], grand_label.shape[1]*grand_label.shape[2], grand_label.shape[3])
    return {'grand_train': grand_train, 'grand_test': grand_test, 'grand_label': grand_label,
            'grand_train_reshaped': grand_train_reshaped, 'grand_test_reshaped': grand_test_reshaped, 'grand_label_reshaped': grand_label_reshaped}

def load_ECG(seq_length=100, stride=1):
    # sequence length:
    # stride: 
    # source: https://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip
    data_path = './datasets/ECG/labeled/'
    datasets = sorted([f for f in os.listdir(f'{data_path}/train') if os.path.isfile(os.path.join(f'{data_path}/train', f))])

    x_train, x_test, y_test = [], [], []
    y_segment_test = []
    train_seq, label_seq, test_seq = [], [], []
    train_dfs = []
    
    for data in datasets:
        train_df = np.array(pd.read_pickle(f'{data_path}/train/{data}'))
        train_df = train_df[:, [0, 1]].astype(float)
        
        test_df = np.array(pd.read_pickle(f'{data_path}/test/{data}'))
        labels = test_df[:, -1].astype(int)
        test_df = test_df[:, [0, 1]].astype(float)
        y_tests = labels.reshape(-1, 1)

        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride))
            x_test.append(_create_sequences(test_df, seq_length, stride))
            y_test.append(_create_sequences(y_tests, seq_length, stride))
        else:
            x_train.append(train_df)
            x_test.append(test_df)
            y_test.append(y_tests)
        
        label_seq.append(labels)
        test_seq.append(test_df)
        train_seq.append(train_df) 
        y_segment_test.append(_count_anomaly_segments(label_seq)[1])
        train_dfs.append(train_df)
            
        
    return {'x_train': x_train, 'x_test': x_test, 'y_test': y_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_segment_test': y_segment_test, 'train_dfs':train_dfs}


def load_tods(form, seq_length=50, stride=1):
    # source: https://github.com/datamllab/tods/tree/benchmark
    data_path = 'datasets/NeurIPSTS'
    if form == 'trend':
        train_df = np.array(pd.read_csv(f'{data_path}/trend_train.csv'))
        train_df = train_df[:, 0].astype(float)
        test_df = np.array(pd.read_csv(f'{data_path}/trend.csv'))

    elif form == 'shaplet':
        train_df = np.array(pd.read_csv(f'{data_path}/train.csv'))
        train_df = train_df[:, 0].astype(float)
        test_df = np.array(pd.read_csv(f'{data_path}/shaplet.csv'))

    elif form == 'seasonal':
        train_df = np.array(pd.read_csv(f'{data_path}/train.csv'))
        train_df = train_df[:, 0].astype(float)
        test_df = np.array(pd.read_csv(f'{data_path}/seasonal.csv'))

    elif form == 'point':
        train_df = np.array(pd.read_csv(f'{data_path}/train.csv'))
        train_df = train_df[:, 0].astype(float)
        test_df = np.array(pd.read_csv(f'{data_path}/point.csv'))

    elif form == 'context':
        train_df = np.array(pd.read_csv(f'{data_path}/train.csv'))
        train_df = train_df[:, 0].astype(float)
        test_df = np.array(pd.read_csv(f'{data_path}/context.csv'))

    else:
        print("Empty Dataset")

    labels = test_df[:, -1].astype(int)
    test_df = test_df[:, 0].astype(float)

    # Univariate reshaping
    train_df = train_df.reshape(-1, 1)
    test_df = test_df.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    if seq_length > 0:
        x_train = _create_sequences(train_df, seq_length, stride)
        x_test =_create_sequences(test_df, seq_length, stride)   
        y_test = np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1)
    else:
        x_train = train_df
        x_test = test_df
        y_test = labels

    label_seq = labels
    test_seq = test_df

    return {'x_train': x_train, 'x_test': x_test, 'y_test': y_test, 'label_seq': label_seq, 'test_seq': test_seq}


def load_PSM(seq_length=100, stride=1, historical=False):
    # seq. length: 60:30 (ref. )
    # source: https://github.com/eBay/RANSynCoders
    # interval:  equally-spaced 1 minute apart
    
    # --- Path Correction ---
    # The 'datasets' directory is inside the 'dualflsim' package, relative to this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory from 'utils' to the 'dualflsim' package root
    package_root = os.path.dirname(script_dir)
    path = os.path.join(package_root, 'datasets', 'PSM')

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    train_seq, label_seq, test_seq = [], [], []
    
    train_df = pd.read_csv(f'{path}/train.csv').iloc[:, 1:].fillna(method="ffill").values
    test_df = pd.read_csv(f'{path}/test.csv').iloc[:, 1:].fillna(method="ffill").values
    labels = pd.read_csv(f'{path}/test_label.csv')['label'].values.astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    print(f"[DEBUG in data_loader.py] Shape of test_df after loading: {test_df.shape}")

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)
    
    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)
    
    valid_labels = labels[:valid_idx]
    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])
    y_valid.append(valid_labels)

        
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


# ===================== Added NASA / SMD loaders (FedKO-style adaptation) ===================== #

def _concat_files_in_dir(dir_path: str, file_ext: str = None, loader='csv') -> np.ndarray:
    """Concatenate all files in a directory vertically.

    Args:
        dir_path: directory containing per-channel / per-machine files
        file_ext: optional extension filter; if None, use all files
        loader: 'csv' for numeric CSV/txt, 'npy' for numpy arrays
    Returns:
        concatenated 2D numpy array (time, features)
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(
            f"Required data directory missing: {dir_path}\n"
            "Ensure you have placed the dataset files correctly. Refer to README dataset section."
        )
    files = sorted([f for f in os.listdir(dir_path) if (file_ext is None or f.endswith(file_ext))])
    if not files:
        raise RuntimeError(
            f"No data files found in {dir_path}.\n"
            "Expected per-channel or per-machine files. Check extraction or dataset download."
        )
    arrays = []
    for f in files:
        fp = os.path.join(dir_path, f)
        if loader == 'csv':
            # Support both comma and whitespace separated numeric txt
            try:
                df = pd.read_csv(fp, header=None)
            except Exception:
                df = pd.read_csv(fp, header=None, delim_whitespace=True)
            arr = df.values.astype(np.float32)
        elif loader == 'npy':
            arr = np.load(fp).astype(np.float32)
        else:
            raise ValueError(f"Unsupported loader type: {loader}")
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def load_SMD(seq_length=100, stride=1, historical=False) -> Dict[str, Any]:
    """Replicates FedKO style: concatenate all SMD train/test files then window like PSM loader.

    Dataset expected at datasets/SMD with subfolders train/, test/, test_label/
    Each file is numeric CSV/txt; labels are 0/1 per line in matching order.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    base = os.path.join(package_root, 'datasets', 'SMD')
    train_dir = os.path.join(base, 'train')
    test_dir = os.path.join(base, 'test')
    label_dir = os.path.join(base, 'test_label')

    try:
        train_df = _concat_files_in_dir(train_dir, loader='csv')
        test_df = _concat_files_in_dir(test_dir, loader='csv')
    except (FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed loading SMD dataset: {e}\n"
            f"Expected structure: {base}/train/*.txt, {base}/test/*.txt, {base}/test_label/*.txt"
        )

    # Load labels: each file contains per-line 0/1; we concatenate in same sorted order
    label_files = sorted(os.listdir(label_dir))
    labels_list: List[np.ndarray] = []
    for lf in label_files:
        with open(os.path.join(label_dir, lf), 'r') as f:
            lines = [l.strip().split(',')[0] for l in f if l.strip()]
            vals = np.array([float(x) for x in lines], dtype=np.float32)
            labels_list.append(vals)
    labels = np.concatenate(labels_list, axis=0).astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    train_seq, test_seq, label_seq = [], [], []

    # For SMD we do not have a separate validation; mimic PSM splitting portion of test for validation (30%)
    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]
    valid_labels = labels[:valid_idx]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)

    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)
    y_valid.append(valid_labels)
    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def _load_smap_msl_combined(seq_length=100, stride=1, historical=False, dataset_name='SMAP') -> Dict[str, Any]:
    """Load SMAP or MSL from the combined NASA format (SMAP+MSL) replicating FedKO stacking.

    We have combined folder structure with train/test .npy per channel and a CSV describing anomalies.
    We'll produce labels by building a binary vector for each channel and concatenating channels of the selected spacecraft.
    NOTE: This assumes variable length series across channels; we'll concatenate sequentially (stack time) like FedKO.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    base = os.path.join(package_root, 'datasets', 'SMAP+MSL', 'data', 'data')
    train_dir = os.path.join(base, 'train')
    test_dir = os.path.join(base, 'test')
    anomalies_csv = os.path.join(package_root, 'datasets', 'SMAP+MSL', 'labeled_anomalies.csv')

    # Parse anomalies metadata
    if not os.path.exists(anomalies_csv):
        raise FileNotFoundError(
            f"Missing anomaly metadata CSV: {anomalies_csv}\n"
            "The combined SMAP+MSL dataset requires 'labeled_anomalies.csv'."
        )
    meta = pd.read_csv(anomalies_csv)
    # Filter spacecraft rows
    filtered = meta[meta['spacecraft'] == dataset_name]
    channel_to_spans = {}
    for _, row in filtered.iterrows():
        chan = row['chan_id']
        spans = eval(row['anomaly_sequences']) if isinstance(row['anomaly_sequences'], str) else []
        channel_to_spans[chan] = spans
    if not channel_to_spans:
        raise RuntimeError(
            f"No channels found for spacecraft={dataset_name} in {anomalies_csv}.\n"
            "Check that 'spacecraft' column contains the correct labels (SMAP/MSL)."
        )

    def build_channel_arrays(directory: str) -> Dict[str, np.ndarray]:
        arrays = {}
        for f in sorted(os.listdir(directory)):
            if f.endswith('.npy'):
                chan = f.replace('.npy', '')
                if chan in channel_to_spans:  # only include selected spacecraft channels
                    arrays[chan] = np.load(os.path.join(directory, f)).astype(np.float32)
        return arrays

    train_arrays = build_channel_arrays(train_dir)
    test_arrays = build_channel_arrays(test_dir)
    # Build labels per channel
    label_arrays = {}
    for chan, arr in test_arrays.items():
        length = arr.shape[0]
        labels = np.zeros(length, dtype=int)
        for s, e in channel_to_spans.get(chan, []):
            e = min(e, length-1)
            labels[s:e+1] = 1
        label_arrays[chan] = labels

    # Concatenate channels sequentially along time axis (FedKO style stacking) maintaining feature dimension
    train_df = np.concatenate(list(train_arrays.values()), axis=0) if train_arrays else np.empty((0,))
    test_df = np.concatenate(list(test_arrays.values()), axis=0) if test_arrays else np.empty((0,))
    labels = np.concatenate(list(label_arrays.values()), axis=0) if label_arrays else np.empty((0,), dtype=int)
    if train_df.ndim == 1:
        train_df = train_df.reshape(-1, 1)
    if test_df.ndim == 1:
        test_df = test_df.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    train_seq, test_seq, label_seq = [], [], []

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]
    valid_labels = labels[:valid_idx]

    if seq_length > 0 and test_df.shape[0] >= seq_length:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)

    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)
    y_valid.append(valid_labels)
    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])

    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq': test_seq, 'train_seq': train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_SMAP(seq_length=100, stride=1, historical=False):
    # Try standalone folder first; if empty fallback to combined
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    smap_dir = os.path.join(package_root, 'datasets', 'SMAP')
    if os.path.isdir(smap_dir) and os.listdir(smap_dir):
        # Placeholder: currently directory empty in workspace; fallback path below will execute.
        pass
    return _load_smap_msl_combined(seq_length, stride, historical, dataset_name='SMAP')


def load_MSL(seq_length=100, stride=1, historical=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    msl_dir = os.path.join(package_root, 'datasets', 'MSL')
    if os.path.isdir(msl_dir) and os.listdir(msl_dir):
        # Placeholder if raw separated MSL format appears later.
        pass
    return _load_smap_msl_combined(seq_length, stride, historical, dataset_name='MSL')


def load_dataset_by_name(name: str, seq_length=100, stride=1, historical=False) -> Dict[str, Any]:
    name_upper = name.upper()
    if name_upper == 'PSM':
        return load_PSM(seq_length, stride, historical)
    if name_upper == 'SMD':
        return load_SMD(seq_length, stride, historical)
    if name_upper == 'SMAP':
        return load_SMAP(seq_length, stride, historical)
    if name_upper == 'MSL':
        return load_MSL(seq_length, stride, historical)
    raise ValueError(f"Unsupported dataset name: {name}. Available: PSM, SMD, SMAP, MSL")


def load_SWaT(seq_length=100, stride=1, historical=False):
    # seq. length: 600:300 (i.e., 10 minutes)
    # source: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
    # interval: 1 second
    
    path = f'./datasets/SWaT/downsampled'
    
    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    label_seq, test_seq, train_seq = [], [], []
    
    train_df = np.load(f'{path}/train.npy')
    test_df = np.load(f'{path}/test.npy')
    labels = np.load(f'{path}/test_label.npy')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    valid_idx = int(test_df.shape[0] * 0.3)
    valid_df = test_df[:valid_idx]

    if seq_length > 0:
        x_train.append(_create_sequences(train_df, seq_length, stride, historical))
        x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
        x_test.append(_create_sequences(test_df, seq_length, stride, historical))
        y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
    else:
        x_train.append(train_df)
        x_valid.append(valid_df)
        x_test.append(test_df)
        y_test.append(labels)
    
    label_seq.append(labels)
    test_seq.append(test_df)
    train_seq.append(train_df)    
    
    valid_labels = labels[:valid_idx]
    y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
    y_segment_test.append(_count_anomaly_segments(label_seq)[1])
    y_valid.append(valid_labels)

        
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq':test_seq, 'train_seq':train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}  


def load_ASD(seq_length=100, stride=1, historical=False):
    # seq. length: 100:50 (ref. OmniAnomaly)
    # source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
    # interval:  equally-spaced 5 minutes apart
    path = f'./datasets/ASD'
    f_names = sorted([f for f in os.listdir(f'{path}/train') if os.path.isfile(os.path.join(f'{path}/train', f))])

    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    label_seq, test_seq, train_seq = [], [], []
    
    for f_name in f_names:
        train_df = pd.read_pickle(f'{path}/train/{f_name}')
        test_df = pd.read_pickle(f'{path}/test/{f_name}')
        labels = pd.read_pickle(f'{path}/test_label/{f_name}').astype(int)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df = test_df[:valid_idx]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
            y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)
            y_test.append(labels)

        label_seq.append(labels)
        test_seq.append(test_df)
        train_seq.append(train_df)

        valid_labels = labels[:valid_idx]
        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(label_seq)[1])
        y_valid.append(valid_labels)


    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq':test_seq, 'train_seq':train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}

def load_CompanyA(seq_length=144, stride=1, historical=False):
    # interval: 10 min
    # seq. length: 144:1 (i.e., 1 day)
    # source: Company A
    data_path = './datasets/CompanyA'
    f_names = sorted([f for f in os.listdir(f'{data_path}/train') if os.path.isfile(os.path.join(f'{data_path}/train', f))])
    
    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    label_seq, test_seq, train_seq = [], [], []
    
    for f_name in f_names:
        train_df = pd.read_csv(f'{data_path}/train/{f_name}')
        test_df = pd.read_csv(f'{data_path}/test/{f_name}')
        
        train_df = np.array(train_df)
        train_df = train_df[:, 1:-1].astype(float)
        test_df = np.array(test_df)
        labels = test_df[:, -1].astype(int)
        test_df = test_df[:, 1:-1].astype(float)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        valid_idx = int(test_df.shape[0] * 0.3)
        valid_df = test_df[:valid_idx]

        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
            y_test.append(np.expand_dims(_create_sequences(labels, seq_length, stride), axis=-1))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)
            y_test.append(labels)

        label_seq.append(labels)
        test_seq.append(test_df)
        train_seq.append(train_df)

        valid_labels = labels[:valid_idx]
        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(label_seq)[1])
        y_valid.append(valid_labels)


    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq':test_seq, 'train_seq':train_seq,
            'y_valid': y_valid, 'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


def load_UCR(seq_length=128, stride=1, historical=False):
    # source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
    # interval: varies
    # remark: This dataset includes: EPG, NASA (KDD), ECG, PVC, Respiration, EPG, Power Demand, Internal Bleeding etc.
        
    path = './datasets/UCR'
    f_names = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    
    x_train, x_valid, x_test = [], [], []
    y_valid, y_test = [], []
    y_segment_valid, y_segment_test = [], []
    label_seq, test_seq, train_seq = [], [], []
    
    for f_name in tqdm(f_names):
        df = pd.read_csv(f'{path}/{f_name}', header=None, dtype=float).values
        idx = np.array(f_name.split('.')[0].split('_')[-3:]).astype(int)
        train_idx, label_start, label_end = idx[0], idx[1], idx[2]+1
        labels = np.zeros(df.shape[0], dtype=int)
        labels[label_start-100:label_end+100] = 1
        
        train_df = df[:train_idx]
        test_df = df[train_idx:]
        labels = labels[train_idx:]
        label_seq.append(labels)
        labels = labels.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        valid_idx = int(test_df.shape[0] * 0.3)
        # valid_df, test_df = test_df[:valid_idx], test_df[valid_idx:]
        valid_df, test_df = test_df, test_df
        
        if seq_length > 0:
            x_train.append(_create_sequences(train_df, seq_length, stride, historical))
            x_valid.append(_create_sequences(valid_df, seq_length, stride, historical))
            x_test.append(_create_sequences(test_df, seq_length, stride, historical))
            y_test.append(_create_sequences(labels, seq_length, stride, historical))
        else:
            x_train.append(train_df)
            x_valid.append(valid_df)
            x_test.append(test_df)
            y_test.append(labels)
        
        
        test_seq.append(test_df)
        train_seq.append(train_df)

        # valid_labels, test_labels = labels[:valid_idx], labels[valid_idx:]
        valid_labels, test_labels = labels, labels
        
        y_segment_valid.append(_count_anomaly_segments(valid_labels)[1])
        y_segment_test.append(_count_anomaly_segments(test_labels)[1])
        
    
    return {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test,
            'label_seq': label_seq, 'test_seq':test_seq, 'train_seq':train_seq,
            'y_test': y_test,
            'y_segment_valid': y_segment_valid, 'y_segment_test': y_segment_test}


############################### Get Time Window ###############################

def get_loader_segment(batch_size, seq_length, form, step=1, mode='train', dataset='NeurIPSTS'):
    
    if dataset == 'NeurIPSTS':
        # Create sliding window sequences
        data_dict = load_tods(form=form, seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'], data_dict['x_test'], data_dict['y_test']
        label_seq, test_seq = data_dict['label_seq'], data_dict['test_seq']

    elif dataset == 'PSM':
        # Create sliding window sequences
        data_dict = load_PSM(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]
    
    elif dataset == 'SWaT':
        # Create sliding window sequences
        data_dict = load_SWaT(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]

    elif dataset == 'ASD':
        # Create sliding window sequences
        data_dict = load_ASD(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]

    elif dataset == 'CompanyA':
        # Create sliding window sequences
        data_dict = load_CompanyA(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]
    
    elif dataset == 'ECG':
        # Create sliding window sequences
        data_dict = load_ECG(seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]            

    # Indexing
    if mode == 'train':
        dataset = TrainingLoader(x_train)
        shuffle = True
    else:
        dataset = GeneralLoader(x_test, y_test)
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader, label_seq, test_seq



############################### Get Outer Window ###############################

def get_loader_grandwin(batch_size, seq_length, nest_length, form, step=1, dataset='NeurIPSTS', data_loader='load_SWaT'):
    if dataset == 'NeurIPSTS':
        # Create sliding window sequences
        data_dict = load_tods(form=form, seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'], data_dict['x_test'], data_dict['y_test']
        label_seq, test_seq = data_dict['label_seq'], data_dict['test_seq']

        grand_dict = generate_frequency_grandwindow(x_train, x_test, y_test, nest_length, step)
        grand_train, grand_test, grand_label = grand_dict['grand_train'], grand_dict['grand_test'], grand_dict['grand_label']
        grand_train_reshaped, grand_test_reshaped, grand_label_reshaped = grand_dict['grand_train_reshaped'], grand_dict['grand_test_reshaped'], grand_dict['grand_label_reshaped']

    else: 
        # Create sliding window sequences
        data_dict = globals()[f'{data_loader}'](seq_length=seq_length, stride=step)
        x_train, x_test, y_test = data_dict['x_train'][form], data_dict['x_test'][form], data_dict['y_test'][form]
        label_seq, test_seq = data_dict['label_seq'][form], data_dict['test_seq'][form]

        grand_dict = generate_frequency_grandwindow(x_train, x_test, y_test, nest_length, step)
        grand_train, grand_test, grand_label = grand_dict['grand_train'], grand_dict['grand_test'], grand_dict['grand_label']
        grand_train_reshaped, grand_test_reshaped, grand_label_reshaped = grand_dict['grand_train_reshaped'], grand_dict['grand_test_reshaped'], grand_dict['grand_label_reshaped']

    print("======================TRAIN Granding======================")
    train_loader = DataLoader(dataset=TrainingLoader(grand_train_reshaped),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    print("======================TEST Granding======================")
    test_loader = DataLoader(dataset=GeneralLoader(grand_test_reshaped, grand_test_reshaped),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    return train_loader, test_loader, label_seq, test_seq, y_test, grand_label
