# Federated Dual-Transformer for Time-Series Anomaly Detection

This project implements a federated learning version of the Dual-Transformer Anomaly Detection (DualTF) model using the Flower framework with a Ray simulation backend. It allows for the distributed training of the time-series and frequency-domain transformers on decentralized data while leveraging multiple GPUs for simulation.

## Project Overview

The core of this project is the federated implementation of the DualTF model, which consists of two main components:
1.  **AnomalyTransformer**: A time-domain model that learns temporal patterns.
2.  **FrequencyTransformer**: A frequency-domain model that learns periodic patterns.

The training is performed in a self-supervised manner, where each model learns to reconstruct normal time-series data. Anomaly scores are then derived from the reconstruction error and the model's internal association discrepancy.

This federated simulation demonstrates how to:
-   Partition a time-series dataset among multiple clients.
-   Train a complex, two-part PyTorch model in a federated setting.
-   Utilize multiple GPUs for simulating clients in parallel using Ray.
-   Aggregate performance metrics (F1-score, Precision, Recall) in a federated evaluation environment.

## Getting Started

### Prerequisites

-   Python 3.8+
-   An environment manager like Conda or venv.
-   Access to one or more NVIDIA GPUs with CUDA installed.

### 1. Setup the Environment

First, create and activate a new Conda environment:

```bash
conda create -n dualtf-env python=3.8.12
conda activate dualtf-env
```

### 2. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 2.1 Configuration (single source of truth)

All run parameters are controlled via a YAML config at `configs/default.yaml`. You can provide a custom file via:

- Environment: `export DUALFLSIM_CONFIG=/path/to/your.yaml`
- CLI: `python run_simulation.py /path/to/your.yaml`

Important consistency rules:
- `model.seq_len` must equal `data.seq_length` (window length for TimeTransformer and data windows)
- For PSM, keep `enc_in` and `c_out` at 25 for both time and freq models
- `model.freq.nest_length` is used for frequency grand-windows, centralized tests, and the frequency model; it’s passed everywhere from config
- Client counts: `simulation.min_available_clients` and `simulation.min_fit_clients` must be ≤ `simulation.num_clients`
 - Dataset-feature alignment (must manually adjust when switching datasets):
     - PSM / SMAP: `model.time.enc_in = model.time.c_out = model.freq.enc_in = model.freq.c_out = 25`
     - SMD: use 38 instead of 25
     - MSL: use 55 instead of 25
     If you forget to update these when changing `data.dataset`, a shape mismatch will occur at model forward time.

You can change any parameter in the YAML without editing code.

### 3. Prepare the Datasets

This implementation uses the datasets from the original DualTF project. Ensure the `datasets` directory is present in the project root and contains the necessary data files (e.g., from the NeurIPSTS collection).

The project structure should look like this:

```
.
├── dualflsim/
│   ├── client_app.py
│   ├── model/
│   ├── task.py
│   └── utils/
├── datasets/
│   └── ... (dataset files)
├── run_simulation.py
└── README.md
```

### 4. Run the Simulation

To start the federated learning simulation, run the main script:

```bash
python run_simulation.py
```

This will initialize a Ray cluster, create 10 virtual clients (distributed across 4 GPUs), and run the federated training and evaluation process for 3 rounds.

Optional with a custom config:

```bash
python run_simulation.py /path/to/your.yaml
```

## How It Works

1.  **`run_simulation.py`**: This is the main entry point. It initializes Ray, defines the Flower strategy (`FedAvg`), and starts the simulation with 10 clients.
2.  **`dualflsim/client_app.py`**: Defines the Flower `ClientApp`. It handles loading the model and data for each client and calls the `train` and `test` functions.
3.  **`dualflsim/task.py`**: Contains the core machine learning logic:
    -   **`load_data`**: Loads the configured dataset (see Dataset Selection section) and partitions it among the clients using a Dirichlet quantity split. It also generates the frequency-domain data via grand-window FFT.
    -   **`train`**: Implements the self-supervised training logic for both the time and frequency models.
    -   **`test`**: Implements the evaluation logic, which calculates anomaly scores, finds the best F1-score via threshold simulation, and returns the final metrics.
4.  **`dualflsim/model/`**: Contains the PyTorch definitions for the `AnomalyTransformer` and `FrequencyTransformer`.

## Interpreting the Results

At the end of the simulation, a `[SUMMARY]` section will be printed with the final aggregated loss and metrics.

```
[SUMMARY]
INFO :      Run finished 3 round(s) in ...s
INFO :      History (loss, distributed):
INFO :              round 1: 0.577...
INFO :              round 2: ...
INFO :              round 3: ...
INFO :      History (metrics, distributed, evaluate):
INFO :      {'f1': [(1, 0.7627), (2, ...), (3, ...)],
INFO :       'precision': [(1, 0.9692), (2, ...), (3, ...)],
INFO :       'recall': [(1, 0.6287), (2, ...), (3, ...)]}
```

-   **F1-Score**: The primary metric for evaluating the model's balance between precision and recall.
-   **Precision**: Indicates the reliability of the model's anomaly predictions.
-   **Recall**: Indicates the model's ability to find all true anomalies.

## Experiment tracking (optional)

We support Weights & Biases tracking behind a config flag. To enable:

1. Log in once: run `wandb login` (or set the `WANDB_API_KEY` env var).
2. In `configs/default.yaml`, set:

```
wandb:
    enabled: true
    project: DualTF-FLSim
    entity: your_wandb_entity   # optional
    run_name: my-experiment     # optional
    tags: [fl, dualtf, psm]
```

What gets logged:
- Per-round aggregated training loss and clients used
- Optional server-side evaluation metrics if `evaluation.enabled: true`
- Final saved time/freq arrays are uploaded as artifacts when generated

Tracking is a no-op when `wandb.enabled` is false or the package is not installed.

## Dataset Selection and Loader Behavior

Supported datasets (configure via `data.dataset` in `configs/default.yaml`):

| Name  | Source / Format | Loader Behavior | Notes |
|-------|-----------------|-----------------|-------|
| PSM   | CSV (train/test/test_label) | MinMax scale train, transform test, 30% validation split from test, sliding windows | 25 features |
| SMD   | Multiple machine `*.txt` files (train/test/test_label) | Concatenate all files (FedKO style), single scaler, 30% validation from test, sliding windows | 38 features (per line) |
| SMAP  | NASA SMAP (loaded from combined `SMAP+MSL`) | Per-channel `.npy` sequences + anomaly spans. Channels filtered to SMAP, labels synthesized, sequences concatenated vertically (time) then windowed | Feature dim inferred (e.g. 25) |
| MSL   | NASA MSL (from combined `SMAP+MSL`) | Same as SMAP but filtering `spacecraft == MSL` | Feature dim inferred (e.g. 25) |

Why “stacking”? FedKO concatenates per-file sequences back-to-back to form a single long multivariate time series before window generation. We mirror that for SMD and NASA (SMAP/MSL) so a single scaler and a uniform partitioning strategy are applied. This preserves rough global distribution statistics but loses explicit channel boundaries (acceptable for self-supervised reconstruction, but you can modify to keep channel segmentation if needed).

Switching datasets:

```yaml
data:
    dataset: SMD   # or PSM / SMAP / MSL
    seq_length: 75
    step: 1
```

The frequency model's `nest_length` remains independent (controls inner FFT window) but should be ≤ `seq_length`.

Post-training array generation (`post_training.generate_arrays: true`) will write time/freq arrays tagged with the chosen dataset name.

If standalone `datasets/SMAP/` or `datasets/MSL/` folders are later populated, loaders can be extended; currently empty folders trigger fallback to the combined `SMAP+MSL` directory.

### Validation Strategy
For datasets without an explicit validation split (SMD, SMAP, MSL), we reserve the first 30% of the scaled test set as a pseudo-validation window for consistency with PSM handling. Adjust this easily inside the respective loader if you prefer a different ratio.

### Extending
Add a new dataset by implementing `load_<NAME>` in `utils/data_loader.py` returning the same dictionary keys and registering it in `load_dataset_by_name`.

## Switch strategies: FedProx vs SCAFFOLD

You can toggle the server strategy in `configs/default.yaml`:

```
strategy:
    type: scaffold   # or 'fedprox'
```

- scaffold: Enables SCAFFOLD-style control variates. The server sends control bytes to clients; the client computes and returns delta_ci for server updates. We force `proximal_mu` to 0 when using SCAFFOLD.
- fedprox: Uses classic FedProx. Set `training.proximal_mu` to a non-zero value if desired (e.g., 0.01). No SCAFFOLD fields are exchanged with clients.

Both modes log per-round aggregated training loss (and server metrics if evaluation is enabled). Post-training array generation works the same in either mode.
