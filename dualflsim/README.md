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
    -   **`load_data`**: Loads the 'seasonal' dataset and partitions it among the clients. It also generates the frequency-domain data.
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

## Switch strategies: FedProx vs SCAFFOLD

You can toggle the server strategy in `configs/default.yaml`:

```
strategy:
    type: scaffold   # or 'fedprox'
```

- scaffold: Enables SCAFFOLD-style control variates. The server sends control bytes to clients; the client computes and returns delta_ci for server updates. We force `proximal_mu` to 0 when using SCAFFOLD.
- fedprox: Uses classic FedProx. Set `training.proximal_mu` to a non-zero value if desired (e.g., 0.01). No SCAFFOLD fields are exchanged with clients.

Both modes log per-round aggregated training loss (and server metrics if evaluation is enabled). Post-training array generation works the same in either mode.
