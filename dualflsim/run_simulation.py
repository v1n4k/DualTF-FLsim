import ray
import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.strategy import FedProx
from flwr.common import ndarrays_to_parameters, Metrics, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate
from typing import List, Tuple, Dict, Optional
import os
import gc
import time
import torch

# Import components from your project
from dualflsim.client_app import client_fn
from dualflsim.task import (
    FederatedDualTF,
    get_weights,
    set_weights,
    load_data,
    test,
)
import numpy as np
import io


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 5,
        # FedProx disabled for SCAFFOLD (kept for compatibility)
        "proximal_mu": 0.0,
        # Client learning rate for both time/freq optimizers
        "lr": 3e-5,
    }
    return config


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate F1, precision, and recall."""
    # Calculate weighted average for each metric
    f1_aggregated = sum([m["f1"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    precision_aggregated = sum([m["precision"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    recall_aggregated = sum([m["recall"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    
    return {"f1": f1_aggregated, "precision": precision_aggregated, "recall": recall_aggregated}


class SaveResultsStrategy(FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Clear the file at the beginning of the simulation
        with open("simulation_summary.txt", "w") as f:
            f.write("Federated Learning Simulation Summary\n")
            f.write("="*40 + "\n\n")

    # When using a server-side evaluate_fn, aggregate_evaluate is not called.
    # Override evaluate to always persist results regardless of evaluation mode.
    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        # Delegate to base class (will call evaluate_fn if provided)
        result = super().evaluate(server_round, parameters)
        loss, metrics = result if result is not None else (None, None)
        if loss is not None and metrics is not None:
            with open("simulation_summary.txt", "a") as f:
                f.write(f"Round {server_round}:\n")
                f.write(f"  Loss: {loss}\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value}\n")
                f.write("\n")
        return result

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and save them to a file."""
        # Call the base class method to perform the actual aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Save results to file
        if aggregated_loss is not None and aggregated_metrics is not None:
            with open("simulation_summary.txt", "a") as f:
                f.write(f"Round {server_round}:\n")
                f.write(f"  Loss: {aggregated_loss}\n")
                for metric, value in aggregated_metrics.items():
                    f.write(f"    {metric}: {value}\n")
                f.write("\n")
        
        return aggregated_loss, aggregated_metrics


def main():
    # 1. Initialize Ray explicitly and tell it about available GPUs
    # Disable the disk space warning by setting an environment variable
    # Reserve 1 GPU for server-side evaluation by exposing only 3 to Ray
    ray.init(num_cpus=32, num_gpus=3)
    print("Ray initialized.")
    print(f"Ray resources: {ray.available_resources()}")

    # 2. Define the client resources directly in code.
    client_resources = {"num_cpus": 2, "num_gpus": 1}

    # 3. Define the strategy
    # Define model configurations based on the original project's defaults
    # For PSM dataset, the number of features (channels) is 25
    time_model_args = {'win_size': 100, 'enc_in': 25, 'c_out': 25, 'e_layers': 3}
    # Frequency grand-window keeps the feature dimension (25) as channels
    # Sequence length becomes (seq_len - nest_len + 1) * floor(nest_len/2)
    freq_model_args = {
        'win_size': (100 - 25 + 1) * (25 // 2),
        'enc_in': 25,
        'c_out': 25,
        'e_layers': 3,
    }

    # Initialize model parameters
    temp_model = FederatedDualTF(time_model_args, freq_model_args)
    ndarrays = get_weights(temp_model)
    parameters = ndarrays_to_parameters(ndarrays)
    # Parameter-only templates for SCAFFOLD control variates (exclude buffers)
    param_templates = [p.detach().cpu().numpy() for p in temp_model.parameters()]

    # ---------- SCAFFOLD helpers ----------
    def pack_ndarrays_to_bytes(arrs: List[np.ndarray]) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(arrs)})
        return buf.getvalue()

    def unpack_bytes_to_ndarrays(data: bytes) -> List[np.ndarray]:
        buf = io.BytesIO(data)
        with np.load(buf) as npz:
            return [npz[k] for k in sorted(npz.files, key=lambda x: int(x[1:]))]

    class ScaffoldStrategy(SaveResultsStrategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Global control variate c and per-client ci
            self.global_c = [np.zeros_like(a) for a in param_templates]
            self.client_ci: Dict[str, List[np.ndarray]] = {}
            # Track last known global parameters to allow graceful fallback
            self.latest_parameters: Optional[fl.common.Parameters] = None

        def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager):
            fit_cfg = super().configure_fit(server_round, parameters, client_manager)
            # Remember the parameters used for this round in case aggregation must fallback
            self.latest_parameters = parameters
            # Attach SCAFFOLD controls to each client's config
            for cp, fin in fit_cfg:
                cid = getattr(cp, "cid", "unknown")
                if cid not in self.client_ci:
                    self.client_ci[cid] = [np.zeros_like(a) for a in self.global_c]
                fin.config["c_bytes"] = pack_ndarrays_to_bytes(self.global_c)
                fin.config["ci_bytes"] = pack_ndarrays_to_bytes(self.client_ci[cid])
                # Ensure FedProx is disabled
                fin.config["proximal_mu"] = 0.0
            return fit_cfg

        def aggregate_fit(self, server_round: int, results, failures):
            # Filter out clients with non-finite parameters or invalid delta_ci
            filtered_results = []
            deltas: List[List[np.ndarray]] = []
            cids: List[str] = []
            filtered_by_params = 0
            filtered_by_delta = 0
            for cp, fit_res in results:
                try:
                    arrs = parameters_to_ndarrays(fit_res.parameters)
                    if any((not np.all(np.isfinite(a))) for a in arrs):
                        filtered_by_params += 1
                        continue
                    filtered_results.append((cp, fit_res))
                    metrics = getattr(fit_res, "metrics", {}) or {}
                    delta_bytes = metrics.get("delta_ci", None)
                    if isinstance(delta_bytes, (bytes, bytearray)):
                        delta = unpack_bytes_to_ndarrays(delta_bytes)
                        if any((not np.all(np.isfinite(d))) for d in delta):
                            filtered_by_delta += 1
                            continue
                        deltas.append(delta)
                        cids.append(getattr(cp, "cid", "unknown"))
                except Exception:
                    continue

            print(f"[Server] aggregate_fit: received={len(results)}, used={len(filtered_results)}, filtered_params={filtered_by_params}, filtered_delta={filtered_by_delta}, failures={len(failures)}")

            if not filtered_results:
                print(f"[Server] Round {server_round}: no valid training updates; keeping previous global weights.")
                # Fallback to previously broadcast parameters
                return (self.latest_parameters, {})

            # FedAvg over filtered results
            weights_results = [
                (parameters_to_ndarrays(fr.parameters), fr.num_examples)
                for _, fr in filtered_results
            ]
            new_weights = aggregate(weights_results)
            aggregated_parameters = ndarrays_to_parameters(new_weights)
            aggregated_metrics: Dict[str, fl.common.Scalar] = {}

            # Update control variates (damped and finite)
            if deltas:
                sum_delta = [np.zeros_like(a) for a in self.global_c]
                alpha = 0.1  # damping factor
                for cid, delta in zip(cids, deltas):
                    if cid not in self.client_ci:
                        self.client_ci[cid] = [np.zeros_like(a) for a in self.global_c]
                    for i in range(len(self.global_c)):
                        d = alpha * delta[i]
                        self.client_ci[cid][i] = self.client_ci[cid][i] + d
                        sum_delta[i] = sum_delta[i] + d
                m = float(len(deltas))
                for i in range(len(self.global_c)):
                    self.global_c[i] = self.global_c[i] + (sum_delta[i] / m)

            # Stash latest successful aggregation
            self.latest_parameters = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

    # Server-side evaluation to centralize threshold sweep
    def server_evaluate_fn(
        server_round: int,
        params: List,  # already a list of ndarrays provided by Flower
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        # Optionally skip round 0 evaluation to avoid logging the untrained baseline
        if server_round == 0:
            print("[Server] Skipping evaluation for round 0")
            return None
        import os
        # Give the server more CPU threads for evaluation
        os.environ.setdefault("OMP_NUM_THREADS", "16")
        os.environ.setdefault("MKL_NUM_THREADS", "16")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "16")
        try:
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
        except Exception:
            pass

        # Prefer a GPU with enough free memory; fall back to CPU if none
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
        # Use the highest-index GPU (kept free by Ray) for server evaluation
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            device_index = torch.cuda.device_count() - 1
            device = torch.device(f"cuda:{device_index}")
        else:
            device = torch.device("cpu")

        print(f"[Server] Evaluating round {server_round} on central test set (device={device})...")
        # Rebuild model and load parameters
        net = FederatedDualTF(time_model_args, freq_model_args)
        # params is already a list of ndarrays; set directly
        # If params contain non-finite values, skip evaluation
        if any((not np.all(np.isfinite(a))) for a in params):
            print("[Server] Skipping evaluation due to non-finite global parameters")
            return None
        set_weights(net, params)

        # Load full test loaders on the server and evaluate sequentially on the reserved GPU
        _, testloader_time, _, testloader_freq = load_data(
            partition_id=0,
            num_partitions=1,
            time_batch_size=64,
            freq_batch_size=32,
            dirichlet_alpha=2.0,
        )
        # Evaluate using existing test() which handles both models and threshold sweep
        loss, metrics = test(net, testloader_time, testloader_freq, device)
        # Cleanup: free VRAM on the reserved eval GPU
        try:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        print(f"[Server] Round {server_round} evaluation done: {metrics}")
        return float(loss), metrics

    # Define strategy (standard partial participation): only selected clients get the global
    strategy = ScaffoldStrategy(
        fraction_fit=0.25,   # ~6 clients train per round
        fraction_evaluate=0.0,  # Use server-side evaluation only
        min_available_clients=24,
        min_fit_clients=6,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        proximal_mu=0.1,
        evaluate_fn=server_evaluate_fn,
    )

    # 4. Start the simulation using fl.simulation.start_simulation
    print("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=24,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=30), # More rounds to see convergence
        strategy=strategy,
    )

    print("Simulation finished.")


if __name__ == "__main__":
    main()
