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
    load_centralized_test_data,
    test,
)
import numpy as np
import io

# Import array generation functionality
from utils.array_generator import generate_evaluation_arrays
from utils.config import load_config
from utils.wandb_utils import maybe_init_wandb, log_metrics, log_artifact, finish as wandb_finish


from typing import Optional

def _get_cfg_path_from_env_or_arg() -> Optional[str]:
    import os
    import sys
    # Allow CLI: python run_simulation.py /path/to/config.yaml
    if len(sys.argv) >= 2 and sys.argv[1].endswith(('.yml', '.yaml')):
        return sys.argv[1]
    return os.environ.get("DUALFLSIM_CONFIG")


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return training configuration dict for each round."""
    return {
        "server_round": server_round,
        "local_epochs": TRAINING_CFG["local_epochs"],
        "proximal_mu": TRAINING_CFG["proximal_mu"],
        "lr": float(TRAINING_CFG["lr"]),
    }


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
        # Track latest aggregated parameters for post-training array generation
        self.latest_parameters: Optional[fl.common.Parameters] = None

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        # Let FedProx perform the standard FedAvg + proximal updates
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        # Record latest parameters for downstream post-training generation
        try:
            if aggregated_parameters is not None:
                self.latest_parameters = aggregated_parameters
        except Exception:
            pass
        # Aggregate client-side train_loss if provided and log via wandb
        try:
            total_examples = 0
            sum_loss = 0.0
            for _, fr in results:
                m = getattr(fr, "metrics", {}) or {}
                if "train_loss" in m and isinstance(m["train_loss"], (int, float)):
                    n = int(getattr(fr, "num_examples", 0))
                    total_examples += n
                    sum_loss += float(m["train_loss"]) * n
            if total_examples > 0:
                avg_loss = sum_loss / float(total_examples)
                if isinstance(aggregated_metrics, dict):
                    aggregated_metrics["train_loss"] = float(avg_loss)
                log_metrics({"train/avg_loss": float(avg_loss), "train/clients_used": float(len(results))}, step=server_round)
        except Exception:
            pass
        return aggregated_parameters, aggregated_metrics


def main():
    # 1. Optionally initialize Ray. We'll use all 4 GPUs for training clients
    # to maximize parallelism, then use 1 GPU for post-training inference.
    # Load configuration (YAML or defaults)
    cfg = load_config(_get_cfg_path_from_env_or_arg())
    # Optional experiment tracking
    maybe_init_wandb(cfg)

    # Unpack commonly used sections
    global TRAINING_CFG
    TRAINING_CFG = cfg.get("training", {})
    MODEL_CFG = cfg.get("model", {})
    DATA_CFG = cfg.get("data", {})
    SIM_CFG = cfg.get("simulation", {})
    RAY_CFG = cfg.get("ray", {})
    RES_CFG = cfg.get("resources", {}).get("client", {})
    SCAFFOLD_CFG = cfg.get("scaffold", {})
    STRAT_CFG = cfg.get("strategy", {})
    EVAL_CFG = cfg.get("evaluation", {})
    POST_CFG = cfg.get("post_training", {})

    # Initialize Ray from config
    ray.init(
        num_cpus=int(RAY_CFG.get("num_cpus", 32)),
        num_gpus=float(RAY_CFG.get("num_gpus", 4)),
        ignore_reinit_error=bool(RAY_CFG.get("ignore_reinit_error", True)),
    )
    print("Ray initialized.")
    print(f"Ray resources: {ray.available_resources()}")

    # 2. Define the client resources to 4cpus + 1gpu
    # Use all 4 GPUs for clients during training for maximum parallelism
    client_resources = {
        "num_cpus": int(RES_CFG.get("num_cpus", 4)),
        "num_gpus": float(RES_CFG.get("num_gpus", 1)),
    }

    # 3. Define the strategy
    # Define model configurations based on the original project's defaults
    # For PSM dataset, the number of features (channels) is 25
    seq_len = int(MODEL_CFG.get("seq_len", 100))
    time_cfg = MODEL_CFG.get("time", {})
    freq_cfg = MODEL_CFG.get("freq", {})
    time_model_args = {
        'win_size': seq_len,
        'enc_in': int(time_cfg.get('enc_in', 25)),
        'c_out': int(time_cfg.get('c_out', 25)),
        'e_layers': int(time_cfg.get('e_layers', 3)),
    }
    # Frequency grand-window keeps the feature dimension (25) as channels
    # Sequence length becomes (seq_len - nest_len + 1) * floor(nest_len/2)
    nest_len = int(freq_cfg.get('nest_length', 25))
    freq_model_args = {
        'win_size': (seq_len - nest_len + 1) * (nest_len // 2),
        'enc_in': int(freq_cfg.get('enc_in', 25)),
        'c_out': int(freq_cfg.get('c_out', 25)),
        'e_layers': int(freq_cfg.get('e_layers', 3)),
        'n_heads': int(freq_cfg.get('n_heads', 4)),
    }

    # Initialize model parameters
    temp_model = FederatedDualTF(time_model_args, freq_model_args)
    ndarrays = get_weights(temp_model)
    parameters = ndarrays_to_parameters(ndarrays)
    # Parameter-only templates for SCAFFOLD control variates (exclude buffers)
    param_templates = [p.detach().cpu().numpy() for p in temp_model.parameters()]

    # Preload centralized test dataloaders once to avoid rebuilding/grandwindow every round
    print("[Server] Preloading centralized test dataloaders...")
    testloader_time_global, testloader_freq_global = load_centralized_test_data(
        time_batch_size=int(DATA_CFG.get('time_batch_size', 128)),
        freq_batch_size=int(DATA_CFG.get('freq_batch_size', 8)),
        seq_length=int(DATA_CFG.get('seq_length', seq_len)),
        nest_length=nest_len,
    )
    print(f"[Server] Test sizes: time={len(testloader_time_global.dataset)}, freq={len(testloader_freq_global.dataset)}")

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

            # Aggregate and log training loss to wandb if available
            try:
                total_examples = 0
                sum_loss = 0.0
                for _, fr in filtered_results:
                    m = getattr(fr, "metrics", {}) or {}
                    if "train_loss" in m and isinstance(m["train_loss"], (int, float)):
                        n = int(getattr(fr, "num_examples", 0))
                        total_examples += n
                        sum_loss += float(m["train_loss"]) * n
                if total_examples > 0:
                    avg_loss = sum_loss / float(total_examples)
                    aggregated_metrics["train_loss"] = float(avg_loss)
                    log_metrics({"train/avg_loss": float(avg_loss), "train/clients_used": float(len(filtered_results))}, step=server_round)
            except Exception:
                pass

            # Update control variates (damped and finite)
            if deltas:
                sum_delta = [np.zeros_like(a) for a in self.global_c]
                alpha = float(SCAFFOLD_CFG.get('damping', 0.1))  # damping factor
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
    eval_state: Dict[str, float] = {}
    total_rounds: int = int(SIM_CFG.get("total_rounds", 10))

    def server_evaluate_fn(
        server_round: int,
        params: List,  # already a list of ndarrays provided by Flower
        config: Dict[str, fl.common.Scalar],
    ) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        # Evaluate every N rounds and on the final round if enabled
        if not bool(EVAL_CFG.get("enabled", False)):
            print(f"[Server] Skipping evaluation for round {server_round} (evaluate every 3 rounds & final round)")
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
        # Avoid expandable_segments due to allocator asserts on some PyTorch builds
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
        # Use the highest-index GPU (kept free by Ray) for server evaluation
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            device_index = torch.cuda.device_count() - 1
            try:
                torch.cuda.set_device(device_index)
            except Exception:
                pass
            device = torch.device(f"cuda:{device_index}")
        else:
            device = torch.device("cpu")
        print(f"[Server] Evaluating round {server_round} on central test set (device={device})...")
        print(f"[Server DEBUG] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}, cuda_count={torch.cuda.device_count()}")
        # Rebuild model and load parameters
        net = FederatedDualTF(time_model_args, freq_model_args)
        # params is already a list of ndarrays; set directly
        # If params contain non-finite values, skip evaluation
        if any((not np.all(np.isfinite(a))) for a in params):
            print("[Server] Skipping evaluation due to non-finite global parameters")
            return None
        set_weights(net, params)

        # Use preloaded test loaders
        testloader_time, testloader_freq = testloader_time_global, testloader_freq_global
        print(f"[Server DEBUG] testloader_time length: {len(testloader_time.dataset)}")
        print(f"[Server DEBUG] testloader_freq length: {len(testloader_freq.dataset)}")
        assert len(testloader_time.dataset) > 0, "Time dataloader is empty!"
        assert len(testloader_freq.dataset) > 0, "Freq dataloader is empty!"

        # Evaluate using existing test() which handles both models and threshold sweep
        # Use fast settings for periodic rounds; thorough on final round
        is_final = (server_round == total_rounds)
        thr_cnt = int(EVAL_CFG.get("num_thresholds_final", 1000)) if is_final else int(EVAL_CFG.get("num_thresholds_periodic", 256))
        step_val = int(EVAL_CFG.get("step", 5))

        loss, metrics = test(
            net,
            testloader_time,
            testloader_freq,
            device,
            eval_state=eval_state,
            min_consecutive=int(EVAL_CFG.get('min_consecutive', 10)),
            num_thresholds=thr_cnt,
            seq_length=int(EVAL_CFG.get('seq_length', 50)),
            step=step_val,
            threshold_quantile_range=tuple(EVAL_CFG.get('threshold_quantile_range', [0.01, 0.99])),
            prefer_gpu_sweep=bool(EVAL_CFG.get('prefer_gpu_sweep', True)),
        )
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
        # Log evaluation metrics to wandb
        try:
            to_log = {"server/loss": float(loss)}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    to_log[f"server/{k}"] = float(v)
            log_metrics(to_log, step=server_round)
        except Exception:
            pass
        return float(loss), metrics

    # Define strategy (standard partial participation): choose by config
    strategy_type = str(STRAT_CFG.get("type", "scaffold")).lower()
    if strategy_type == "fedprox":
        print("[Server] Using FedProx strategy")
        strategy = SaveResultsStrategy(
            fraction_fit=float(SIM_CFG.get('fraction_fit', 0.25)),
            fraction_evaluate=float(SIM_CFG.get('fraction_evaluate', 0.0)),
            min_available_clients=int(SIM_CFG.get('min_available_clients', 6)),
            min_fit_clients=int(SIM_CFG.get('min_fit_clients', 6)),
            initial_parameters=parameters,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            proximal_mu=float(TRAINING_CFG.get('proximal_mu', 0.0)),
            evaluate_fn=None if not bool(EVAL_CFG.get('enabled', False)) else server_evaluate_fn,
        )
    else:
        print("[Server] Using SCAFFOLD strategy")
        strategy = ScaffoldStrategy(
            fraction_fit=float(SIM_CFG.get('fraction_fit', 0.25)),
            fraction_evaluate=float(SIM_CFG.get('fraction_evaluate', 0.0)),
            min_available_clients=int(SIM_CFG.get('min_available_clients', 6)),
            min_fit_clients=int(SIM_CFG.get('min_fit_clients', 6)),
            initial_parameters=parameters,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            # Disable FedProx term when running SCAFFOLD; keep 0.0 for clarity
            proximal_mu=0.0,
            # Keep server evaluation controlled by EVAL_CFG.enabled; set evaluate_fn accordingly
            evaluate_fn=None if not bool(EVAL_CFG.get('enabled', False)) else server_evaluate_fn,
        )

    # 4. Start the simulation using fl.simulation.start_simulation
    print("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(SIM_CFG.get('num_clients', 24)),
        client_resources=client_resources,
        config=ServerConfig(num_rounds=total_rounds),
        # Use all 4 GPUs for training clients
        ray_init_args={
            "num_cpus": int(RAY_CFG.get('num_cpus', 32)),
            "num_gpus": float(RAY_CFG.get('num_gpus', 4)),
            "ignore_reinit_error": bool(RAY_CFG.get('ignore_reinit_error', True)),
        },
        strategy=strategy,
    )

    print("Simulation finished.")

    # === POST-TRAINING ARRAY GENERATION ===
    print("\n=== Starting Post-Training Array Generation ===")

    # Get the final model parameters from the strategy
    final_parameters = strategy.latest_parameters
    if final_parameters is None:
        print("[Warning] No final parameters found, using initial parameters")
        final_parameters = parameters

    # Convert parameters to numpy arrays
    final_params_numpy = parameters_to_ndarrays(final_parameters)

    # Setup device for post-training inference (any available GPU)
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        # Use GPU 0 (or any available GPU after training completes)
        post_train_device = torch.device("cuda:0")
        print(f"[Post-Training] Using {post_train_device} for array generation")
    else:
        post_train_device = torch.device("cpu")
        print(f"[Post-Training] Using {post_train_device} for array generation")

    # Create final model and load aggregated weights
    final_model = FederatedDualTF(time_model_args, freq_model_args)
    set_weights(final_model, final_params_numpy)
    final_model.to(post_train_device)

    # Generate evaluation arrays using centralized test data
    if bool(POST_CFG.get('generate_arrays', True)):
        print("[Post-Training] Generating evaluation arrays...")
        try:
            time_path, freq_path, time_df, freq_df = generate_evaluation_arrays(
                model=final_model,
                testloader_time=testloader_time_global,
                testloader_freq=testloader_freq_global,
                device=post_train_device,
                dataset=str(POST_CFG.get('dataset', 'PSM')),
                form=None,
                data_num=int(POST_CFG.get('data_num', 0)),
                seq_length=int(POST_CFG.get('seq_length', seq_len)),
                nest_length=int(POST_CFG.get('nest_length', nest_len)),
            )

            print(f"[Post-Training] Arrays saved to:")
            print(f"  - Time array: {time_path}")
            print(f"  - Freq array: {freq_path}")
            print(f"[Post-Training] Array shapes:")
            print(f"  - Time: {time_df.shape}")
            print(f"  - Freq: {freq_df.shape}")

            # Optionally log arrays as artifacts
            try:
                if os.path.isfile(time_path):
                    log_artifact(time_path, type_="array")
                if os.path.isfile(freq_path):
                    log_artifact(freq_path, type_="array")
            except Exception:
                pass

        except Exception as e:
            print(f"[Post-Training] Error generating arrays: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup post-training resources
    try:
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    print("=== Post-Training Array Generation Complete ===\n")
    print("Ready for evaluation! Use the generated arrays with evaluation_fl.py")

    # Finish wandb run if started
    try:
        wandb_finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
