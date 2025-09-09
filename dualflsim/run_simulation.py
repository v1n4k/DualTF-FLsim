import ray
import flwr as fl
from flwr.server.server import ServerConfig
from flwr.server.strategy import FedProx
from flwr.common import ndarrays_to_parameters, Metrics
from typing import List, Tuple, Dict

# Import components from your project
from dualflsim.client_app import client_fn
from dualflsim.task import FederatedDualTF, get_weights


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        "local_epochs": 1,
        "proximal_mu": 0.1  # Set the proximal term for FedProx
    }
    return config


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate F1, precision, and recall."""
    # Calculate weighted average for each metric
    f1_aggregated = sum([m["f1"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    precision_aggregated = sum([m["precision"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    recall_aggregated = sum([m["recall"] * num_examples for num_examples, m in metrics]) / sum([num_examples for num_examples, _ in metrics])
    
    return {"f1": f1_aggregated, "precision": precision_aggregated, "recall": recall_aggregated}


def main():
    # 1. Initialize Ray explicitly and tell it about available GPUs
    # Disable the disk space warning by setting an environment variable
    ray.init(num_gpus=4)
    print("Ray initialized.")
    print(f"Ray resources: {ray.available_resources()}")

    # 2. Define the client resources directly in code.
    client_resources = {"num_cpus": 1, "num_gpus": 1}

    # 3. Define the strategy
    # Define model configurations based on the original project's defaults
    time_model_args = {'win_size': 100, 'enc_in': 1, 'c_out': 1, 'e_layers': 3}
    freq_model_args = {'win_size': (100 - 25 + 1) * (25 // 2), 'enc_in': 1, 'c_out': 1, 'e_layers': 3}

    # Initialize model parameters
    temp_model = FederatedDualTF(time_model_args, freq_model_args)
    ndarrays = get_weights(temp_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedProx(
        fraction_fit=0.25,  # Train on 25% of clients per round
        fraction_evaluate=1, # Evaluate on 100% of clients
        min_available_clients=55,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
        proximal_mu=0.1
    )

    # 4. Start the simulation using fl.simulation.start_simulation
    print("Starting Flower simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=55,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=20), # More rounds to see convergence
        strategy=strategy,
    )

    # 5. Save results to a file
    summary_file = "simulation_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Federated Learning Simulation Summary\n")
        f.write("="*40 + "\n\n")
        
        # Write loss history
        if history.losses_distributed:
            f.write("Loss (Distributed):\n")
            for round_num, loss in history.losses_distributed:
                f.write(f"  Round {round_num}: {loss}\n")
        
        # Write metrics history
        if history.metrics_distributed:
            f.write("\nMetrics (Distributed, Evaluate):\n")
            # Transpose the metrics dictionary for easier writing
            metrics_by_round = {}
            for metric, values in history.metrics_distributed.items():
                for round_num, value in values:
                    if round_num not in metrics_by_round:
                        metrics_by_round[round_num] = {}
                    metrics_by_round[round_num][metric] = value
            
            for round_num, metrics in sorted(metrics_by_round.items()):
                f.write(f"  Round {round_num}:\n")
                for metric, value in metrics.items():
                    f.write(f"    {metric}: {value}\n")

    print(f"Summary saved to {summary_file}")

    # 6. Shut down Ray
    print("Shutting down Ray...")
    ray.shutdown()
    print("Simulation finished.")


if __name__ == "__main__":
    main()
