"""DualFLSim: A Flower / PyTorch app."""

import os
import gc
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import io
import numpy as np

from dualflsim.task import (
    FederatedDualTF,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader_time, valloader_time, trainloader_freq, valloader_freq, partition_id):
        self.net = net
        self.trainloader_time = trainloader_time
        self.valloader_time = valloader_time
        self.trainloader_freq = trainloader_freq
        self.valloader_freq = valloader_freq
        # Ray assigns a specific GPU to this actor, which PyTorch sees as `cuda:0`
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Keep model on CPU by default; train/test moves submodules to device as needed


    def fit(self, parameters, config):
        local_epochs = config["local_epochs"]
        proximal_mu = config["proximal_mu"]
        lr = float(config.get("lr", 1e-5))
        # Decode SCAFFOLD controls if provided
        def unpack_bytes_to_ndarrays(data: bytes):
            buf = io.BytesIO(data)
            with np.load(buf) as npz:
                return [npz[k] for k in sorted(npz.files, key=lambda x: int(x[1:]))]
        c_bytes = config.get("c_bytes", None)
        ci_bytes = config.get("ci_bytes", None)
        control_c = control_ci = None
        if isinstance(c_bytes, (bytes, bytearray)) and isinstance(ci_bytes, (bytes, bytearray)):
            try:
                control_c = unpack_bytes_to_ndarrays(c_bytes)
                control_ci = unpack_bytes_to_ndarrays(ci_bytes)
            except Exception:
                control_c = control_ci = None

        set_weights(self.net, parameters)
        # If local_epochs is 0, return weights unchanged with zero contribution
        if int(local_epochs) == 0:
            # Cleanup to keep VRAM low between rounds
            try:
                self.net.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
            return get_weights(self.net), 0, {"train_loss": 0.0}
        # Capture weights before training for SCAFFOLD delta_ci computation
        w_before = [p.detach().cpu().numpy().copy() for p in self.net.parameters()]
        # Pass both training dataloaders to the train function
        train_loss, num_steps = train(
            self.net,
            self.trainloader_time,
            self.trainloader_freq,
            local_epochs,
            self.device,
            proximal_mu,
            lr=lr,
            control_c=control_c,
            control_ci=control_ci,
        )
        # Weights after training
        w_after = [p.detach().cpu().numpy().copy() for p in self.net.parameters()]
        # Cleanup: move model to CPU and clear CUDA caches to reduce VRAM residency
        try:
            self.net.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        # Return the total number of training samples
        num_examples = len(self.trainloader_time.dataset) + len(self.trainloader_freq.dataset)
        # Prepare metrics, include SCAFFOLD delta_ci if available
        metrics = {"train_loss": float(train_loss)}
        if control_c is not None and control_ci is not None and num_steps > 0:
            inv = 1.0 / (float(num_steps) * float(lr))
            alpha = 0.1  # damping
            # Compute raw deltas
            deltas = [(-c_np + inv * (wb - wa)).astype(wb.dtype, copy=False)
                      for wb, wa, c_np, ci_np in zip(w_before, w_after, control_c, control_ci)]
            # Global-norm clip
            total_sq = sum(float((d**2).sum()) for d in deltas)
            total_norm = total_sq ** 0.5
            max_norm = 5.0
            scale = 1.0
            if total_norm > max_norm:
                scale = max_norm / (total_norm + 1e-12)
            delta_ci = [alpha * (d * scale) for d in deltas]
            buf = io.BytesIO()
            np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(delta_ci)})
            metrics["delta_ci"] = buf.getvalue()
        return get_weights(self.net), num_examples, metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        # Pass both testing dataloaders to the test function
        loss, metrics = test(self.net, self.valloader_time, self.valloader_freq, self.device)
        # Cleanup after evaluation on client (precaution; server uses server-side eval)
        try:
            self.net.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()
        # Return the total number of testing samples
        num_examples = len(self.valloader_time.dataset) + len(self.valloader_freq.dataset)
        # Ensure the loss is a standard Python float
        return float(loss), num_examples, metrics


def client_fn(context: Context):
    # Align CPU threading with Ray CPU allocation per client
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    # CUDA allocator tuning (avoid expandable_segments due to allocator assert on some setups)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    except Exception:
        pass
    # Load model and data
    # Define model configurations based on the original project's defaults
    # For PSM dataset, the number of features (channels) is 25
    time_model_args = {'win_size': 100, 'enc_in': 25, 'c_out': 25, 'e_layers': 3}
    # The input to the frequency model depends on the `nest_length` used in data generation
    nest_length = 25
    freq_win_size = (100 - nest_length + 1) * (nest_length // 2)
    # Frequency grand-window keeps the feature dimension (25) as channels
    freq_model_args = {'win_size': freq_win_size, 'enc_in': 25, 'c_out': 25, 'e_layers': 3}
    net = FederatedDualTF(time_model_args, freq_model_args)
    
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    # Load all four dataloaders with smaller batch sizes to reduce GPU memory
    trainloader_time, valloader_time, trainloader_freq, valloader_freq = load_data(
        partition_id=partition_id, 
        num_partitions=num_partitions,
        time_batch_size=64,   # Safer peak VRAM for time model
        freq_batch_size=4,    # Safer peak VRAM for freq model
        dirichlet_alpha=2.0,  # Moderately balanced quantity split for PSM
    )

    # Return Client instance with all dataloaders
    return FlowerClient(net, trainloader_time, valloader_time, trainloader_freq, valloader_freq, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
