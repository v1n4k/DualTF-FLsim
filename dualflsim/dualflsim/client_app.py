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
from utils.config import load_config

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
            k=5.0,
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
        # Client-side evaluation disabled: avoid any thresholding or heavy work.
        # Keep a placeholder shape for compatibility if ever invoked by strategy.
        return 0.0, 0, {"client_eval": "disabled"}


def client_fn(context: Context):
    # Align CPU threading with Ray CPU allocation per client
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    # CUDA allocator tuning (avoid expandable_segments due to allocator assert on some setups)
    # Avoid expandable_segments due to allocator asserts on some PyTorch builds
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    except Exception:
        pass
    # Load configuration (client reads same YAML; environment variable can point to a specific file)
    cfg = load_config(os.environ.get("DUALFLSIM_CONFIG"))
    MODEL_CFG = cfg.get("model", {})
    DATA_CFG = cfg.get("data", {})

    # Load model and data from config
    seq_len = int(MODEL_CFG.get("seq_len", 100))
    time_cfg = MODEL_CFG.get("time", {})
    freq_cfg = MODEL_CFG.get("freq", {})
    time_model_args = {
        'win_size': seq_len,
        'enc_in': int(time_cfg.get('enc_in', 25)),
        'c_out': int(time_cfg.get('c_out', 25)),
        'e_layers': int(time_cfg.get('e_layers', 3)),
    }
    nest_length = int(freq_cfg.get("nest_length", 25))
    freq_win_size = (seq_len - nest_length + 1) * (nest_length // 2)
    freq_model_args = {
        'win_size': freq_win_size,
        'enc_in': int(freq_cfg.get('enc_in', 25)),
        'c_out': int(freq_cfg.get('c_out', 25)),
        'e_layers': int(freq_cfg.get('e_layers', 3)),
        'n_heads': int(freq_cfg.get('n_heads', 4)),
    }
    net = FederatedDualTF(time_model_args, freq_model_args)
    
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    # Load all four dataloaders with smaller batch sizes to reduce GPU memory
    trainloader_time, valloader_time, trainloader_freq, valloader_freq = load_data(
        partition_id=partition_id, 
        num_partitions=num_partitions,
        time_batch_size=int(DATA_CFG.get('time_batch_size', 64)),
        freq_batch_size=int(DATA_CFG.get('freq_batch_size', 16)),
        dirichlet_alpha=float(DATA_CFG.get('dirichlet_alpha', 2.0)),
        seq_length=int(DATA_CFG.get('seq_length', seq_len)),
        nest_length=nest_length,
    )

    # Return Client instance with all dataloaders
    return FlowerClient(net, trainloader_time, valloader_time, trainloader_freq, valloader_freq, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
