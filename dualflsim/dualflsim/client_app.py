"""DualFLSim: A Flower / PyTorch app."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

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
        self.net.to(self.device)


    def fit(self, parameters, config):
        local_epochs = config["local_epochs"]
        proximal_mu = config["proximal_mu"]
        set_weights(self.net, parameters)
        # Pass both training dataloaders to the train function
        train_loss = train(
            self.net,
            self.trainloader_time,
            self.trainloader_freq,
            local_epochs,
            self.device,
            proximal_mu,
        )
        # Return the total number of training samples
        num_examples = len(self.trainloader_time.dataset) + len(self.trainloader_freq.dataset)
        return get_weights(self.net), num_examples, {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        # Pass both testing dataloaders to the test function
        loss, metrics = test(self.net, self.valloader_time, self.valloader_freq, self.device)
        # Return the total number of testing samples
        num_examples = len(self.valloader_time.dataset) + len(self.valloader_freq.dataset)
        # Ensure the loss is a standard Python float
        return float(loss), num_examples, metrics


def client_fn(context: Context):
    # Load model and data
    # Define model configurations based on the original project's defaults
    time_model_args = {'win_size': 100, 'enc_in': 1, 'c_out': 1, 'e_layers': 3}
    # The input to the frequency model depends on the `nest_length` used in data generation
    nest_length = 25
    freq_win_size = (100 - nest_length + 1) * (nest_length // 2)
    freq_model_args = {'win_size': freq_win_size, 'enc_in': 1, 'c_out': 1, 'e_layers': 3}
    net = FederatedDualTF(time_model_args, freq_model_args)
    
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    # Load all four dataloaders with different batch sizes
    trainloader_time, valloader_time, trainloader_freq, valloader_freq = load_data(
        partition_id=partition_id, 
        num_partitions=num_partitions,
        time_batch_size=64,  # Keep original batch size for time model
        freq_batch_size=8    # Use a smaller batch size for frequency model
    )

    # Return Client instance with all dataloaders
    return FlowerClient(net, trainloader_time, valloader_time, trainloader_freq, valloader_freq, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
