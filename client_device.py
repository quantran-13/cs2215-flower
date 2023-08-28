import os

import flwr as fl
import torch
from torchvision.models import mobilenet_v2

from helper_func import (
    get_parameters,
    load_datasets,
    set_parameters,
    test,
    train,
)

# Make PyTorch log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, net, train_dataloader, test_dataloader):
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_parameters(self, config):
        # server_round = config["server_round"]
        # print(
        #     f"[Client, round {server_round}] get_parameters, config: {config}"
        # )

        return get_parameters(self.net)

    def set_parameters(self, parameters, config):
        # server_round = config["server_round"]
        # print(
        #     f"[Client, round {server_round}] set_parameters, config: {config}"
        # )

        set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Client, round {server_round}] fit, config: {config}")

        print("len(self.train_dataloader): ", len(self.train_dataloader))
        print(
            "len(self.train_dataloader.dataset): ",
            len(self.train_dataloader.dataset),
        )

        self.set_parameters(parameters, config)
        train(
            self.net,
            self.train_dataloader,
            epochs=int(local_epochs),
            device=device,
        )

        return self.get_parameters({}), len(self.train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        server_round = config["server_round"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Client, round {server_round}] evaluate, config: {config}")

        self.set_parameters(parameters, config)
        loss, accuracy = test(self.net, self.test_dataloader, device=device)
        print(f"Client-side evaluation loss {loss} / accuracy {accuracy}")

        return (
            float(loss),
            len(self.test_dataloader.dataset),
            {"accuracy": float(accuracy)},
        )


def main():
    # Load model and data (MobileNetV2, CIFAR-10)
    net = mobilenet_v2(weights=None, num_classes=10)
    net.to(torch.float32)

    # Load and preprocess your dataset
    train_dataloader, test_dataloader = load_datasets(batch_size=32)

    # Create the Flower client
    client = CifarClient(
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    # Start the training process
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
