import os
import torch
import torchvision
from torchvision.transforms import transforms

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from flwr.server.client_manager import SimpleClientManager


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
model.to(torch.float32)


# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        print("[Client] get_parameters")
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        new_parameters = [torch.tensor(p, dtype=torch.float32)
                          for p in parameters]
        for current_param, new_param in zip(self.model.parameters(), new_parameters):
            current_param.data = new_param

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client, round {server_round}] fit, config: {config}")

        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(local_epochs):
            for inputs, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    # NOTE: this is federated evaluation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                loss_sum += torch.nn.functional.cross_entropy(
                    outputs, labels, reduction="sum").item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        loss = loss_sum / total
        accuracy = correct / total
        return loss, total, {"accuracy": accuracy}


def client_fn(cid: str):
    # Return a standard Flower client

    # Load and preprocess your dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    return CifarClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader
    )


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "num_rounds": server_round,
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    # NOTE: my client resources
    # client get 5% of the CPU & no GPU
    my_client_resources = {'num_cpus': 0.05, 'num_gpus': 0.0}

    NUM_ROUNDS = 1
    # Specify number of FL rounds
    server_config = ServerConfig(num_rounds=NUM_ROUNDS)

    strategy = FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    client_manager = SimpleClientManager()

    # Launch the simulation
    hist = start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=2,  # Total number of clients available
        config=server_config,
        strategy=strategy,  # A Flower strategy
        client_resources=my_client_resources,
        client_manager=client_manager
    )


if __name__ == "__main__":
    main()
