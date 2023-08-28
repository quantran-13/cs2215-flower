import os
import csv
import time
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Subset, DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from flwr.server.client_manager import SimpleClientManager

# Define a function to save metrics to a CSV file
def save_metrics_to_csv(filename, metrics_list):
    if not os.path.exists("metrics"):
        os.makedirs("metrics")
    file_path = os.path.join("metrics", filename)

    with open(file_path, mode="w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "validation_loss",
                        "validation_accuracy", "epoch_time"])
        writer.writerows(metrics_list)

# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, test_loader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        print("[Client] get_parameters")
        self.model.cpu()  # Transfer model to CPU
        return [p.detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters, config):
        new_parameters = [torch.tensor(p, dtype=torch.float32)
                          for p in parameters]
        for current_param, new_param in zip(self.model.parameters(), new_parameters):
            # Transfer parameters to the specified device (CPU/GPU)
            current_param.data = new_param.to(device)

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        print(f"[Client, round {server_round}] fit, config: {config}")

        self.set_parameters(parameters, config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        metrics_list = []

        for epoch in range(local_epochs):
            start_time = time.time()
            total_loss = 0.0

            for inputs, labels in self.train_loader:
                # transfer to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            epoch_time = time.time() - start_time
            avg_loss = total_loss / len(self.train_loader)
            metrics_list.append([epoch, avg_loss, 0.0, 0.0, epoch_time])

        save_metrics_to_csv(
            f"client_{self.client_id}_train_metrics.csv", metrics_list)
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    # NOTE: this is federated evaluation
    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                # transfer to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)

                loss_sum += torch.nn.functional.cross_entropy(
                    outputs, labels, reduction="sum").item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        loss = loss_sum / total
        accuracy = correct / total

        metrics_list = [[0, 0.0, loss, accuracy, 0.0]]
        save_metrics_to_csv(
            f"client_{self.client_id}_eval_metrics.csv", metrics_list)

        return loss, total, {"accuracy": accuracy}


def client_fn(cid: str):
    # Load model and data (MobileNetV2, CIFAR-10)
    model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
    model.to(torch.float32)

    # Load and preprocess your dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # TODO: fit data and assign to different client
    return CifarClient(
        client_id=cid,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader
    )

def seperate_dataset():

    # Data transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load full CIFAR-10 datasets
    full_train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    full_test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    # Calculate samples per client
    total_train_samples = len(full_train_dataset)
    samples_per_client_train = total_train_samples // NUM_CLIENTS

    total_test_samples = len(full_test_dataset)
    samples_per_client_test = total_test_samples // NUM_CLIENTS

    # Create subsets for each client
    train_datasets = []
    test_datasets = []

    for i in range(NUM_CLIENTS):
        train_start_idx = i * samples_per_client_train
        train_end_idx = (i + 1) * samples_per_client_train
        train_subset = Subset(
            full_train_dataset, list(range(train_start_idx, train_end_idx)))
        train_datasets.append(train_subset)

        test_start_idx = i * samples_per_client_test
        test_end_idx = (i + 1) * samples_per_client_test
        test_subset = Subset(
            full_test_dataset, list(range(test_start_idx, test_end_idx)))
        test_datasets.append(test_subset)

    # Printing the number of samples in each client's dataset
    for i in range(NUM_CLIENTS):
        print(
            f"Client {i+1}: Train samples - {len(train_datasets[i])}, Test samples - {len(test_datasets[i])}")

    return train_datasets, test_datasets




def client_fn_gpu(cid: str):
    # Load model (MobileNetV2)
    model = torchvision.models.mobilenet_v2(
        pretrained=False, num_classes=10).to(device)

    # Load train and test datasets for the specific client
    train_dataset = client_train_datasets[int(cid)]
    test_dataset = client_test_datasets[int(cid)]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Move the model and data loader to the same device
    model.to(device)

    return CifarClient(
        client_id=cid,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader
    )


def fit_config(server_round: int, local_epochs: int  ):
    """Return training configuration dict for each round.
    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "num_rounds": server_round,
        "local_epochs": local_epochs,  #
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
    # client get 5% of the CPU & 10% GPU because
    # estimate from Raspberrypi 4GB to RTX 2070 & Ryzen 5 2600
    my_client_resources = {'num_cpus': 0.05, 'num_gpus': 0.1}

    # Specify number of FL rounds
    server_config = ServerConfig(num_rounds=NUM_ROUNDS)

    strategy = FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    client_manager = SimpleClientManager()

    # Launch the simulation
    hist = start_simulation(
        # client_fn=client_fn,  # A function to run a _virtual_ client when required
        client_fn=client_fn_gpu,  # A function to run a _virtual_ client when required
        num_clients=2,  # Total number of clients available
        config=server_config,
        strategy=strategy,  # A Flower strategy
        client_resources=my_client_resources,
        client_manager=client_manager
    )


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_CLIENTS = 2
    NUM_ROUNDS = 1
    LOCAL_EPOCHS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_train_datasets, client_test_datasets = seperate_dataset()
    main()
