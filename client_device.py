import os
import flwr as fl
import torch
import torchvision
import torchvision.transforms as transforms

# Make PyTorch log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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


def main():
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    # Create the Flower client
    client = CifarClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    # Start the training process
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
