import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def load_datasets(batch_size: int = 32):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = CIFAR10(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        "./data", train=False, download=True, transform=transform
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dataloader, test_dataloader


def get_parameters(net) -> list[np.ndarray]:
    return [p.cpu().detach().numpy() for p in net.parameters()]


def set_parameters(net, parameters: list[np.ndarray]):
    new_parameters = [torch.tensor(p, dtype=torch.float32) for p in parameters]
    for current_param, new_param in zip(net.parameters(), new_parameters):
        current_param.data = new_param


def train(net, trainloader, epochs: int, device: torch.device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, device: torch.device):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
