import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Subset, DataLoader


def seperate_dataset(num_clients: int):

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
    samples_per_client_train = total_train_samples // num_clients

    total_test_samples = len(full_test_dataset)
    samples_per_client_test = total_test_samples // num_clients

    # Create subsets for each client
    train_datasets = []
    test_datasets = []

    for i in range(num_clients):
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
    for i in range(num_clients):
        print(
            f"Client {i+1}: Train samples - {len(train_datasets[i])}, Test samples - {len(test_datasets[i])}")

    return train_datasets, test_datasets

