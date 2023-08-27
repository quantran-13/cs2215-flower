import flwr as fl
from flwr.common import Metrics
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config


def main():
    client_manager = SimpleClientManager()
    # client_manager = (SimpleClientManager([client]),)
    print("Number of clients: ", len(client_manager))
    strategy = FedAvg(
        # fraction_fit=1.0,
        # fraction_evaluate=0.5,
        # min_fit_clients=10,
        # min_evaluate_clients=5,
        # min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,  # Pass the metric aggregation function
        # initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config,  # Pass the fit_config function to the server
    )

    # Start the Flower server and connect the client
    server_config = ServerConfig(num_rounds=10)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        client_manager=client_manager,
        strategy=strategy,
    )
    print("Number of clients: ", len(client_manager))


if __name__ == "__main__":
    main()
