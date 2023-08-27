import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg


def main():

    client_manager = SimpleClientManager()
    strategy = FedAvg()

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=3,
        ),
        client_manager=client_manager,
        strategy=strategy
    )


if __name__ == "__main__":
    main()
