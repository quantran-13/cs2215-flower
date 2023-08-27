import flwr as fl


def main():
    # Define the aggregation strategy (Federated Averaging)
    strategy = fl.server.strategy.FedAvg()

    # Define the central server with the aggregation strategy
    server = fl.server.Server(strategy=strategy)

    # Start the server on a specific host and port
    server.start(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
