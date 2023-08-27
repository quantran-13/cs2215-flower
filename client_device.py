import os
import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, xy_train, xy_test):
        self.model = model
        self.xy_train = xy_train
        self.xy_test = xy_test

    def get_parameters(self, config):  # override
        print("[Client] get_parameters")
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client, round {server_round}] fit, config: {config}")
        self.model.set_weights(parameters)
        self.model.fit(
            self.xy_train[0],
            self.xy_train[1],
            epochs=local_epochs,
            batch_size=32,
            verbose=0,
        )
        # NOTE: return parameters, num_examples, metrics
        return self.model.get_weights(), len(self.xy_train[0]), {}

    def evaluate(self, parameters, config):
        print(f"[Client] evaluate, config: {config}")
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.xy_test[0], self.xy_test[1])
        return float(loss), len(self.xy_test[0]), {"accuracy": accuracy}


def main():
    # Load model and data (MobileNetV2, CIFAR-10)
    model = tf.keras.applications.MobileNetV2(
        (32, 32, 3), classes=10, weights=None
    )
    model.compile(
        "adam", "sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Load and preprocess your dataset (for illustration purposes, we assume you have loaded CIFAR-10 data)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create the Flower client
    client = CifarClient(
        model=model,
        xy_train=(x_train, y_train),
        xy_test=(x_test, y_test),
    )

    # Start the training process
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
