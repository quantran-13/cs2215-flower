import flwr as fl
import tensorflow as tf


# Define a simple Keras model
def create_compiled_keras_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train, y_train = x_train / 255.0, y_train

    # Create a Flower client
    client = fl.client.Client("localhost:8080", grpc=True)

    # Start the client and connect to the central server
    client.start()

    # Define a compiled Keras model
    model = create_compiled_keras_model()

    # Define the training loop
    for round_num in range(10):  # Adjust the number of rounds as needed
        # Get the global model from the server
        model_weights = client.get_parameters()

        # Set the model's weights to the global model
        model.set_weights(model_weights)

        # Train the model on local data
        model.fit(x_train, y_train, epochs=1)

        # Get the updated model weights
        new_weights = model.get_weights()

        # Send the updated weights to the server
        client.send_parameters(new_weights)


if __name__ == "__main__":
    main()
