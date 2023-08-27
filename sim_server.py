import os

import flwr as fl
import tensorflow as tf

from flwr.server.strategy import FedAvg


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}




# each client gets 1xCPU (this is the default if no resources are specified)
# my_client_resources = {'num_cpus': 1, 'num_gpus': 0.0}
# # each client gets 2xCPUs and half a GPU. (with a single GPU, 2 clients run concurrently)
# my_client_resources = {'num_cpus': 2, 'num_gpus': 0.5}
# # 10 client can run concurrently on a single GPU, but only if you have 20 CPU threads.
# my_client_resources = {'num_cpus': 2, 'num_gpus': 0.1}

# NOTE: my client resources
# client get 5% of the CPU & no GPU  
my_client_resources = {'num_cpus': 0.05, 'num_gpus': 0.0}





def client_fn(cid: str):
    # Return a standard Flower client
    return CifarClient()

# Launch the simulation
hist = fl.simulation.start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=10, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=3), # Specify number of FL rounds
    strategy=FedAvg(), # A Flower strategy
    client_resources = my_client_resources # A Python dict specifying CPU/GPU resources
)
