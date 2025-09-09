"""
Explainable-IDS - Server Node

This script runs on the server node in the IDS system.  
It is responsible for:
- Coordinating Federated Learning among IDS client nodes.
- Aggregating model updates received from clients.
- Managing global model updates and distributing them back to clients.
- Using the Flower Federated Learning framework for decentralized training.

Author:
- Charles Stolz (cstolz2@und.edu)

Acknowledgments:
This project builds upon the following open-source contributions:

Flower Federated Learning Framework:
- Used for decentralized machine learning model coordination and aggregation.
- Documentation: https://flower.ai/docs/
- GitHub: https://github.com/adap/flower
- Reference Paper: Beutel, D.J., Topal, T., Mathur, A. et al. (2020). 
  "Flower: A Friendly Federated Learning Framework."
"""

import os
import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from flwr.server.strategy import FedAvg
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "../../models/ann_model_server.keras"
INPUT_SHAPE = 78

# Build ANN model
def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(INPUT_SHAPE,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Loaded existing ANN model for server.")
    except Exception as e:
        logger.error(f"Error loading model: {e}. Rebuilding...")
        model = build_model()
else:
    model = build_model()
    logger.info("No saved model found. Creating new.")

# Get initial model parameters
initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

# Create FedAvg strategy with example-weighted aggregation
strategy = FedAvg(
    initial_parameters=initial_parameters,
    min_available_clients=1,  # Allow training with just 1 client
    min_fit_clients=1,        # Minimum clients for training
    min_evaluate_clients=0,   # No evaluation required
)

logger.info("Starting Flower Server with built-in FedAvg...")
fl.server.start_server(
    server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=1, round_timeout=300),
    strategy=strategy,
)

