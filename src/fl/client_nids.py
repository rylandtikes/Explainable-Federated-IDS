
# # IDS - Client Node
#
# ## Author  
# **Charles Stolz**  
# cstolz2@und.edu  
#
# ## Acknowledgments  
# This project builds upon the following open-source contributions:  
#
# ### Flower Federated Learning Framework  
# - Used for decentralized model training and updates across IDS nodes.  
# - **Documentation:** [Flower AI Docs](https://flower.ai/docs/)  
# - **GitHub:** [Flower GitHub](https://github.com/adap/flower)  
# - **Reference Paper:** Beutel, D.J., Topal, T., Mathur, A. et al. (2020). *Flower: A Friendly Federated Learning Framework.*  
#
from prometheus_client import start_http_server, Gauge, Counter
import time
import json
import logging
import hashlib
import os, time, psutil, numpy as np
import pandas as pd
import flwr as fl
import tensorflow as tf
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

#XAI_ENABLED = os.getenv("XAI", "0") == "1"
XAI_ENABLED = True

# Individual XAI method flags - set to True to enable each method
TEST_GRADIENT_INPUT = True   # Lightweight gradient-based explanations
TEST_SHAP = True            # SHAP explanations (computationally intensive)  
TEST_LIME = True            # LIME explanations (moderate overhead)

CANARY_PATH = os.getenv("CANARY_PATH", "/home/rtikes/Explainable-Federated-IDS/experiments/fedavg_xai/canary_64.csv")
proc = psutil.Process(os.getpid())

# Manual device type configuration for research paper
DEVICE_TYPE = "RaspberryPi-4B"  # Options: "RaspberryPi-4B", "RaspberryPi-5", "JetsonOrinNano"

print(f"XAI_CONFIG XAI_ENABLED={XAI_ENABLED} CANARY_PATH={CANARY_PATH}")

# Hardware platform identification for research paper
import platform
print(f"PLATFORM_INFO Device={DEVICE_TYPE} CPU={platform.machine()} Cores={os.cpu_count()} "
      f"RAM={psutil.virtual_memory().total/(1024**3):.1f}GB")

# Initialize metrics tracking
metrics_history = []

def log_metrics_to_file(round_num, metrics, node_name, training_time_s=None, peak_mem_mb=None, xai_results=None):
    """Log metrics to a clean file and maintain history for macro averaging"""
    global metrics_history
    
    # Add to history
    metrics_history.append({
        'round': round_num,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'], 
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'node': node_name
    })
    
    # Write current metrics with resource usage
    with open(METRICS_FILE, 'a') as f:
        # Basic metrics
        f.write(f"ROUND={round_num} NODE={node_name} ACC={metrics['accuracy']:.4f} "
                f"PREC={metrics['precision']:.4f} REC={metrics['recall']:.4f} "
                f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")
        
        # Add resource metrics if provided
        if training_time_s is not None:
            f.write(f" TRAIN_TIME_S={training_time_s:.3f}")
        if peak_mem_mb is not None:
            f.write(f" PEAK_MEM_MB={peak_mem_mb:.1f}")
        
        # Add XAI results if provided
        if XAI_ENABLED and xai_results:
            # Gradient×Input results
            if xai_results.get('gradxinput_ms') is not None:
                f.write(f" GRADXINPUT_MS={xai_results['gradxinput_ms']:.2f}")
            if xai_results.get('gradxinput_mem_mb') is not None:
                f.write(f" GRADXINPUT_MEM_MB={xai_results['gradxinput_mem_mb']:.1f}")
            
            # SHAP results
            if xai_results.get('shap_ms') is not None:
                f.write(f" SHAP_MS={xai_results['shap_ms']:.2f}")
            if xai_results.get('shap_mem_mb') is not None:
                f.write(f" SHAP_MEM_MB={xai_results['shap_mem_mb']:.1f}")
            
            # LIME results
            if xai_results.get('lime_ms') is not None:
                f.write(f" LIME_MS={xai_results['lime_ms']:.2f}")
            if xai_results.get('lime_mem_mb') is not None:
                f.write(f" LIME_MEM_MB={xai_results['lime_mem_mb']:.1f}")
            
            # Summary of enabled methods
            enabled_methods = []
            if TEST_GRADIENT_INPUT and xai_results.get('gradxinput_ms'): enabled_methods.append("GRAD")
            if TEST_SHAP and xai_results.get('shap_ms'): enabled_methods.append("SHAP") 
            if TEST_LIME and xai_results.get('lime_ms'): enabled_methods.append("LIME")
            
            if enabled_methods:
                f.write(f" XAI_METHODS={','.join(enabled_methods)}")
            else:
                f.write(" XAI_STATUS=FAILED")
        elif XAI_ENABLED:
            f.write(" XAI_STATUS=FAILED")
        else:
            f.write(" XAI_STATUS=DISABLED")
        
        f.write("\n")
    
    # Calculate macro average for last 10 rounds
    if len(metrics_history) >= 10:
        last_10 = metrics_history[-10:]
        macro_avg = {
            'accuracy': sum(m['accuracy'] for m in last_10) / 10,
            'precision': sum(m['precision'] for m in last_10) / 10,
            'recall': sum(m['recall'] for m in last_10) / 10,
            'f1': sum(m['f1'] for m in last_10) / 10,
            'auc': sum(m['auc'] for m in last_10) / 10
        }
        
        with open(METRICS_FILE, 'a') as f:
            f.write(f"MACRO_AVG_LAST10 NODE={node_name} ACC={macro_avg['accuracy']:.4f} "
                    f"PREC={macro_avg['precision']:.4f} REC={macro_avg['recall']:.4f} "
                    f"F1={macro_avg['f1']:.4f} AUC={macro_avg['auc']:.4f}\n")
        
        print(f"MACRO_AVG_LAST10 NODE={node_name} ACC={macro_avg['accuracy']:.4f} "
              f"PREC={macro_avg['precision']:.4f} REC={macro_avg['recall']:.4f} "
              f"F1={macro_avg['f1']:.4f} AUC={macro_avg['auc']:.4f}")

def grad_x_input(model, x_batch):
    """
    Gradient×Input XAI method for IoT-feasible explanations
    Computes feature importance via gradient-based attribution
    Suitable for resource-constrained federated IDS deployment
    """
    import tensorflow as tf
    x = tf.convert_to_tensor(x_batch.astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)  # CRITICAL: training=False for post-hoc analysis
    grads = tape.gradient(y, x)  # dy/dx
    # Element-wise multiplication for feature attribution scores
    attributions = grads.numpy() * x_batch
    return attributions

def measure_round(train_one_local_epoch_fn):
    """Measure training time and peak memory usage"""
    t0 = time.time()
    rss0 = proc.memory_info().rss
    train_one_local_epoch_fn()
    dur = time.time() - t0
    peak_mb = max(rss0, proc.memory_info().rss) / (1024**2)
    return dur, peak_mb

def explain_canary(model, scaler):
    """
    Post-hoc XAI analysis on canary dataset for feasibility assessment
    Tests selected XAI methods based on flags: TEST_GRADIENT_INPUT, TEST_SHAP, TEST_LIME
    Measures explanation latency and memory overhead on resource-constrained devices
    """
    if not (XAI_ENABLED and CANARY_PATH):
        return None, None
    
    try:
        import pandas as pd
        df = pd.read_csv(CANARY_PATH)
        X = df.drop(columns=["Label"]).values.astype(np.float32)
        X = scaler.transform(X) if scaler is not None else X
        # Sample up to 64 for consistent timing (per paper methodology)
        X = X[:64]
        
        enabled_methods = []
        if TEST_GRADIENT_INPUT: enabled_methods.append("Gradient×Input")
        if TEST_SHAP: enabled_methods.append("SHAP")
        if TEST_LIME: enabled_methods.append("LIME")
        
        print(f"XAI_INFO Running {len(enabled_methods)} XAI methods on {len(X)} canary samples: {', '.join(enabled_methods)}")
        
        xai_results = {}
        
        # Method 1: Gradient×Input (if enabled)
        if TEST_GRADIENT_INPUT:
            print("XAI_METHOD Testing Gradient×Input...")
            t0 = time.time()
            mem_before = proc.memory_info().rss / (1024**2)
            
            explanations_grad = grad_x_input(model, X)
            
            dt_grad = (time.time() - t0) * 1000.0  # ms
            mem_after_grad = proc.memory_info().rss / (1024**2)
            xai_results['gradxinput_ms'] = dt_grad / len(X)
            xai_results['gradxinput_mem_mb'] = mem_after_grad - mem_before
            print(f"  → GradientxInput: {xai_results['gradxinput_ms']:.2f}ms/sample, {xai_results['gradxinput_mem_mb']:.1f}MB overhead")
        
        # Method 2: SHAP (if enabled and available)
        if TEST_SHAP:
            try:
                import shap
                print("XAI_METHOD Testing SHAP...")
                t0 = time.time()
                mem_before = proc.memory_info().rss / (1024**2)
                
                # Use a subset for SHAP due to computational cost
                X_shap = X[:16]  # Reduce for SHAP overhead
                explainer = shap.KernelExplainer(model.predict, X_shap[:8])  # Background samples
                shap_values = explainer.shap_values(X_shap)
                
                dt_shap = (time.time() - t0) * 1000.0  # ms
                mem_after_shap = proc.memory_info().rss / (1024**2)
                xai_results['shap_ms'] = dt_shap / len(X_shap)
                xai_results['shap_mem_mb'] = mem_after_shap - mem_before
                print(f"  → SHAP: {xai_results['shap_ms']:.2f}ms/sample, {xai_results['shap_mem_mb']:.1f}MB overhead")
                
            except ImportError:
                print("XAI_WARNING SHAP not available - install with: pip install shap")
                xai_results['shap_ms'] = None
                xai_results['shap_mem_mb'] = None
            except Exception as e:
                print(f"XAI_ERROR SHAP failed: {e}")
                xai_results['shap_ms'] = None
                xai_results['shap_mem_mb'] = None
        
        # Method 3: LIME (if enabled and available) 
        if TEST_LIME:
            try:
                from lime import lime_tabular
                print("XAI_METHOD Testing LIME...")
                t0 = time.time()
                mem_before = proc.memory_info().rss / (1024**2)
                
                # LIME for tabular data
                X_lime = X[:16]  # Reduce for LIME overhead
                explainer = lime_tabular.LimeTabularExplainer(X_lime, mode='classification')
                
                # Explain one instance
                def model_predict_proba(x):
                    return np.column_stack([1 - model.predict(x), model.predict(x)])
                
                explanation = explainer.explain_instance(X_lime[0], model_predict_proba, num_features=10)
                
                dt_lime = (time.time() - t0) * 1000.0  # ms
                mem_after_lime = proc.memory_info().rss / (1024**2)
                xai_results['lime_ms'] = dt_lime  # Per instance
                xai_results['lime_mem_mb'] = mem_after_lime - mem_before
                print(f"  → LIME: {xai_results['lime_ms']:.2f}ms/instance, {xai_results['lime_mem_mb']:.1f}MB overhead")
                
            except ImportError:
                print("XAI_WARNING LIME not available - install with: pip install lime")
                xai_results['lime_ms'] = None
                xai_results['lime_mem_mb'] = None
            except Exception as e:
                print(f"XAI_ERROR LIME failed: {e}")
                xai_results['lime_ms'] = None
                xai_results['lime_mem_mb'] = None
        
        # Report comparative feasibility metrics
        device_info = f"{DEVICE_TYPE}-{os.cpu_count()}cores"
        print(f"XAI_FEASIBILITY [{device_info}] Comparative Results:")
        for method, ms in [('GradxInput', xai_results.get('gradxinput_ms')), 
                          ('SHAP', xai_results.get('shap_ms')), 
                          ('LIME', xai_results.get('lime_ms'))]:
            if ms is not None:
                print(f"  {method}: {ms:.2f}ms/sample")
            else:
                print(f"  {method}: DISABLED/UNAVAILABLE")
        
        # Return all XAI results for logging
        return xai_results
        
    except Exception as e:
        print(f"XAI_ERROR Failed to run XAI feasibility analysis: {e}")
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "../../configs/clients/beta.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CLIENT_ID = config["client_id"]
SERVER_ADDRESS = config["server_address"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
USE_GPU = config["use_gpu"]
DATASET_PATH = config["dataset_path"]

# Initialize metrics file after CLIENT_ID is loaded
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
METRICS_FILE = f"../../experiments/fedavg_xai/{CLIENT_ID}_metrics_{timestamp}.log"
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

# Clear metrics file at start
with open(METRICS_FILE, 'w') as f:
    f.write(f"# Metrics log for client {CLIENT_ID} - Run started at {datetime.now()}\n")
    f.write(f"# XAI_ENABLED={XAI_ENABLED} DEVICE_TYPE={DEVICE_TYPE}\n")
    f.write(f"# XAI Methods: GRADXINPUT={TEST_GRADIENT_INPUT} SHAP={TEST_SHAP} LIME={TEST_LIME}\n")
    f.write("# Format: ROUND=X NODE=Y ACC=Z ... [GRADXINPUT_MS=A GRADXINPUT_MEM_MB=B] [SHAP_MS=C SHAP_MEM_MB=D] [LIME_MS=E LIME_MEM_MB=F] XAI_METHODS=G,H,I\n")

if not USE_GPU:
    tf.config.set_visible_devices([], "GPU")
    logger.info("using CPU.")

df = pd.read_csv(DATASET_PATH)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

logger.info(f"dataset shape: {X.shape}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

import collections
original_counts = collections.Counter(y)
print(f"Original class distribution: {original_counts}")

df["label_binary"] = df.iloc[:, -1].apply(lambda x: 0 if x == "BENIGN" else 1)
y = df["label_binary"].values.astype(np.int32)

ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

oversampled_counts = collections.Counter(y_resampled)
print(f"After oversampling: {oversampled_counts}")

X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "Precision", "Recall", "AUC"])
    return model


def simulate_detection(X_stream, y_stream, model):
    logger.info("Starting real-time detection loop with Prometheus metrics...")
    attack_count = 0
    for x, y_true in zip(X_stream, y_stream):
        start = time.time()
        y_pred = model.predict(np.expand_dims(x, axis=0))[0][0]
        latency = (time.time() - start) * 1000
        inference_latency.set(latency)
        predicted_class.set(1 if y_pred >= 0.5 else 0)
        if y_pred >= 0.5:
            attack_count += 1
        alerts_triggered.set(attack_count)
        logger.info(f"Inference latency: {latency:.2f} ms | Prediction: {y_pred:.4f} | Total alerts: {attack_count}")
        time.sleep(1)

logger.info("initializing model")
model = build_model()

start_http_server(9100)

inference_latency = Gauge('inference_latency_ms', 'Latency per prediction in milliseconds')
alerts_triggered = Gauge('alerts_triggered', 'Number of attack predictions')
predicted_class = Gauge('predicted_class', 'Latest predicted class (0 or 1)')
model_training_rounds = Counter('model_training_rounds_total', 'Total number of FL training rounds completed')

# Round counter for metrics logging
round_counter = 0

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.training_time_s = None
        self.peak_mem_mb = None
        self.xai_results = None

    def get_parameters(self, config):
        return model.get_weights()

    def set_parameters(self, parameters):
        model.set_weights(parameters)

    def fit(self, parameters, config):
        global round_counter
        round_counter += 1
        
        logger.info(f"received training request from server start training round {round_counter}")
        self.set_parameters(parameters)
        
        def train_one_local_epoch_fn():
            """Pure training function - no XAI interference"""
            model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=7,
                verbose=2,
                validation_data=(X_val, y_val)
            )
        
        # Measure training performance
        dur_s, mem_mb = measure_round(train_one_local_epoch_fn)
        self.training_time_s = dur_s
        self.peak_mem_mb = mem_mb
        print(f"ROUND_METRICS time_s={dur_s:.3f} peak_mem_mb={mem_mb:.1f}")

        # Post-hoc XAI analysis (does NOT affect training)
        if XAI_ENABLED:
            logger.info("Running post-hoc XAI analysis...")
            xai_results = explain_canary(model, scaler)
            if xai_results:
                self.xai_results = xai_results
                print(f"XAI_SUMMARY Completed XAI analysis with results: {xai_results}")
            else:
                self.xai_results = None
                print("XAI_METRICS skipped (disabled or error)")
        else:
            self.xai_results = None
        
        logger.info(f"Training completed for round {round_counter}.")
        current_weights = self.get_parameters(config)
        return current_weights, len(X_train), {}

    def evaluate(self, parameters, config):
        global round_counter
        logger.info(f"evaluating model for round {round_counter}")
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, auc = model.evaluate(X_val, y_val)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Prepare metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        # Log metrics to file with resource usage
        log_metrics_to_file(
            round_counter, 
            metrics, 
            CLIENT_ID, 
            training_time_s=self.training_time_s,
            peak_mem_mb=self.peak_mem_mb,
            xai_results=self.xai_results
        )
        
        logger.info(f"Evaluation completed. Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
        return loss, len(X_val), {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

logger.info(f" starting flower Client {CLIENT_ID}. connecting to server at {SERVER_ADDRESS}...")
fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=FlowerClient())

# simulate_detection(X_val, y_val, model) 
