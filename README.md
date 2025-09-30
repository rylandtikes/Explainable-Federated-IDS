# Explainable-Federated-IDS

Explainable-Federated-IDS

## Overview

This project implements a lightweight, distributed Intrusion Detection System (IDS) tailored for resource-constrained IoT environments. It integrates:

- Federated Learning (via [Flower](https://flower.dev)) for collaborative anomaly detection across edge devices
- Support for heterogeneous edge devices including Raspberry Pi and ESP32
- XAI
This work supports experiments presented in the paper:

> **Towards Explainable Federated Intrusion Detection for IoT in Resource-Constrained Environments**  
> (Submitted to CARS 2025)

---

## System Architecture

- Raspberry Pi 4/5 nodes, Nvidia Orin,  host anomaly-based ML detection
- Model updates are shared via Flower
- ESP32 nodes simulate lightweight IoT sensors

---

## Features

- Network Based IDS with Federated anomaly detection using ANN
- Host Based IDS with Bert Mini
- Lightweight edge deployment using Raspberry Pi
- Evaluation using CICIDS2017, HDFS Datsets

---

## Dependencies
---

## Authors

| Author           | Email                     |
|------------------|----------------------------|
| Charles Stolz    | charles.stolz@und.edu      |
| Dr. Jielun Zhang | jielun.zhang@und.edu      |

---

## Related Work


---

## License

This project is licensed for research and educational use. For commercial or derivative uses, please contact the authors.

### Third-Party Licenses and Attribution
