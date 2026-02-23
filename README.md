# Robust Keyword Spotting (KWS) for Edge Devices 🎙️⚙️

> **Status: Work In Progress (WIP) 🚧**
> *The data pipeline and model architecture have been designed and implemented. Currently in the testing and training phase, to be followed by Edge optimization (Quantization).*

## Project Overview
This project aims to build a robust, production-ready Keyword Spotting (KWS) system designed specifically for resource-constrained Edge devices. The focus is not just on accuracy in a quiet room, but on **Real-World Robustness** against background noise and out-of-vocabulary words.

## Key Engineering & Design Decisions:

### 1. Data Augmentation Strategy (Clean vs. Noisy)
- **Training Set:** Deliberately split into 50% clean and 50% noisy data. This forces the model to learn the acoustic features of the keywords independently from the noise, preventing the network from memorizing noise as part of the target audio.
- **Evaluation Strategy (Dual Test Sets):** The model is evaluated on two separate test sets:
  - `Test_Clean`: To establish a baseline of how well the model learned the ideal features (checking for high Bias/Variance).
  - `Test_Noisy`: To measure the true robustness and degradation in real-world environments.

### 2. Handling the "Continuous Stream" Problem
In a real-world Edge scenario, the microphone is always listening. To prevent false positives:
- **`Unknown` Class:** Added to train the model to aggressively reject out-of-vocabulary words rather than forcing them into a known keyword category.
- **`Silence` Class:** Trained explicitly on background noise segments to teach the model to ignore ambient sounds when no speech is present.

### 3. Future Roadmap: Edge Deployment 🚀
The final phase of this project will involve converting the trained TensorFlow model into **TFLite**. I will implement **Post-Training Quantization (PTQ)** to reduce the model size and inference latency, making it deployable on microcontrollers with strict memory limits.
