# Quantum Machine Learning for Traffic Sign Recognition

Implementation of classical, hybrid, and quantum approaches for traffic sign recognition using TensorFlow Quantum, Qiskit, and PennyLane frameworks.

## Project Structure

```
.
├── generate_dataset.py          # Dataset preprocessing for GTSRB dataset
├── classical_and_hybrid_model.py # Hybrid quantum-classical model (PennyLane)
├── quantum_model_binary.py      # Binary quantum classifier (TF Quantum)
├── quantum_model_10cls.py       # 10-class quantum classifier (Qiskit)
└── inference.py                 # Unified inference pipeline
```

## Models

### Classical CNN
- Sequential model with Conv2D, MaxPooling2D layers
- Input: 28x28 grayscale images
- Output: 43 traffic sign classes

### Hybrid Quantum-Classical (PennyLane)
- Classical CNN backbone + quantum feature processing
- 4-qubit quantum circuit with RandomLayers
- 2x2 image block quantum encoding

### Pure Quantum Models
1. Binary Classifier (TensorFlow Quantum):
   - 16 qubits (4x4 grid)
   - Direct image-to-qubit mapping
   - Parameterized quantum circuit (PQC) layer

2. Multi-class QCNN (Qiskit):
   - 8-qubit circuit with ZFeatureMap
   - Quantum convolution and pooling layers
   - One-vs-all strategy for 10 classes

## Setup

Required packages:
```bash
tensorflow==2.13.0
tensorflow-quantum==0.7.0
qiskit==0.44.0
pennylane==0.31.0
cirq==1.2.0
scikit-learn==1.3.0
pillow==10.0.0
numpy==1.23.5
```

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB):
1. Generate dataset:
```bash
python generate_dataset.py
```

## Training

```bash
# Classical-Quantum Hybrid
python classical_and_hybrid_model.py

# Binary Quantum Classification
python quantum_model_binary.py

# 10-class Quantum CNN
python quantum_model_10cls.py
```

## Inference

```bash
python inference.py
# Select model type when prompted (classical/quantum)
```


