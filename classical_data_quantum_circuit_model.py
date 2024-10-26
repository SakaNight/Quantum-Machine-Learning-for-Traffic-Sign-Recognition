# /opt/anaconda3/bin/python GTSRB_quantum_model.py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from typing import Tuple, List
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Modified preprocessing function with PCA
def preprocess_image(image_path: str, size: tuple = (32, 32), pca=None) -> np.ndarray:
    """Preprocess image and reduce dimensions using PCA"""
    try:
        # Load and resize image
        image = Image.open(image_path)
        image = image.resize(size)
        
        # Convert to numpy array and normalize
        normalized = np.array(image) / 255.0
        
        # Flatten and combine RGB channels
        r = normalized[:,:,0].flatten()
        g = normalized[:,:,1].flatten()
        b = normalized[:,:,2].flatten()
        combined = np.concatenate([r, g, b])
        
        # Apply PCA if provided
        if pca is not None:
            combined = pca.transform([combined])[0]
        
        # Normalize for quantum amplitude encoding
        norm = np.sqrt(np.sum(combined**2))
        if norm > 0:
            normalized_amplitude = combined / norm
        else:
            normalized_amplitude = combined
            
        return normalized_amplitude
        
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def prepare_batch(df: pd.DataFrame, pca=None, batch_size: int = 32) -> Tuple[List[np.ndarray], np.ndarray]:
    """Prepare a batch of images with PCA dimensionality reduction"""
    processed_images = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_images = [preprocess_image(path, pca=pca) for path in batch_df['Path']]
        processed_images.extend(batch_images)
    return processed_images, df['ClassId'].values

def conv_circuit(params):
    target = QuantumCircuit(4)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 2)
    target.cx(2, 3)
    target.ry(params[3], 3)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    num_conv_blocks = num_qubits // 4
    params = ParameterVector(param_prefix, length=num_conv_blocks * 4)
    
    for i in range(num_conv_blocks):
        param_index = i * 4
        qubits = range(i * 4, (i + 1) * 4)
        qc = qc.compose(conv_circuit(params[param_index:param_index + 4]), qubits)
        qc.barrier()

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    return target

def pool_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    num_pool_blocks = num_qubits // 2
    params = ParameterVector(param_prefix, length=num_pool_blocks * 2)
    
    for i in range(num_pool_blocks):
        param_index = i * 2
        qubits = [i * 2, i * 2 + 1]
        qc = qc.compose(pool_circuit(params[param_index:param_index + 2]), qubits)
        qc.barrier()

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def build_model(num_qubits=12):
    """Build the quantum CNN model with proper parameter management"""
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    
    # First Convolutional Layer: 12 qubits -> 12 qubits
    # Uses 3 conv blocks (4 qubits each)
    ansatz.compose(conv_layer(12, "c1"), range(12), inplace=True)
    
    # First Pooling Layer: 12 qubits -> 6 qubits
    # Uses 3 pool blocks (2 qubits each)
    ansatz.compose(pool_layer(6, "p1"), range(6), inplace=True)
    
    # Second Convolutional Layer: 6 qubits -> 6 qubits
    # Uses 1 conv block and 1 partial block
    ansatz.compose(conv_layer(6, "c2"), range(6), inplace=True)
    
    # Second Pooling Layer: 6 qubits -> 3 qubits
    # Uses 1 pool block and 1 partial block
    ansatz.compose(pool_layer(3, "p2"), range(3), inplace=True)
    
    # Final Convolutional Layer: 3 qubits -> 3 qubits
    # Uses 1 partial conv block
    ansatz.compose(conv_layer(3, "c3"), range(3), inplace=True)
    
    return ansatz

def save_weights(weights, filename="qcnn_weights.json"):
    with open(filename, "w") as f:
        json.dump(weights, f)

def load_weights(filename="qcnn_weights.json"):
    try:
        with open(filename, "r") as f:
            weights = json.load(f)
        return weights
    except FileNotFoundError:
        return None

def callback_graph(weights, obj_func_eval):
    global pbar
    pbar.set_postfix({
        'loss': f'{obj_func_eval:.5f}'
    })
    pbar.update(1)

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")

    # Get a balanced subset by sampling from two classes
    class_1_samples = train_df[train_df['ClassId'] == 1]
    class_2_samples = train_df[train_df['ClassId'] == 2]
    train_df = pd.concat([class_1_samples, class_2_samples])
    
    # Sample test data from the same classes
    test_class_1 = test_df[test_df['ClassId'] == 1]
    test_class_2 = test_df[test_df['ClassId'] == 2]
    test_df = pd.concat([test_class_1, test_class_2])
    
    print(f"Using {len(train_df)} training samples and {len(test_df)} test samples")

    num_epochs = 3
    total_iterations = 50

    # First, process a small batch to initialize PCA
    init_images = []
    for path in train_df['Path'].iloc[:1000]:  # Use first 1000 images for PCA
        image = Image.open(path)
        image = image.resize((32, 32))
        normalized = np.array(image) / 255.0
        flattened = normalized.reshape(-1)
        init_images.append(flattened)
    
    # Initialize and fit PCA
    print("Fitting PCA...")
    pca = PCA(n_components=12)  # Reduce to 12 dimensions to match quantum circuit
    pca.fit(init_images)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_.cumsum()[-1])
    
    # Prepare training data with PCA
    print("Preparing training data...")
    train_images, train_labels = prepare_batch(train_df, pca=pca, batch_size=32)
    train_images = np.array(train_images)
    # train_labels = np.array(train_labels)
    
    # Convert labels to binary (-1, 1)
    train_labels = np.where(train_labels == 1, 1, -1)
    
    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    print("Input shape after PCA:", x_train.shape)
    
    # Build quantum circuit
    feature_map = ZFeatureMap(12)
    ansatz = build_model(12)
    
    # Combine feature map and ansatz
    circuit = QuantumCircuit(12)
    circuit.compose(feature_map, range(12), inplace=True)
    circuit.compose(ansatz, range(12), inplace=True)
    
    # Define observable
    observable = SparsePauliOp.from_list([("Z" + "I" * 11, 1)])
    
    # Create QNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    
    # Load pre-trained weights if available
    initial_point = load_weights()
    
    # Create classifier
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=total_iterations),
        callback=callback_graph,
        initial_point=initial_point
    )
    
    # Training loop
    for epoch in range(num_epochs):
        global pbar
        pbar = tqdm(total=total_iterations, 
                   desc=f"Epoch [{epoch+1}/{num_epochs}]", 
                   unit='it')
        
        classifier.fit(x_train, y_train)
        pbar.close()
        
        # Compute metrics
        train_accuracy = classifier.score(x_train, y_train)
        val_accuracy = classifier.score(x_val, y_val)
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Validation accuracy: {val_accuracy:.3f}")
    
    # Save weights
    if hasattr(classifier, '_fit_result') and hasattr(classifier._fit_result, 'x'):
        trained_weights = classifier._fit_result.x.tolist()
        save_weights(trained_weights)
        print("Saved trained weights to 'qcnn_weights.json'")
    
    # Test predictions
    print("Preparing test data...")
    test_images, test_labels = prepare_batch(test_df, pca=pca, batch_size=32)
    test_images = np.array(test_images)
    test_labels = np.where(np.array(test_labels) == 1, 1, -1)
    
    test_accuracy = classifier.score(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy:.3f}")

if __name__ == "__main__":
    main()