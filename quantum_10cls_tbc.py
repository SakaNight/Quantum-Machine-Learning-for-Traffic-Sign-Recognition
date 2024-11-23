import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

def preprocess_image(image_path: str, size: tuple = (32, 32)) -> np.ndarray:
    """需要确保输出是8维特征向量"""
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        normalized = np.array(image) / 255.0
        
        if len(normalized.shape) == 3:
            grayscale = np.mean(normalized, axis=2)
        else:
            grayscale = normalized
            
        # 提取8个特征
        h, w = grayscale.shape
        reduced = np.zeros(8)
        pool_size = h // 2
        
        for i in range(8):
            row_start = (i // 4) * pool_size
            row_end = row_start + pool_size
            col_start = (i % 4) * pool_size
            col_end = col_start + pool_size
            reduced[i] = np.mean(grayscale[row_start:row_end, col_start:col_end])
            
        return reduced
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(8)
    
def create_encoding_circuit(input_data):
    """Create quantum circuit for encoding the input data."""
    num_qubits = 8  # 12 -> log2(3072) ≈ 12
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Apply initial Hadamard gates
    circuit.h(qr)

    # Encode amplitudes using controlled rotations
    for i in range(min(len(input_data), 2**num_qubits)):
        if abs(input_data[i]) > 1e-10:
            binary = format(i, f'0{num_qubits}b')
            
            # Apply X gates for control qubits
            for j, bit in enumerate(binary):
                if bit == '1':
                    circuit.x(qr[j])
            
            # Calculate rotation angle
            angle = 2 * np.arccos(np.sqrt(abs(input_data[i]))) if input_data[i] > 0 else 0
            
            # Multi-controlled rotation
            if num_qubits > 1:
                circuit.mcry(
                    theta=angle,
                    q_controls=[qr[j] for j in range(num_qubits-1)],
                    q_target=qr[-1]
                )
            else:
                circuit.ry(angle, qr[0])
            
            # Uncompute X gates
            for j, bit in enumerate(binary):
                if bit == '1':
                    circuit.x(qr[j])

    return circuit

def conv_circuit(params):
    """Create a two-qubit convolutional circuit."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    """Create a convolutional layer."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    param_index = 0
    
    # Apply convolutions to adjacent pairs
    for i in range(0, num_qubits - 1, 2):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [i, i + 1])
        qc.barrier()
        param_index += 3
    
    return qc

def pool_circuit(params):
    """Create a pooling circuit."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    """Create a pooling layer that operates on specific qubit pairs."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    
    for source, sink in zip(sources, sinks):
        # Create subcircuit for this pair of qubits
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    return qc

def build_qcnn_model():
    """Build the complete QCNN model with corrected qubit handling."""
    num_qubits = 8
    
    # Create the feature map circuit for 8 input features
    feature_map = ZFeatureMap(num_qubits)
    
    # Create the ansatz (QCNN layers)
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")
    
    # Layer 1: First Convolutional Layer (8 qubits)
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    
    # Layer 1: First Pooling Layer (8 -> 4 qubits)
    # Handle pairs (0,1), (2,3), (4,5), (6,7)
    for i in range(0, num_qubits, 2):
        pool_circuit_params = ParameterVector(f"p1_{i}", 3)
        ansatz.compose(pool_circuit(pool_circuit_params), [i, i+1], inplace=True)
    
    # Layer 2: Second Convolutional Layer (4 qubits on first half)
    ansatz.compose(conv_layer(4, "c2"), list(range(4)), inplace=True)
    
    # Layer 2: Second Pooling Layer (4 -> 2 qubits on first half)
    pool_params_2 = ParameterVector("p2", 3)
    ansatz.compose(pool_circuit(pool_params_2), [0, 1], inplace=True)
    
    # Layer 3: Final Convolution (2 qubits)
    ansatz.compose(conv_layer(2, "c3"), [0, 1], inplace=True)
    
    # Combine feature map and ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    
    # Define observable (measure first qubit)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    
    # Create QNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    
    return qnn

def create_classifier(qnn, initial_point=None):
    """Create a quantum neural network classifier."""
    return NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=200),
        initial_point=initial_point
    )

class QCNN:
    def __init__(self):
        self.qnn = build_qcnn_model()
        self.classifier = create_classifier(self.qnn)
        
    def fit(self, X, y):
        """Train the QCNN model."""
        # Ensure input shape matches expected dimensions
        if X.shape[1] != 8:
            raise ValueError(f"Input data must have 8 features, got {X.shape[1]}")
        return self.classifier.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.classifier.predict(X)
    
    def score(self, X, y):
        """Calculate the accuracy score."""
        return self.classifier.score(X, y)

class MulticlassQCNN:
    def __init__(self, num_classes):
        """Initialize a multi-class QCNN using multiple binary classifiers."""
        self.num_classes = num_classes
        self.classifiers = [QCNN() for _ in range(num_classes)]
        
    def fit(self, X, y):
        """Train one classifier per class using one-vs-all approach."""
        for i in range(self.num_classes):
            # Create binary labels for current class
            binary_y = (y == i).astype(int)
            # Train classifier for current class
            print(f"Training classifier for class {i}")
            self.classifiers[i].fit(X, binary_y)
        return self
    
    def predict(self, X):
        """Predict class by taking argmax of all classifier predictions."""
        # Get predictions from all classifiers
        predictions = []
        for clf in self.classifiers:
            pred = clf.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # Return class with highest prediction value
        return np.argmax(predictions, axis=0)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Main execution code remains the same
if __name__ == "__main__":
    # Load a small dataset for testing
    print("Loading dataset...")

    # load MNIST dataset
    # digits = load_digits()
    # X = digits.data[:100]  # Take only first 100 samples for testing
    # y = digits.target[:100]

    # load GTRSB dataset
    with open("pkls/train_dataset_1k.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("pkls/test_dataset_100.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # 选择10个类别的数据
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 可以选择任意10个类别
    
    # 过滤数据
    train_paths = []
    train_labels = []
    for idx, label in train_data['labels'].items():
        if label in selected_classes:
            train_paths.append(train_data['image_paths'][idx])
            # 重新映射标签到0-9
            new_label = selected_classes.index(label)
            train_labels.append(new_label)
            
    # 转换为numpy数组
    X = np.array([preprocess_image(path) for path in train_paths])
    y = np.array(train_labels)
    
    print("Data loaded. Shape:", X.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Classes present:", np.unique(y))
    
    # Reduce dimensionality to 8 features using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=8)
    X_reduced = pca.fit_transform(X_scaled)
    
    print("Data reduced to 8 features. Shape:", X_reduced.shape)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    
    # Create and train multi-class model
    print("\nCreating multi-class QCNN model...")
    model = MulticlassQCNN(num_classes=10)  # 10 classes for MNIST
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print("\nResults:")
    print(f"Test accuracy: {accuracy:.4f}")
    print("True labels:", y_test)
    print("Predicted labels:", predictions)
