import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import os

def preprocess_image(image_path: str, size: tuple = (32, 32)) -> np.ndarray:
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return np.zeros(8)

        image = Image.open(image_path)
        image = image.resize(size)
        normalized = np.array(image) / 255.0
        
        if len(normalized.shape) == 3:
            grayscale = np.mean(normalized, axis=2)
        else:
            grayscale = normalized
            
        h, w = grayscale.shape
        reduced = np.zeros(8)
        pool_size = h // 2
        
        for i in range(8):
            row_start = (i // 4) * pool_size
            row_end = row_start + pool_size
            col_start = (i % 4) * pool_size
            col_end = col_start + pool_size
            region = grayscale[row_start:row_end, col_start:col_end]
            
            if region.size > 0:
                reduced[i] = np.mean(region)
            else:
                reduced[i] = 0.0
                
        if np.any(np.isnan(reduced)):
            print(f"Warning: NaN values detected in {image_path}, replacing with zeros")
            reduced = np.nan_to_num(reduced, 0.0)
            
        return reduced
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(8)

def conv_circuit(params):
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
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    param_index = 0
    
    for i in range(0, num_qubits - 1, 2):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [i, i + 1])
        qc.barrier()
        param_index += 3
    
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    param_index = 0
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    return qc

def build_qcnn_model():
    num_qubits = 8
    
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="QCNN Ansatz")
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
    
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    
    return qnn

def create_classifier(qnn, initial_point=None):
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
        if X.shape[1] != 8:
            raise ValueError(f"Input data must have 8 features, got {X.shape[1]}")
        
        print(f"Training data shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        try:
            self.classifier.fit(X, y)

            train_pred = self.predict(X)
            train_acc = np.mean(train_pred == y)
            print(f"Training accuracy: {train_acc:.4f}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
        return self
    
    def predict(self, X):
        try:
            raw_predictions = self.classifier.predict(X)
            binary_predictions = (raw_predictions > 0.5).astype(int)
            return binary_predictions
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

class MulticlassQCNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classifiers = [QCNN() for _ in range(num_classes)]
        
    def fit(self, X, y):
        self.training_accuracies = []
        
        for i in range(self.num_classes):
            print(f"\nTraining classifier for class {i}")
            binary_y = (y == i).astype(int)
            print(f"Class {i} samples: {np.sum(binary_y)}/{len(binary_y)}")
            
            try:
                self.classifiers[i].fit(X, binary_y)
                train_pred = self.classifiers[i].predict(X)
                acc = np.mean(train_pred == binary_y)
                self.training_accuracies.append(acc)
                print(f"Classifier {i} training accuracy: {acc:.4f}")
            except Exception as e:
                print(f"Error training classifier {i}: {str(e)}")
                self.training_accuracies.append(0.0)
        
        print("\nTraining accuracies for all classifiers:", 
              [f"{acc:.4f}" for acc in self.training_accuracies])
        return self
    
    def predict(self, X):
        predictions = np.zeros((len(self.classifiers), len(X)))
        confidence_scores = np.zeros((len(self.classifiers), len(X)))

        for i, clf in enumerate(self.classifiers):
            try:
                pred = clf.predict(X)
                predictions[i] = pred.flatten()
                
                raw_pred = clf.classifier.predict(X)
                confidence_scores[i] = np.abs(raw_pred.flatten() - 0.5) * 2
            except Exception as e:
                print(f"Error in classifier {i}: {str(e)}")
                predictions[i] = np.zeros(len(X))
                confidence_scores[i] = np.zeros(len(X))
        
        print("\nPrediction confidences:")
        for i in range(self.num_classes):
            print(f"Class {i} mean confidence: {np.mean(confidence_scores[i]):.4f}")
        
        weighted_predictions = predictions * confidence_scores
        final_predictions = np.argmax(weighted_predictions, axis=0)
        
        return final_predictions
    
    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        print("\nDetailed classification report:")
        from sklearn.metrics import classification_report
        print(classification_report(y, predictions))
        
        print("\nConfusion Matrix:")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predictions)
        print(cm)
        
        return accuracy

def create_classifier(qnn, initial_point=None):
    return NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=500),
        callback=lambda weights, obj_func_eval: print(f"Current objective value: {obj_func_eval:.4f}"),
        initial_point=initial_point
    )

if __name__ == "__main__":
    print("Loading dataset...")

    try:
        with open("pkls/train_dataset_43cls_1000.pkl", 'rb') as f:
            train_data = pickle.load(f)
        with open("pkls/test_dataset_43cls_200.pkl", 'rb') as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        exit(1)

    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    train_paths = []
    train_labels = []
    for idx, label in train_data['labels'].items():
        if label in selected_classes:
            train_paths.append(train_data['image_paths'][idx])
            new_label = selected_classes.index(label)
            train_labels.append(new_label)
            
    print("Processing images...")
    X = []
    valid_indices = []
    for i, path in enumerate(train_paths):
        features = preprocess_image(path)
        if not np.any(np.isnan(features)):
            X.append(features)
            valid_indices.append(i)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(train_paths)} images")
    
    X = np.array(X)
    y = np.array([train_labels[i] for i in valid_indices])
    
    print("Data loaded. Shape:", X.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Classes present:", np.unique(y))
    
    print("Performing standardization and PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if np.any(np.isnan(X_scaled)):
        print("Warning: NaN values found after scaling, replacing with zeros")
        X_scaled = np.nan_to_num(X_scaled, 0.0)
    
    pca = PCA(n_components=8)
    X_reduced = pca.fit_transform(X_scaled)
    
    print("Data reduced to 8 features. Shape:", X_reduced.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    
    print("\nCreating multi-class QCNN model...")
    model = MulticlassQCNN(num_classes=10)
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    
    print("\nResults:")
    print(f"Test accuracy: {accuracy:.4f}")
    print("True labels:", y_test)
    print("Predicted labels:", predictions)