import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns

SAVE_PATH = "./quantum_data/"
PLOTS_DIR = os.path.join(SAVE_PATH, 'plots')
EVAL_DIR = os.path.join(SAVE_PATH, 'evaluation_results')
MODEL_DIR = os.path.join(SAVE_PATH, 'models')
for dir_path in [SAVE_PATH, PLOTS_DIR, EVAL_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

n_epochs = 40
n_layers = 1
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def enhanced_circuit(inputs):
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
    RandomLayers(rand_params, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def process_image_blocks(image):
    out = np.zeros((14, 14, 4))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            block = [
                float(image[j, k, 0]),
                float(image[j, k + 1, 0]),    
                float(image[j + 1, k, 0]),
                float(image[j + 1, k + 1, 0])
            ]
            q_results = enhanced_circuit(block)
            for c in range(4):
                out[j // 2, k // 2, c] = float(q_results[c])
    return out

def ClassicalModel():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(43, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def QuantumModel():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(14, 14, 4)),
        keras.layers.MaxPooling2D((2,2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(43, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, x_test, y_test, model_type, timestamp):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, 
        y_pred_classes,
        zero_division=0
    )
    
    full_report = classification_report(
        y_test, 
        y_pred_classes,
        target_names=[str(i) for i in range(43)],
        digits=4,
        zero_division=0,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred_classes)
    cm_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{model_type}_{timestamp}.png')
    plot_confusion_matrix(cm, [str(i) for i in range(43)], cm_path)
    
    accuracies = []
    for i in range(len(precision)):
        mask = y_test == i
        class_accuracy = (y_pred_classes[mask] == i).mean() if mask.any() else 0
        accuracies.append(class_accuracy)
    
    results_df = pd.DataFrame({
        'Class': range(len(precision)),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracies,
        'Support': support
    })
    
    overall_metrics = pd.DataFrame({
        'Class': ['macro_avg', 'weighted_avg'],
        'Precision': [full_report['macro avg']['precision'], full_report['weighted avg']['precision']],
        'Recall': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']],
        'F1': [full_report['macro avg']['f1-score'], full_report['weighted avg']['f1-score']],
        'Accuracy': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']],
        'Support': [sum(support), sum(support)]
    })
    
    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)
    csv_path = os.path.join(EVAL_DIR, f"evaluation_results_{model_type}_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.4f')
    
    return results_df

def main():
    print("Loading data from pickle files...")
    with open("pkls/train_dataset_43cls_0.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("pkls/test_dataset_43cls_0.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print("\nProcessing training images...")
    train_paths = list(train_data['image_paths'].values())
    train_images = np.array([tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    ) / 255.0 for path in tqdm(train_paths, desc="Processing train images")])
    train_labels = np.array(list(train_data['labels'].values()))

    print("\nProcessing test images...")
    test_paths = list(test_data['image_paths'].values())
    test_images = np.array([tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(path, target_size=(28, 28), color_mode='grayscale')
    ) / 255.0 for path in tqdm(test_paths, desc="Processing test images")])
    test_labels = np.array(list(test_data['labels'].values()))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nTraining classical model...")
    classical_model = ClassicalModel()
    classical_history = classical_model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        batch_size=32,
        epochs=n_epochs,
        verbose=2
    )
    classical_model.save(os.path.join(MODEL_DIR, f'classical_model_{timestamp}.h5'))
    classical_results = evaluate_model(classical_model, test_images, test_labels, 'classical', timestamp)

    if PREPROCESS:
        print("\nProcessing quantum data...")
        q_train_images = np.array([process_image_blocks(img) for img in tqdm(train_images, desc="Quantum processing train images")])
        q_test_images = np.array([process_image_blocks(img) for img in tqdm(test_images, desc="Quantum processing test images")])
        
        np.savez(os.path.join(SAVE_PATH, 'quantum_features.npz'),
                 train_images=q_train_images,
                 test_images=q_test_images,
                 train_labels=train_labels,
                 test_labels=test_labels)

        print("\nTraining quantum model...")
        quantum_model = QuantumModel()
        quantum_history = quantum_model.fit(
            q_train_images, train_labels,
            validation_data=(q_test_images, test_labels),
            batch_size=32,
            epochs=n_epochs,
            verbose=2
        )
        quantum_model.save(os.path.join(MODEL_DIR, f'quantum_model_{timestamp}.h5'))
        quantum_results = evaluate_model(quantum_model, q_test_images, test_labels, 'quantum', timestamp)

        plt.style.use("default")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

        ax1.plot(quantum_history.history["val_accuracy"], "-ob", label="With quantum layer")
        ax1.plot(classical_history.history["val_accuracy"], "-og", label="Without quantum layer")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(quantum_history.history["val_loss"], "-ob", label="With quantum layer")
        ax2.plot(classical_history.history["val_loss"], "-og", label="Without quantum layer")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(top=2.5)
        ax2.set_xlabel("Epoch")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, 'comparison_results.png'))
        plt.close()

if __name__ == "__main__":
    main()