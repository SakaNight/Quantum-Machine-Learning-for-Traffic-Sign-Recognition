# 标准库导入
import os
import pickle
from datetime import datetime

# 第三方库导入
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Qiskit相关导入
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# 本地模块导入
from tools import train_model, plot_confusion_matrix


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def preprocess_image(image_path: str, size: tuple = (32, 32)) -> np.ndarray:
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        normalized = np.array(image) / 255.0
        r = normalized[:,:,0].flatten()
        g = normalized[:,:,1].flatten()
        b = normalized[:,:,2].flatten()
        combined = np.concatenate([r, g, b])
        norm = np.sqrt(np.sum(combined**2))
        if norm > 0:
            normalized_amplitude = combined / norm
        else:
            normalized_amplitude = combined
        return normalized_amplitude
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")


def quantum_encoding(data: np.ndarray) -> np.ndarray:
    """Encode classical data into quantum state and apply quantum convolution"""
    num_qubits = 12

    # Create quantum circuit with quantum and classical registers
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    qc = QuantumCircuit(qr, cr)  # 添加经典寄存器

    # Apply initial Hadamard gates
    qc.h(range(num_qubits))

    # Encode amplitudes
    for i in range(min(len(data), 2 ** num_qubits)):
        if abs(data[i]) > 1e-10:
            binary = format(i, f'0{num_qubits}b')
            for j, bit in enumerate(binary):
                if bit == '1':
                    qc.x(j)
            angle = 2 * np.arccos(np.sqrt(abs(data[i]))) if data[i] > 0 else 0
            if num_qubits > 1:
                qc.mcry(theta=angle,
                        q_controls=[qr[j] for j in range(num_qubits - 1)],
                        q_target=qr[-1])
            else:
                qc.ry(angle, 0)
            for j, bit in enumerate(binary):
                if bit == '1':
                    qc.x(j)

    # Add quantum convolution layer
    qc.barrier()
    for i in range(0, num_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.rz(np.pi / 4, i + 1)
        qc.cx(i, i + 1)

    # 添加测量
    qc.measure(qr, cr)  # 添加测量操作

    # Simulate circuit
    backend = AerSimulator()
    compiled_circuit = transpile(qc, backend)
    job = backend.run(compiled_circuit, shots=512)
    result = job.result()

    # Get counts and convert to features
    counts = result.get_counts()

    # Convert counts to normalized features
    features = np.zeros(num_qubits)
    total_shots = sum(counts.values())
    for state, count in counts.items():
        for i, bit in enumerate(state):  # 移除了split()[0]
            features[i] += int(bit) * count / total_shots

    return features


class HybridNet(nn.Module):
    def __init__(self, input_size, num_classes=43):
        super(HybridNet, self).__init__()

        # Classical layers after quantum features
        self.classical_layers = nn.Sequential(
            nn.Linear(12, 64),  # 12 is num_qubits from quantum encoding
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classical_layers(x)

    
def main():
    num_epochs = 80
    num_classes = 43  # todo check before run
    device = torch.device('cpu')
    trainset = "pkls/train_dataset_43cls_10000.pkl"
    testset = "pkls/test_dataset_43cls_1000.pkl"

    # 添加预训练模型相关参数
    pretrained_path = "" # 设置预训练模型路径
    resume_training = False  # 是否继续训练

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('runs', 'early_encoding', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 创建子目录
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    last_model_path = os.path.join(checkpoints_dir, "last.pth")

    # [数据加载部分保持不变...]
    print("Loading data from pickle files...")
    with open(trainset, 'rb') as f:
        train_data = pickle.load(f)

    with open(testset, 'rb') as f:
        test_data = pickle.load(f)

    # processing data
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    # 创建特征缓存目录
    FEATURES_CACHE_DIR = os.path.join('pkls', 'quantum_features')
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)

    # 生成一个唯一的特征文件名（基于数据集大小和类别数）
    feature_cache_path = os.path.join(FEATURES_CACHE_DIR, "early_{}".format(trainset.split('/')[-1]))
    test_feature_cache_path = os.path.join(FEATURES_CACHE_DIR, "early_{}".format(testset.split('/')[-1]))

    # 检查是否存在缓存的特征
    if os.path.exists(feature_cache_path) and os.path.exists(test_feature_cache_path):
        print("Loading cached quantum features...")
        with open(feature_cache_path, 'rb') as f:
            train_features = pickle.load(f)
        with open(test_feature_cache_path, 'rb') as f:
            test_features = pickle.load(f)
    else:
        print("Encoding training data...")
        train_features = []
        for path in tqdm(train_paths):
            data = preprocess_image(path)
            encoded_data = quantum_encoding(data)
            train_features.append(encoded_data)
        train_features = np.array(train_features)

        print("Encoding test data...")
        test_features = []
        for path in tqdm(test_paths):
            data = preprocess_image(path)
            encoded_data = quantum_encoding(data)
        test_features = np.array(encoded_data)

        # 保存特征到缓存
        print("Saving quantum features to cache...")
        with open(feature_cache_path, 'wb') as f:
            pickle.dump(train_features, f)
        with open(test_feature_cache_path, 'wb') as f:
            pickle.dump(test_features, f)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Create data loaders
    train_dataset = ImageDataset(train_features, train_labels)
    test_dataset = ImageDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Initializing model...")
    num_features = train_features.shape[1]
    model = HybridNet(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    model = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs, device, save_dir,
        pretrained_path=pretrained_path,  # 添加预训练模型路径
        resume_training=resume_training  # 添加是否继续训练的标志
    )

    print("Evaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.*test_correct/test_total
    print(f'\nTest Accuracy: {test_acc:.2f}%')

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
    }, last_model_path)
    print(f'Final model saved with test accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()