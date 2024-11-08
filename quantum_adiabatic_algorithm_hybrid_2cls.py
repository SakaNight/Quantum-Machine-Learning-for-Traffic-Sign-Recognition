import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile

import os
import pandas as pdtorch
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

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
        # 加载和调整图像大小
        image = Image.open(image_path)
        image = image.resize(size)
        
        # 归一化到[0,1]
        normalized = np.array(image) / 255.0
        
        # 分离RGB通道并展平
        r = normalized[:,:,0].flatten()
        g = normalized[:,:,1].flatten()
        b = normalized[:,:,2].flatten()
        combined = np.concatenate([r, g, b])
        
        # 使用量子绝热编码器处理数据
        encoder = QuantumAdiabaticEncoder(
            n_qubits=12,
            steps=10,  # 可以调整这些参数
            shots=1024
        )
        quantum_features = encoder.encode_data(combined[:12])
        
        return quantum_features
        
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")
    
class QuantumAdiabaticEncoder:
    def __init__(self, n_qubits=12, steps=10, shots=1024):
        """
        基于量子绝热算法的编码器
        Args:
            n_qubits: 量子比特数量
            steps: 绝热演化步数
            shots: 测量次数
        """
        self.n_qubits = n_qubits
        self.steps = steps
        self.shots = shots
        self.simulator = AerSimulator(method='statevector')
        
    def create_adiabatic_circuit(self, data):
        """
        创建量子绝热电路
        按照论文公式(2): H(s) = s(Σ Wij ZiZj + Σ hz_i Zi) + (1-s)Σ hx_i Xi
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # 初始化为|+⟩态
        for i in range(self.n_qubits):
            circuit.h(qr[i])
        
        # 实现时间演化
        for s in range(self.steps):
            s_param = s / self.steps  # 归一化时间参数 s = t/t0
            
            # 1. 实现 Wij ZiZj 项 (最近邻耦合)
            for i in range(self.n_qubits-1):
                circuit.cx(qr[i], qr[i+1])
                circuit.rz(2 * s_param * data[i], qr[i+1])  # Wij参数来自输入数据
                circuit.cx(qr[i], qr[i+1])
            
            # 闭合边界条件
            circuit.cx(qr[self.n_qubits-1], qr[0])
            circuit.rz(2 * s_param * data[self.n_qubits-1], qr[0])
            circuit.cx(qr[self.n_qubits-1], qr[0])
            
            # 2. 实现 hz_i Zi 项 (纵向磁场)
            for i in range(self.n_qubits):
                circuit.rz(s_param * data[i], qr[i])
            
            # 3. 实现 (1-s)Σ hx_i Xi 项 (横向磁场)
            for i in range(self.n_qubits):
                circuit.rx((1-s_param) * np.pi, qr[i])
            
            circuit.barrier()
        
        # 测量
        circuit.measure(qr, cr)
        return circuit

    def encode_data(self, data):
        """
        使用量子绝热算法编码数据
        Returns:
            np.ndarray: 12维的量子特征向量
        """
        # 确保输入数据维度正确
        if len(data) < self.n_qubits:
            raise ValueError(f"Input data must have at least {self.n_qubits} dimensions")
            
        # 标准化数据
        data = data[:self.n_qubits]  # 取前12维
        data = data / np.linalg.norm(data)
        
        # 创建和编译电路
        circuit = self.create_adiabatic_circuit(data)
        compiled_circuit = transpile(circuit, self.simulator, optimization_level=3)
        
        # 运行电路
        job = self.simulator.run(compiled_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        # 处理测量结果
        features = np.zeros(self.n_qubits)
        total_shots = sum(counts.values())
        
        for state, count in counts.items():
            state = state.split()[-1] if ' ' in state else state
            for i, bit in enumerate(state):
                features[i] += int(bit) * count / total_shots
                
        return features

def quantum_encode_data(data: np.ndarray) -> np.ndarray:
    """
    与现有代码兼容的接口函数
    """
    encoder = QuantumAdiabaticEncoder()
    return encoder.encode_data(data)

class HybridNet(nn.Module):
    def __init__(self, num_features=12, num_classes=43):
        super(HybridNet, self).__init__()

        # Classical layers after quantum features
        self.classical_layers = nn.Sequential(
            nn.Linear(num_features, 64),
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.*val_correct/val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.2f}%')

    return model

def main():
    num_epochs = 10
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print("Loading data...")
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")

    print("Processing data...")
    train_features = [preprocess_image(path) for path in tqdm(train_df['Path'])]
    train_labels = train_df['ClassId'].values

    test_features = [preprocess_image(path) for path in tqdm(test_df['Path'])]
    test_labels = test_df['ClassId'].values

    x_train, x_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    num_features = len(x_train[0])  # 根据输入特征的长度设置输入层大小
    model = HybridNet(num_features=num_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training model...")
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device
    )

    # 加载最佳模型权重
    if os.path.exists('best_hybrid_model.pth'):
        model.load_state_dict(torch.load('best_hybrid_model.pth', map_location=device))
        print("Best model weights loaded.")

    # Evaluate on test set
    print("Evaluating on test set...")

    test_dataset = ImageDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100.*test_correct/test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()