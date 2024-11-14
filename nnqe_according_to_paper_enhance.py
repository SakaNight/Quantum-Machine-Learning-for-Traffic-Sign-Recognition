import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import pickle

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data):
        self.image_paths = list(data['image_paths'].values())
        self.labels = list(data['labels'].values())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = Image.open(self.image_paths[idx])
        image = image.resize((32, 32))
        image = image.convert('L')  # 转灰度
        image = np.array(image) / 255.0
        
        # 使用滑动窗口提取特征
        features = []
        for i in range(0, 30, 2):  # 32-2+1 patches
            for j in range(0, 30, 2):
                # 提取2x2 patch
                patch = image[i:i+2, j:j+2].flatten()
                # 量子编码
                quantum_features = nnqe_quantum_encode(patch)
                features.extend(quantum_features)
                
        return torch.FloatTensor(features), torch.tensor(self.labels[idx], dtype=torch.long)

def nnqe_quantum_encode(data: np.ndarray) -> np.ndarray:
    """简化的NNQE量子编码：使用2层强纠缠而不是5层"""
    num_qubits = 4  # 使用4个量子比特处理2x2窗口
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # 1. RY编码 - 保持不变，这是核心编码步骤
    for i in range(num_qubits):
        theta = np.pi * data[i]
        circuit.ry(theta, qr[i])
    
    # 2. Hadamard层 - 保持不变，创建必要的叠加态
    for i in range(num_qubits):
        circuit.h(qr[i])
    
    # 3. 强纠缠层 - 减少到2层(原来是5层)
    for _ in range(2):
        # 三轴旋转 - 每个量子比特
        for i in range(num_qubits):
            # 使用固定的旋转角度而不是随机角度
            circuit.rx(np.pi/2, qr[i])  # 固定的X旋转
            circuit.ry(np.pi/2, qr[i])  # 固定的Y旋转
            circuit.rz(np.pi/2, qr[i])  # 固定的Z旋转
        
        # CNOT连接 - 保持交替模式
        for i in range(0, num_qubits-1, 2):
            circuit.cx(qr[i], qr[i+1])
        for i in range(1, num_qubits-1, 2):
            circuit.cx(qr[i], qr[i+1])
        
        circuit.barrier()
    
    # 4. 测量
    circuit.measure(qr, cr)
    
    # 5. 运行电路
    backend = AerSimulator(method='statevector')
    compiled_circuit = transpile(circuit, backend, optimization_level=3)
    job = backend.run(compiled_circuit, shots=512)  # 减少shots数量
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    
    # 处理测量结果
    features = np.zeros(num_qubits)
    total_shots = sum(counts.values())
    for state, count in counts.items():
        for i, bit in enumerate(reversed(state)):
            features[i] += int(bit) * count / total_shots
            
    return features

def preprocess_image(image_path: str, size: tuple = (32, 32)) -> np.ndarray:
    """图像预处理为2x2块并进行量子编码"""
    try:
        # 加载和预处理图像
        image = Image.open(image_path)
        image = image.resize(size)
        image = image.convert('L')  # 转为灰度
        image = np.array(image) / 255.0
        
        features = []
        # 使用步长为2的2x2滑动窗口
        for i in range(0, size[0]-1, 2):
            for j in range(0, size[1]-1, 2):
                # 提取2x2 patch
                patch = image[i:i+2, j:j+2].flatten()
                # 量子编码
                quantum_features = nnqe_quantum_encode(patch)
                features.extend(quantum_features)
                
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(900)  # 15*15*4=900维特征向量

class NNQECNN(nn.Module):
    def __init__(self, num_classes=43):
        super(NNQECNN, self).__init__()
        
        # 计算输入维度：15x15个窗口，每个窗口4个特征
        self.input_size = 15 * 15 * 4  # 900
        
        # 使用简单的分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=2, device='cpu'):
    # best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)

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

    return model

def main():
    print("Starting program...")
    
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    try:
        with open(os.path.join("pkls", "train_dataset_1k.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        # with open(os.path.join("pkls", "val_dataset.pkl"), 'rb') as f:
        #     val_data = pickle.load(f)
        with open(os.path.join("pkls", "test_dataset_100.pkl"), 'rb') as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # 创建 ImageDataset 实例时,使用 train_data, val_data 和 test_data 中的数据
    train_dataset = ImageDataset(data=train_data)
    # val_dataset = ImageDataset(data=val_data)
    test_dataset = ImageDataset(data=test_data)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Initializing model...")
    model = NNQECNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    try:
        model = train_model(
            model, train_loader, criterion, optimizer,
            num_epochs=num_epochs, device=device
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

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
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()
