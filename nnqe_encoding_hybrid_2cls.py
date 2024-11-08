import os
import numpy as np
import pandas as pd  # 修复导入
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def quantum_encode_data(data: np.ndarray) -> np.ndarray:
    """使用NNQE风格的量子编码"""
    num_qubits = 12
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # 数据编码
    for i in range(num_qubits):
        theta = np.pi * data[i]
        circuit.ry(theta, qr[i])
    
    # Hadamard层
    for i in range(num_qubits):
        circuit.h(qr[i])
    
    circuit.barrier()
    circuit.measure(qr, cr)
    
    # 运行电路
    backend = AerSimulator(method='statevector')
    compiled_circuit = transpile(circuit, backend, optimization_level=3)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    
    # 处理测量结果
    features = np.zeros(num_qubits)
    total_shots = sum(counts.values())
    for state, count in counts.items():
        state = state.split()[-1] if ' ' in state else state
        for i, bit in enumerate(state):
            features[i] += int(bit) * count / total_shots
            
    return features

def preprocess_image(image_path: str, size: tuple = (32, 32)) -> np.ndarray:
    """图像预处理和量子编码"""
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        normalized = np.array(image) / 255.0
        
        # 转换为灰度
        if len(normalized.shape) == 3:
            grayscale = np.mean(normalized, axis=2)
        else:
            grayscale = normalized
            
        # 取样12个点用于量子编码
        flattened = grayscale.flatten()
        step = len(flattened) // 12
        input_data = flattened[::step][:12]
        
        # 确保有12个点
        if len(input_data) < 12:
            input_data = np.pad(input_data, (0, 12 - len(input_data)))
        
        # 使用量子编码
        quantum_features = quantum_encode_data(input_data)
        return quantum_features
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(12)  # 返回零向量而不是抛出异常

class NNQECNN(nn.Module):
    def __init__(self, num_quantum_features=12, num_classes=43):
        super(NNQECNN, self).__init__()
        
        # 特征扩展层
        self.feature_expansion = nn.Sequential(
            nn.Linear(num_quantum_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        self.feature_size = 16  # 256 = 16*16
        
        # CNN部分
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        
        # 计算卷积层输出大小
        self.conv_output_size = 128 * 2 * 2
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 特征扩展
        x = self.feature_expansion(x)
        
        # 重构为图像格式
        x = x.view(-1, 1, self.feature_size, self.feature_size)
        
        # CNN特征提取
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        return x

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
            torch.save(model.state_dict(), 'best_nnqe_model.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.2f}%')

    return model

def main():
    print("Starting program...")  # 添加调试信息
    
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # 添加调试信息

    print("Loading data...")
    try:
        train_df = pd.read_csv("Train.csv")
        test_df = pd.read_csv("Test.csv")
    except Exception as e:
        print(f"Error loading CSV files: {str(e)}")
        return

    print("Processing data...")
    train_features = [preprocess_image(path) for path in tqdm(train_df['Path'])]
    train_labels = train_df['ClassId'].values

    test_features = [preprocess_image(path) for path in tqdm(test_df['Path'])]
    test_labels = test_df['ClassId'].values

    print(f"Train features shape: {len(train_features)}, {len(train_features[0])}")  # 添加调试信息

    x_train, x_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = NNQECNN(num_quantum_features=12, num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training model...")
    try:
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=num_epochs, device=device
        )
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

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
