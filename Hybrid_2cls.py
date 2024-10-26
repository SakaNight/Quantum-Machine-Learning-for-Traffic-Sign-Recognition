import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from qiskit import QuantumCircuit
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
    """使用参数化电路编码经典数据并应用量子层"""
    num_qubits = len(data)  # 根据数据维度调整量子比特数量

    # 创建量子电路
    qr = QuantumCircuit(num_qubits)

    # 使用参数化的 RY 门编码数据
    for i in range(num_qubits):
        qr.ry(data[i] * np.pi, i)

    # 添加纠缠层（可选）
    for i in range(num_qubits - 1):
        qr.cz(i, i + 1)

    # 测量量子比特
    qr.measure_all()

    # 模拟电路
    backend = AerSimulator()
    compiled_circuit = transpile(qr, backend)
    job = backend.run(compiled_circuit, shots=1024)
    result = job.result()

    # 获取测量结果并转换为特征向量
    counts = result.get_counts()
    features = np.zeros(num_qubits)
    total_shots = sum(counts.values())
    for state, count in counts.items():
        for i, bit in enumerate(reversed(state)):  # 需要反转状态字符串
            features[i] += int(bit) * count / total_shots

    return features

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', start_epoch=0):
    best_val_acc = 0.0

    # 如果存在之前的训练状态，加载最佳验证准确率
    if os.path.exists('best_val_acc.pth'):
        best_val_acc = torch.load('best_val_acc.pth')

    for epoch in range(start_epoch, num_epochs):
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
            # 保存最佳模型权重和最佳验证准确率
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(best_val_acc, 'best_val_acc.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.2f}%')

        # 保存当前的模型和优化器状态
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, 'checkpoint.pth')

    return model

def main():
    num_epochs = 10
    # Device configuration
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print("Loading data...")
    train_df = pd.read_csv("Filtered_ClassIds_12.csv")
    test_df = pd.read_csv("Filtered_ClassIds_12_Test.csv")

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

    # 加载预训练模型（如果存在）
    start_epoch = 0
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth', map_location=device)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Checkpoint file does not contain expected keys. Starting training from scratch.")
            start_epoch = 0

    # Train model
    print("Training model...")
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device, start_epoch=start_epoch
    )

    # 加载最佳模型权重
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
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
