# /opt/anaconda3/bin/python nnqe.py
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

class NNQEEncoder:
    def __init__(self, n_qubits=4, layers=5, shots=1024):
        """论文中的NNQE编码器"""
        self.n_qubits = n_qubits
        self.layers = layers
        self.shots = shots
        self.simulator = AerSimulator(method='statevector')
        
    def create_nnqe_circuit(self, data):
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # 数据编码
        for i in range(self.n_qubits):
            theta = np.pi * data[i]
            circuit.ry(theta, qr[i])
        
        # Hadamard层
        for i in range(self.n_qubits):
            circuit.h(qr[i])
        
        # 5层强纠缠
        for layer in range(self.layers):
            # 三轴旋转
            for i in range(self.n_qubits):
                theta_x = np.random.uniform(0, 2*np.pi)
                theta_y = np.random.uniform(0, 2*np.pi)
                theta_z = np.random.uniform(0, 2*np.pi)
                
                circuit.rx(theta_x, qr[i])
                circuit.ry(theta_y, qr[i])
                circuit.rz(theta_z, qr[i])
            
            # CNOT层
            for i in range(0, self.n_qubits-1, 2):
                circuit.cx(qr[i], qr[i+1])
            for i in range(1, self.n_qubits-1, 2):
                circuit.cx(qr[i], qr[i+1])
            
            circuit.barrier()
        
        circuit.measure(qr, cr)
        return circuit

    def encode_patch(self, patch_data):
        if len(patch_data) != self.n_qubits:
            raise ValueError(f"Patch data must have exactly {self.n_qubits} values")
        
        circuit = self.create_nnqe_circuit(patch_data)
        compiled_circuit = transpile(circuit, self.simulator, optimization_level=3)
        
        job = self.simulator.run(compiled_circuit, shots=self.shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        
        features = np.zeros(self.n_qubits)
        total_shots = sum(counts.values())
        
        for state, count in counts.items():
            state = state.split()[-1] if ' ' in state else state
            for i, bit in enumerate(state):
                features[i] += int(bit) * count / total_shots
                
        return features

class GTSRBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class OriginalNNQEModel(nn.Module):
    def __init__(self, num_classes=43):
        super(OriginalNNQEModel, self).__init__()
        
        # 第一部分：特征提取CNN
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        
        # NNQE编码器
        self.nnqe = NNQEEncoder(n_qubits=4, layers=5)
        
        # 最终分类CNN
        self.classifier = nn.Sequential(
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            # 全连接层
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def process_features_with_nnqe(self, x):
        """使用NNQE处理特征图"""
        batch_size, channels, height, width = x.shape
        processed_features = []
        
        # 对每个batch处理
        for b in range(batch_size):
            feature_map = x[b].cpu().numpy()
            quantum_features = []
            
            # 使用2x2滑动窗口
            for i in range(0, height-1, 2):
                for j in range(0, width-1, 2):
                    # 从所有通道提取2x2 patch
                    patch = feature_map[:, i:i+2, j:j+2].reshape(-1)[:4]  # 取前4个值
                    # 量子编码
                    quantum_feature = self.nnqe.encode_patch(patch)
                    quantum_features.extend(quantum_feature)
            
            processed_features.append(quantum_features)
        
        # 转换回适合CNN的格式
        processed_features = torch.tensor(processed_features, device=x.device)
        new_size = int(np.sqrt(len(quantum_features)))
        processed_features = processed_features.view(batch_size, 64, new_size, new_size)
        
        return processed_features

    def forward(self, x):
        # 特征提取
        x = self.features(x)
        
        # NNQE处理
        x = self.process_features_with_nnqe(x)
        
        # 分类
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
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
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100.*correct/total})
        
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

def main():
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    print("Loading data...")
    train_df = pd.read_csv("Train.csv")
    test_df = pd.read_csv("Test.csv")
    
    # 创建数据集
    train_dataset = GTSRBDataset(train_df['Path'].values, train_df['ClassId'].values, transform)
    test_dataset = GTSRBDataset(test_df['Path'].values, test_df['ClassId'].values, transform)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = OriginalNNQEModel(num_classes=43).to(device)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("Training model...")
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
    # 测试模型
    print("Testing model...")
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