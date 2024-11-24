import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
import pickle
from datetime import datetime

from PIL import Image
from sklearn.model_selection import train_test_split
import torch

from multiprocessing import Pool

from tools import train_model


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
    job = backend.run(compiled_circuit, shots=256)  # 减少shots数量,512到256
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
        # 使用步长为2的2x2滑动窗口，确保覆盖整个32x32图像
        for i in range(0, size[0]-1, 2):  # 改回 size[0]-1 避免边界溢出
            for j in range(0, size[1]-1, 2):
                # 提取2x2 patch
                patch = image[i:i+2, j:j+2].flatten()
                # 量子编码
                quantum_features = nnqe_quantum_encode(patch)
                features.extend(quantum_features)

        features = np.array(features)
        if len(features) != 1024:  # 16x16x4 = 1024
            # 如果特征数量不对，用零填充到正确的大小
            padded_features = np.zeros(1024)
            padded_features[:len(features)] = features
            features = padded_features

        return features

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(1024)  # 16x16x4=1024维特征向量

def preprocess_images_parallel(image_paths, num_workers=2):
    with Pool(num_workers) as pool:
        # imap_unordered 保持进度条显示
        features = list(tqdm(
            pool.imap_unordered(preprocess_image, image_paths),
            total=len(image_paths),
            desc="Processing images"
        ))
    return features

class NNQECNN(nn.Module):
    def __init__(self, num_classes):
        super(NNQECNN, self).__init__()
        
        # 计算输入维度：16x16个窗口，每个窗口4个特征
        self.input_size = 16 * 16 * 4  # 1024
        
        # 使用简单的分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.shape[1] != self.input_size:
            print(f"Warning: Input shape {x.shape} does not match expected shape (batch_size, {self.input_size})")
        return self.classifier(x)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    num_epochs = 80
    num_classes = 2  # todo check before run
    device = torch.device('cpu')

    # 添加预训练模型相关参数
    pretrained_path = "" # "runs/nnqe/20241123_10cls_1k/best_model.pth"  # 设置预训练模型路径
    resume_training = False  # 是否继续训练

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('runs', 'nnqe', timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 创建子目录
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    last_model_path = os.path.join(checkpoints_dir, "last.pth")

    # [数据加载部分保持不变...]
    print("Loading data from pickle files...")
    with open("pkls/train_dataset_2cls_debug.pkl", 'rb') as f:  # todo check before run
        train_data = pickle.load(f)

    with open("pkls/test_dataset_2cls_debug.pkl", 'rb') as f:  # todo check before run
        test_data = pickle.load(f)

    # processing data
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    print("Encoding training data...")
    train_features = []
    for path in tqdm(train_paths):
        classical_data = preprocess_image(path)
        train_features.append(classical_data)

    train_features = np.array(train_features)

    print("Encoding test data...")
    test_features = []
    for path in tqdm(test_paths):
        classical_data = preprocess_image(path)
        test_features.append(classical_data)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Create data loaders
    train_dataset = ImageDataset(train_features, train_labels)
    test_dataset = ImageDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Initializing model...")
    model = NNQECNN(num_classes).to(device)
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
    print(f'Test Accuracy: {test_acc:.2f}%')

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
