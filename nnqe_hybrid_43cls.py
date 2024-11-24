import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from training_tools import DataLoader, TrainingTools, EvaluationTools, ModelLoader
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.image_paths = list(data['image_paths'].values())
        self.labels = list(data['labels'].values())
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return torch.FloatTensor(preprocess_image(self.image_paths[idx])), torch.tensor(self.labels[idx], dtype=torch.long)

def quantum_encode_data(data: np.ndarray) -> np.ndarray:
    """改进的NNQE风格量子编码"""
    num_qubits = 12
    qr = QuantumRegister(num_qubits, 'q')
    cr = ClassicalRegister(num_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # 数据编码 - 使用输入数据的实际值而不是随机值
    for i in range(num_qubits):
        # 将输入数据映射到[0, 2π]范围
        theta = 2 * np.pi * data[i]
        circuit.ry(theta, qr[i])
    
    # 添加固定的纠缠层
    for _ in range(3):  # 使用3层纠缠
        # 使用CZ门创建纠缠
        for i in range(num_qubits-1):
            circuit.cz(qr[i], qr[i+1])
        # 添加单量子比特旋转
        for i in range(num_qubits):
            circuit.ry(np.pi/4, qr[i])  # 固定角度的旋转
    
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
    """改进的图像预处理"""
    try:
        image = Image.open(image_path)
        image = image.resize(size)
        normalized = np.array(image) / 255.0
        
        # 转换为灰度
        if len(normalized.shape) == 3:
            grayscale = np.mean(normalized, axis=2)
        else:
            grayscale = normalized
            
        # 使用更智能的降维方法
        h, w = grayscale.shape
        block_h, block_w = h // 3, w // 4  # 将图像分成12个区块
        
        input_data = []
        for i in range(3):
            for j in range(4):
                # 计算每个区块的平均值
                block = grayscale[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                input_data.append(np.mean(block))
        
        input_data = np.array(input_data)
        
        # 使用量子编码
        quantum_features = quantum_encode_data(input_data)
        return quantum_features
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(12)

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

def main():
    print("Starting program...")
    
    # 配置参数
    num_epochs = 2
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_name = "NNQE_hybrid_43cls"
    
    # 创建必要的目录
    save_dir = "model_zoos"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化工具类
    model_loader = ModelLoader(checkpoint_dir=os.path.join(save_dir, model_name))
    trainer = TrainingTools(
        model_save_dir=os.path.join(save_dir, model_name),
        tensorboard_dir=f'runs/{model_name}'
    )
    evaluator = EvaluationTools(
        results_dir=os.path.join(save_dir, f'{model_name}_results')
    )

    print("Loading data...")
    try:
        with open(os.path.join("pkls", "train_dataset_1k.pkl"), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join("pkls", "test_dataset_100.pkl"), 'rb') as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    print("Creating datasets...")
    train_dataset = ImageDataset(data=train_data)
    val_dataset = ImageDataset(data=train_data)
    test_dataset = ImageDataset(data=test_data)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Initializing model...")
    model = NNQECNN(num_quantum_features=12, num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 检查是否有检查点
    checkpoint_path = os.path.join(save_dir, f'{model_name}/{model_name}_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print("Found existing checkpoint. Attempting to resume training...")
        model, optimizer, start_epoch, best_val_acc, history = model_loader.load_model_and_state(
            model=model,
            model_path=checkpoint_path,
            optimizer=optimizer,
            device=device
        )
    else:
        start_epoch = 0
        history = None
        print("Starting fresh training...")

    print("Training model...")
    model, history = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name=model_name,
        start_epoch=start_epoch
    )

    print("Loading best model for evaluation...")
    model = model_loader.load_for_inference(
        model=model,
        weights_path=os.path.join(save_dir, f'{model_name}/{model_name}_best.pth'),
        device=device
    )
    
    if model is not None:
        print("Starting model evaluation...")
        results, avg_loss, accuracy = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            class_names=[str(i) for i in range(43)]
        )
        
        print(f"\nFinal Test Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Failed to load best model for evaluation")

if __name__ == "__main__":
    main()