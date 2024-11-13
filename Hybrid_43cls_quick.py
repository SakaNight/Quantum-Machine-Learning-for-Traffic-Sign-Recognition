import os
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

import matplotlib.pyplot as plt
import seaborn as sns


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
        r = normalized[:, :, 0].flatten()
        g = normalized[:, :, 1].flatten()
        b = normalized[:, :, 2].flatten()
        combined = np.concatenate([r, g, b])
        norm = np.sqrt(np.sum(combined ** 2))
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                device='cpu', start_epoch=0, save_model="best_model.pth"):
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

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
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

            pbar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})

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

        val_acc = 100. * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 修复：使用不同的文件名保存模型权重和准确率
            torch.save(model.state_dict(), save_model)  # 保存模型权重
            torch.save(best_val_acc, f'{os.path.splitext(save_model)[0]}_acc.pth')  # 保存准确率到不同文件
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

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, test_loader, device, class_names=None):
    """
    评估模型性能并生成CSV格式的结果
    """
    import pandas as pd
    from datetime import datetime

    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    # 获取预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算总体损失
    avg_loss = total_loss / test_total

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算各个类别的precision, recall, f1-score
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds)

    # 获取完整的分类报告
    full_report = classification_report(all_labels, all_preds,
                                        target_names=class_names if class_names else [str(i) for i in
                                                                                      range(len(precision))],
                                        digits=4, output_dict=True)

    # 计算每个类别的准确率
    accuracies = []
    for i in range(len(precision)):
        mask = all_labels == i
        class_accuracy = (all_preds[mask] == i).mean() if mask.any() else 0
        accuracies.append(class_accuracy)

    # 创建DataFrame
    results_df = pd.DataFrame({
        'Class': range(len(precision)),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracies
    })

    # 计算总体指标
    overall_metrics = pd.DataFrame({
        'Class': ['macro_avg', 'weighted_avg'],
        'Precision': [full_report['macro avg']['precision'], full_report['weighted avg']['precision']],
        'Recall': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']],
        'F1': [full_report['macro avg']['f1-score'], full_report['weighted avg']['f1-score']],
        'Accuracy': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']]  # accuracy 使用 recall
    })

    # 合并结果
    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)

    # 保存结果
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.csv")

    # 保存为CSV，设置保留4位小数
    results_df.to_csv(csv_path, index=False, float_format='%.4f')

    # 同时打印到控制台
    print("\nEvaluation Results:")
    print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))

    # 打印总体损失
    print(f"\nTotal Loss: {avg_loss:.4f}")

    return results_df, avg_loss

def load_model_weights(model, model_path, device):
    """
    加载模型权重，包含错误处理和多种格式支持
    """
    try:
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # 检查加载的内容类型
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # checkpoint格式
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # 另一种常见的保存格式
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接的状态字典
                model.load_state_dict(checkpoint)
        elif isinstance(checkpoint, torch.nn.Module):
            # 整个模型被保存
            model.load_state_dict(checkpoint.state_dict())
        else:
            raise TypeError(f"Unexpected checkpoint format. Got type: {type(checkpoint)}")

        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Checkpoint content type: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print("Available keys in checkpoint:", checkpoint.keys())
        return False

def main():
    batch_size = 32
    num_epochs = 10
    device = torch.device('cpu')
    
    # 创建保存目录
    save_dir = "model_zoos"
    os.makedirs(save_dir, exist_ok=True)
    save_model = os.path.join(save_dir, "Hybrid_43cls.pth")

    # 创建评估结果目录
    os.makedirs("evaluation_results", exist_ok=True)
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)

    print("Loading data from pickle files...")
    # 加载训练数据
    with open("pkls/train_dataset.pkl", 'rb') as f:
        train_data = pickle.load(f)

    # 加载测试数据
    with open("pkls/test_dataset.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print("Processing training data...")
    # 转换字典格式的图片路径和标签为列表
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())

    print("Processing test data...")
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    # 处理图像
    train_features = [preprocess_image(path) for path in tqdm(train_paths)]
    test_features = [preprocess_image(path) for path in tqdm(test_paths)]

    # 转换为numpy数组以加快处理速度
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    num_features = train_features.shape[1]  # 使用shape获取特征数量
    model = HybridNet(num_features=num_features, num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载预训练模型（如果存在）
    checkpoint_path = os.path.join("checkpoints", "checkpoint.pth")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("Checkpoint file does not contain expected keys. Starting training from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting training from scratch.")
            start_epoch = 0

    # Train model
    print("Training model...")
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, device=device, start_epoch=start_epoch, save_model=save_model
    )

    # 创建测试数据加载器
    test_dataset = ImageDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    print("Initializing model...")
    model = HybridNet(num_features=num_features, num_classes=43).to(device)

    # 加载模型权重
    if not os.path.exists(save_model):
        print(f"Model file not found: {save_model}")
        return

    # 尝试加载模型
    if not load_model_weights(model, save_model, device):
        print("Failed to load model weights. Exiting...")
        return

    # 定义类别名称
    class_names = [str(i) for i in range(43)]

    # 评估模型
    print("Starting evaluation...")
    try:
        evaluate_model(model, test_loader, device, class_names)
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return
    
if __name__ == "__main__":
    main()