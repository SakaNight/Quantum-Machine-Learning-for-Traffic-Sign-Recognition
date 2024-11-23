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
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes


from sklearn.decomposition import PCA
import torch.nn.functional as F


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# todo: change here
def preprocess_image(image_path: str, label: int, size: tuple = (32, 32), aug_prob: float = 0.5, is_training: bool = False) -> np.ndarray:
    try:
        image = Image.open(image_path)
        
        # 只在训练模式下考虑数据增强
        if is_training and np.random.random() < aug_prob:
            # 随机选择一个或多个增强方法
            augmentations = [
                ('rotate', np.random.random() < 0.3),
                ('flip', np.random.random() < 0.3),
                ('brightness', np.random.random() < 0.3),
                ('contrast', np.random.random() < 0.3),
                ('noise', np.random.random() < 0.3)
            ]
            
            # 应用选中的增强方法
            for aug_type, apply in augmentations:
                if apply:
                    if aug_type == 'rotate':
                        # 随机旋转 -15 到 15 度
                        angle = np.random.uniform(-15, 15)
                        image = image.rotate(angle, expand=False, resample=Image.BILINEAR)
                    
                    elif aug_type == 'flip' and label not in [33,34,36,37,38,39]:
                        # 随机水平翻转
                        if np.random.random() < 0.5:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    
                    elif aug_type == 'brightness':
                        # 随机调整亮度
                        from PIL import ImageEnhance
                        factor = np.random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Brightness(image)
                        image = enhancer.enhance(factor)
                    
                    elif aug_type == 'contrast':
                        # 随机调整对比度
                        from PIL import ImageEnhance
                        factor = np.random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(factor)
                    
                    elif aug_type == 'noise':
                        # 添加随机噪声
                        img_array = np.array(image)
                        noise = np.random.normal(0, 5, img_array.shape)
                        noisy_img = img_array + noise
                        noisy_img = np.clip(noisy_img, 0, 255)
                        image = Image.fromarray(noisy_img.astype(np.uint8))

        # 调整图像大小
        image = image.resize(size)
        
        # 标准化处理
        normalized = np.array(image) / 255.0
        
        # 分离和展平 RGB 通道
        r = normalized[:, :, 0].flatten()
        g = normalized[:, :, 1].flatten()
        b = normalized[:, :, 2].flatten()

        # 合并通道
        combined = np.concatenate([r, g, b])

        # L2 归一化
        norm = np.sqrt(np.sum(combined ** 2))
        if norm > 0:
            normalized_amplitude = combined / norm
        else:
            normalized_amplitude = combined

        return normalized_amplitude

    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")


class ZFeatureMapsEncoder:
    def __init__(self, num_qubits, feature_map_depth):
        self.feature_map = ZFeatureMap(num_qubits, reps=feature_map_depth)

    def __call__(self, data):
        num_samples = data.shape[0]
        encoded_data = np.zeros((num_samples, self.feature_map.num_parameters))
        for i in range(num_samples):
            encoded_data[i] = self.feature_map.bind_parameters(data[i]).parameters
        return encoded_data

class RealAmplitudesEncoder:
    def __init__(self, num_qubits, reps=1):
        self.real_amplitudes = RealAmplitudes(num_qubits, reps=reps)

    def __call__(self, data):
        num_samples = data.shape[0]
        encoded_data = np.zeros((num_samples, self.real_amplitudes.num_parameters))
        for i in range(num_samples):
            encoded_data[i] = self.real_amplitudes.bind_parameters(data[i]).parameters
        return encoded_data


def parameterized_quantum_encoding(data: torch.Tensor) -> torch.Tensor:
    """使用参数化电路编码经典数据并应用量子层"""
    batch_size, num_features = data.shape
    num_qubits = num_features  # 根据数据维度调整量子比特数量

    # 创建量子电路
    qr = QuantumCircuit(num_qubits)

    # 使用参数化的 RY 门编码数据
    for sample_idx in range(batch_size):
        for i in range(num_qubits):
            qr.ry(data[sample_idx, i].numpy() * np.pi, i)

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
    features = torch.zeros(batch_size, num_qubits)
    for sample_idx in range(batch_size):
        for state, count in counts.items():
            for i, bit in enumerate(reversed(state)):
                features[sample_idx, i] += int(bit) * count / 1024

    return features.reshape(batch_size, 3, 3, 3)


import torch
import torch.nn as nn

class HybridNet(nn.Module):
    def __init__(self, num_features, num_classes, encoding_method):
        super(HybridNet, self).__init__()

        # Quantum encoding layer
        self.quantum_encoding = encoding_method

        # Deconvolution layers to upsample to (batch_size, 3, 32, 32)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(3, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 16, 6, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (batch_size, 32, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (batch_size, 16, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)   # (batch_size, 3, 32, 32)
        )

        # Classical layers
        self.classical_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (batch_size, 64, 8, 8)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # (batch_size, 128, 4, 4)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    # def forward(self, x):
        # # Quantum encoding step
        # x = self.quantum_encoding(x)  # 输入假定为 (batch_size, num_features)
        # x = x.view(x.size(0), 3, 3, 3)  # 变形为 (batch_size, 3, 3, 3)
        #
        # # Upsample using deconvolution
        # x = self.deconv_layers(x)  # 输出形状为 (batch_size, 3, 32, 32)
        #
        # # Classical layers
        # x = self.classical_layers(x)  # 提取特征
        # x = x.view(x.size(0), -1)    # 展平为全连接层输入
        # x = self.fc_layers(x)        # 分类
        # return x
    def forward(self, x):
        # Quantum encoding
        x = self.quantum_encoding(x)
        x = x.view(x.size(0), 3, 3, 3)
        x = self.deconv_layers(x)  # 输出 (batch_size, 3, 32, 32)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

        # Classical layers
        x = self.classical_layers(x)  # 提取特征
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                device='cpu', start_epoch=0, save_model="best_model.pth"):
    best_val_acc = 0.0
    
    # 创建列表来存储训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    save_dir = os.path.dirname(save_model)

    # 如果存在之前的训练状态，加载最佳验证准确率
    if os.path.exists('best_val_acc.pth'):
        best_val_acc = torch.load('best_val_acc.pth')

    print(f"Starting training for {num_epochs} epochs...")
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

        # 计算当前epoch的平均训练损失和准确率
        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 计算当前epoch的平均验证损失和准确率
        epoch_val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        # 存储训练历史
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {epoch_train_loss:.4f}, Training Acc: {epoch_train_acc:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_model)
            torch.save(best_val_acc, f'{os.path.splitext(save_model)[0]}_acc.pth')
            print(f'Best model saved with accuracy: {best_val_acc:.2f}%')

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        torch.save(checkpoint, 'checkpoint.pth')

    print("Training completed. Generating training curves...")
    try:
        # 确保plots目录存在
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, pad=15)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_facecolor('#f8f9fa')
        ax1.set_axisbelow(True)
        
        # 绘制准确率曲线
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, pad=15)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_facecolor('#f8f9fa')
        ax2.set_axisbelow(True)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plots_dir, f'training_curves_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Training curves have been saved to: {save_path}")
    except Exception as e:
        print(f"Error saving training curves: {str(e)}")

    return model

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='./'):
    """
    绘制训练和验证的损失曲线与准确率曲线
    """
    # 创建plots目录
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, pad=15)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_facecolor('#f8f9fa')
    ax1.set_axisbelow(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, pad=15)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_facecolor('#f8f9fa')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图像，使用时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plots_dir, f'training_curves_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Training curves have been saved to: {save_path}")

def plot_confusion_matrix(cm, class_names, save_dir='./', timestamp=None):
    """
    绘制并保存混淆矩阵
    """
    # 创建plots目录
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # 使用时间戳命名文件
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plots_dir, f'confusion_matrix_{timestamp}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix has been saved to: {save_path}")

def evaluate_model(model, test_loader, device, class_names=None, save_dir='./'):
    """
    评估模型性能并生成结果
    """
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

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, save_dir, timestamp)

    # 计算评估指标
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds)
    
    # 获取完整的分类报告
    full_report = classification_report(all_labels, all_preds,
                                     target_names=class_names if class_names else [str(i) for i in range(len(precision))],
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
        'Accuracy': accuracies,
        'Support': support
    })

    # 计算总体指标
    overall_metrics = pd.DataFrame({
        'Class': ['macro_avg', 'weighted_avg'],
        'Precision': [full_report['macro avg']['precision'], full_report['weighted avg']['precision']],
        'Recall': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']],
        'F1': [full_report['macro avg']['f1-score'], full_report['weighted avg']['f1-score']],
        'Accuracy': [full_report['macro avg']['recall'], full_report['weighted avg']['recall']],
        'Support': [sum(support), sum(support)]
    })

    # 合并结果
    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)

    # 创建评估结果目录
    results_dir = os.path.join(save_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    # 保存为CSV
    csv_path = os.path.join(results_dir, f"evaluation_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.4f')

    # 打印结果
    print("\nEvaluation Results:")
    print(results_df.to_string(float_format=lambda x: '{:.4f}'.format(x)))
    print(f"\nTotal Loss: {avg_loss:.4f}")
    print(f"\nResults have been saved to: {csv_path}")

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
    num_epochs = 40
    device = torch.device('cpu')
    save_dir = "model_zoos"
    os.makedirs(save_dir, exist_ok=True)
    save_model = os.path.join(save_dir, "Hybrid_43cls_pqe.pth")

    print("Loading data from pickle files...")
    # 加载训练数据
    with open("pkls/train_dataset_1k.pkl", 'rb') as f:
        train_data = pickle.load(f)

    # 加载测试数据
    with open("pkls/test_dataset_100.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print("Processing training data...")
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())

    print("Processing test data...")
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    # 训练数据处理 - 使用数据增强
    print("Processing training images with augmentation...") # todo: change here
    train_features = [preprocess_image(path, label, is_training=True) for (path, label) in zip(train_paths, train_labels)]

    # 测试数据处理 - 不使用数据增强
    print("Processing test images without augmentation...") # todo: change here
    test_features = [preprocess_image(path, label, is_training=False) for (path, label) in zip(test_paths, test_labels)]

    # 转换为numpy数组
    train_features = np.array(train_features)
    test_features = np.array(test_features)

    # PCA降维
    # 初始化PCA并进行降维
    # 将训练集和测试集的特征数据合并
    feats_all = np.concatenate([train_features, test_features], axis=0)

    # 使用合并后的数据训练 PCA 模型
    pca = PCA(n_components=27)
    feats_all_reduced = pca.fit_transform(feats_all)

    # 将降维后的数据拆分回训练集和测试集
    feats_train_reduced = feats_all_reduced[:len(train_features)]
    feats_test_reduced = feats_all_reduced[len(test_features):]

    # Split training data
    x_train, x_val, y_train, y_val = train_test_split(
        feats_train_reduced, train_labels, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    num_features = feats_train_reduced.shape[1]  # 使用shape获取特征数量
    model = HybridNet(num_features=num_features, num_classes=43, encoding_method=parameterized_quantum_encoding).to(device)
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
        num_epochs=num_epochs, device=device, start_epoch=start_epoch, 
        save_model=save_model
    )

    # 创建测试数据加载器
    test_dataset = ImageDataset(feats_test_reduced, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    print("Initializing model...")
    model = HybridNet(num_features=num_features, num_classes=43, encoding_method=parameterized_quantum_encoding).to(device)

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
        evaluate_model(model, test_loader, device, class_names, save_dir=save_dir)
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return
    
if __name__ == "__main__":
    main()