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

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library import ZFeatureMap
from qiskit.circuit import Parameter, ParameterVector

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


# 量子编码器的抽象基类
class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.parameters = []

    def encode(self, qc, qr, parameters):
        raise NotImplementedError

    def get_param_count(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# ZFeatureMap编码器
class ZFeatureMapEncoder(QuantumEncoder):
    def __init__(self, n_qubits, reps=2):
        super().__init__(n_qubits)
        self.reps = reps
        self.feature_map = ZFeatureMap(n_qubits, reps=reps)
        self.simulator = AerSimulator()

        # Create parameter list for consistent referencing
        self.param_list = [Parameter(f'θ_{i}') for i in range(self.n_qubits * self.reps)]

    def encode(self, qc, qr, parameters):
        for i in range(self.n_qubits):
            qc.h(qr[i])

        for rep in range(self.reps):
            for i in range(self.n_qubits):
                param_idx = rep * self.n_qubits + i
                qc.p(parameters[self.param_list[param_idx]], qr[i])

            for i in range(self.n_qubits - 1):
                qc.cx(qr[i], qr[i + 1])
            if self.n_qubits > 1:
                qc.cx(qr[self.n_qubits - 1], qr[0])

        return qc

    def get_param_count(self):
        return self.n_qubits * self.reps

    def forward(self, x):
        batch_size = x.shape[0]
        results = []

        for sample in x:
            # 创建量子电路
            qr = QuantumRegister(self.n_qubits)
            cr = ClassicalRegister(self.n_qubits)
            qc = QuantumCircuit(qr, cr)

            # 编码数据
            params = sample.detach().cpu().numpy()
            parameter_dict = {self.param_list[i]: float(params[i])
                              for i in range(min(len(self.param_list), len(params)))}

            # 应用量子门
            self.encode(qc, qr, parameter_dict)

            # 测量
            qc.measure(qr, cr)

            # 运行电路
            transpiled_circuit = transpile(qc, self.simulator)
            job = self.simulator.run(transpiled_circuit, shots=1024)
            counts = job.result().get_counts()

            # 处理结果
            feature_vector = np.zeros(2 ** self.n_qubits)
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                feature_vector[index] = count / 1024

            results.append(feature_vector)

        return torch.tensor(results, dtype=torch.float32, device=x.device)


# 混合量子-经典神经网络
class NewHybridNet(nn.Module):
    def __init__(self, input_size, n_qubits, n_classes):
        super(NewHybridNet, self).__init__()

        # 输入预处理层
        self.input_preprocessor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits * 2),  # ZFeatureMap的参数数量
            nn.Tanh()
        )

        # 量子层
        self.quantum_layer = ZFeatureMapEncoder(n_qubits=n_qubits)

        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        quantum_output_size = 2 ** n_qubits
        self.fc_layers = nn.Sequential(
            nn.Linear(quantum_output_size + 128 * 4 * 4, 256),  # 合并量子特征和卷积特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # 经典预处理
        quantum_params = self.input_preprocessor(x)

        # 量子处理
        quantum_features = self.quantum_layer(quantum_params)

        # 卷积处理
        # 重塑输入以适应卷积层 (batch_size, channels, height, width)
        conv_input = x.view(x.size(0), 3, 32, 32)  # 假设输入是32x32的图像
        conv_features = self.conv_layers(conv_input)
        conv_features = conv_features.view(conv_features.size(0), -1)

        # 特征融合
        combined_features = torch.cat([quantum_features, conv_features], dim=1)

        # 最终分类
        output = self.fc_layers(combined_features)
        return output


def preprocess_image(image_path: str, label: int, size: tuple = (32, 32), aug_prob: float = 0.5,
                     is_training: bool = False) -> np.ndarray:
    try:
        image = Image.open(image_path)

        if is_training and np.random.random() < aug_prob:
            augmentations = [
                ('rotate', np.random.random() < 0.3),
                ('flip', np.random.random() < 0.3),
                ('brightness', np.random.random() < 0.3),
                ('contrast', np.random.random() < 0.3),
                ('noise', np.random.random() < 0.3)
            ]

            for aug_type, apply in augmentations:
                if apply:
                    if aug_type == 'rotate':
                        angle = np.random.uniform(-15, 15)
                        image = image.rotate(angle, expand=False, resample=Image.BILINEAR)
                    elif aug_type == 'flip' and label not in [33, 34, 36, 37, 38, 39]:
                        if np.random.random() < 0.5:
                            image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif aug_type == 'brightness':
                        from PIL import ImageEnhance
                        factor = np.random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Brightness(image)
                        image = enhancer.enhance(factor)
                    elif aug_type == 'contrast':
                        from PIL import ImageEnhance
                        factor = np.random.uniform(0.8, 1.2)
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(factor)
                    elif aug_type == 'noise':
                        img_array = np.array(image)
                        noise = np.random.normal(0, 5, img_array.shape)
                        noisy_img = img_array + noise
                        noisy_img = np.clip(noisy_img, 0, 255)
                        image = Image.fromarray(noisy_img.astype(np.uint8))

        image = image.resize(size)
        normalized = np.array(image) / 255.0
        r = normalized[:, :, 0].flatten()
        g = normalized[:, :, 1].flatten()
        b = normalized[:, :, 2].flatten()
        combined = np.concatenate([r, g, b])
        norm = np.sqrt(np.sum(combined ** 2))
        normalized_amplitude = combined / norm if norm > 0 else combined
        return normalized_amplitude

    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                device='cpu', start_epoch=0, save_dir="model_zoos"):
    """
    训练模型并保存训练过程
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_model = os.path.join(save_dir, "best_hybrid_model.pth")

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
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

            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })

        epoch_train_loss = running_loss / total
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证阶段
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

        epoch_val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {epoch_train_loss:.4f}, Training Acc: {epoch_train_acc:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_model)
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
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

        # 绘制训练曲线
        if (epoch + 1) % 5 == 0:
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir)

    return model


def evaluate_model(model, test_loader, device, class_names=None, save_dir='./'):
    """
    评估模型性能并生成详细报告
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

    # 计算评估指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_dir)

    # 生成详细报告
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names if class_names else [str(i) for i in range(len(precision))],
        digits=4
    )

    # 保存结果
    results = {
        'accuracy': test_correct / test_total * 100,
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

    # 保存评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(save_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    # 保存为CSV
    results_df = pd.DataFrame({
        'Class': range(len(precision)),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Support': support
    })
    results_df.to_csv(os.path.join(results_dir, f'evaluation_results_{timestamp}.csv'),
                      index=False)

    # 打印结果
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print("\nDetailed Classification Report:")
    print(report)

    return results


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='./'):
    """
    绘制训练和验证的损失曲线与准确率曲线
    """
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # 保存图像
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f'training_curves_{timestamp}.png'))
    plt.close()


def plot_confusion_matrix(cm, class_names, save_dir='./'):
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

    # 保存图像
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{timestamp}.png'))
    plt.close()

def main():
    # 参数设置
    batch_size = 32
    num_epochs = 40
    n_qubits = 6
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    save_dir = "runs/"


    # 数据加载和预处理
    print("Loading data...")
    with open("pkls/train_dataset_1k.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("pkls/test_dataset_100.pkl", 'rb') as f:
        test_data = pickle.load(f)

    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    # 处理图像数据
    print("Processing images...")
    train_features = [preprocess_image(path, label, is_training=True)
                      for path, label in zip(train_paths, train_labels)]
    test_features = [preprocess_image(path, label, is_training=False)
                     for path, label in zip(test_paths, test_labels)]

    train_features = np.array(train_features)
    test_features = np.array(test_features)

    # 创建数据加载器
    x_train, x_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )

    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)
    test_dataset = ImageDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    input_size = train_features.shape[1]  # 获取输入特征维度
    model = NewHybridNet(
        input_size=input_size,
        n_qubits=n_qubits,
        n_classes=43
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("Starting model training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir
    )

    # 定义类别名称
    class_names = [str(i) for i in range(43)]  # 43个类别

    # 评估模型
    print("Starting model evaluation...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=save_dir
    )

    print("Training and evaluation completed!")
    return results


if __name__ == "__main__":
    main()