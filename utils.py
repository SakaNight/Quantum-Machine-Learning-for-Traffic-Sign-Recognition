# 基础库
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image


# PyTorch相关
import torch
import torch.nn as nn
from torchvision import models, transforms

# 数据处理和评估相关
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd

# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns

def load_resnet18(model_path, device='cuda'):
    """
    从本地加载ResNet18模型
    """
    try:
        # 创建模型架构
        model = models.resnet18(pretrained=False)

        # 加载权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        print(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用预训练的ResNet18作为特征提取器
        resnet = models.resnet18(pretrained=pretrained)
        # 只使用特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 添加自定义的特征处理层
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 统一特征图大小
            nn.Conv2d(512, 256, 1),  # 降低通道数
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.feature_processor(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=3072, latent_dim=256):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, latent_dim),
            nn.Tanh()  # 限制潜在空间的范围
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()  # 图像数值范围[0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ImprovedPreprocessing:
    def __init__(self, size=(32, 32), use_autoencoder=True, use_feature_extractor=True):
        self.size = size
        self.use_autoencoder = use_autoencoder
        self.use_feature_extractor = use_feature_extractor

        # 数据增强转换
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 特征提取器
        if use_feature_extractor:
            self.feature_extractor = FeatureExtractor(pretrained=True)
            self.feature_extractor.eval()

        # 自动编码器
        if use_autoencoder:
            self.autoencoder = AutoEncoder()
            # 这里需要先训练自动编码器
            self.train_autoencoder()
            self.autoencoder.eval()

    def train_autoencoder(self, train_data=None):
        """训练自动编码器"""
        if train_data is None:
            return  # 实际使用时需要提供训练数据

        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        criterion = nn.MSELoss()

        for epoch in range(100):  # 训练100个epoch
            for batch in train_data:
                optimizer.zero_grad()
                output = self.autoencoder(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

    def extract_features(self, image):
        """提取高级特征"""
        # 转换为tensor
        img_tensor = self.transform(image).unsqueeze(0)
        features = []

        # 使用特征提取器
        if self.use_feature_extractor:
            with torch.no_grad():
                feat = self.feature_extractor(img_tensor)
                features.append(feat.flatten())

        # 使用自动编码器
        if self.use_autoencoder:
            with torch.no_grad():
                img_flat = img_tensor.flatten()
                encoded = self.autoencoder.encoder(img_flat)
                features.append(encoded)

        # 合并所有特征
        if features:
            combined_features = torch.cat(features)
            return combined_features.numpy()
        return img_tensor.numpy()

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """完整的预处理流程"""
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')

            # 提取特征
            features = self.extract_features(image)

            return features

        except Exception as e:
            raise Exception(f"Error preprocessing image {image_path}: {str(e)}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='./'):
    """
    绘制训练和验证的损失曲线与准确率曲线
    """
    plt.figure(figsize=(15, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # 保存图像
    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, 'plots', f'training_curves_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {save_path}")


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

    # 收集预测结果
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.view(-1, 3, 32, 32)  # 确保输入形状正确
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = total_loss / test_total
    accuracy = 100. * test_correct / test_total

    # 计算每个类别的指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 创建结果目录
    results_dir = os.path.join(save_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存混淆矩阵图
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{timestamp}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 创建详细的分类报告
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names if class_names else [str(i) for i in range(len(precision))],
        digits=4
    )

    # 创建DataFrame存储详细结果
    results_df = pd.DataFrame({
        'Class': class_names if class_names else range(len(precision)),
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Support': support
    })

    # 添加总体指标
    macro_avg = results_df.mean()[1:-1]  # 不包括Class和Support列
    weighted_avg = np.average(
        results_df[['Precision', 'Recall', 'F1-score']],
        weights=results_df['Support'],
        axis=0
    )

    results_df.loc['macro_avg'] = ['macro_avg'] + list(macro_avg) + [support.sum()]
    results_df.loc['weighted_avg'] = ['weighted_avg'] + list(weighted_avg) + [support.sum()]

    # 保存结果
    results_df.to_csv(os.path.join(results_dir, f'evaluation_results_{timestamp}.csv'),
                      index=False, float_format='%.4f')

    # 保存完整报告
    with open(os.path.join(results_dir, f'classification_report_{timestamp}.txt'), 'w') as f:
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # 打印主要结果
    print("\nEvaluation Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'detailed_results': results_df
    }


