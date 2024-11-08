import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image


class ImageDataset(Dataset):
    """Custom Dataset for loading images"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        return image, label


def preprocess_image(image_path):
    """
    Preprocess a single image
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
        return img_tensor
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return torch.zeros((3, 32, 32))


class ClassicalCNNWithConv(nn.Module):
    def __init__(self, num_classes=43):
        super(ClassicalCNNWithConv, self).__init__()

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

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def evaluate_model(model, data_loader, criterion, device, class_names=None):
    """
    详细评估模型性能
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_preds) == 0 or len(all_labels) == 0:
        print("Warning: No predictions or labels found!")
        return None

    unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
    n_classes = len(unique_labels)

    if n_classes == 0:
        print("Warning: No classes found in the data!")
        return None

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=unique_labels, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    if cm.size == 0:
        print("Warning: Empty confusion matrix!")
        return None

    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        if np.sum(cm[i]) != 0:
            per_class_acc[i] = cm[i, i] / np.sum(cm[i])

    # 保存评估结果
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)

    # 绘制混淆矩阵
    try:
        plt.figure(figsize=(15, 12))
        annot = True if n_classes <= 10 else False
        sns.heatmap(cm, annot=annot, fmt='d', cmap='Blues',
                    xticklabels=unique_labels,
                    yticklabels=unique_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {str(e)}")

    # 绘制性能指标图
    try:
        plt.figure(figsize=(15, 6))
        x = np.arange(n_classes)
        width = 0.2

        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-score')

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Performance Metrics per Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'performance_metrics.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create performance metrics plot: {str(e)}")

    eval_results = {
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'accuracy': per_class_acc.tolist(),
            'support': support.tolist()
        },
        'overall_metrics': {
            'macro_precision': np.mean(precision),
            'macro_recall': np.mean(recall),
            'macro_f1': np.mean(f1),
            'macro_accuracy': np.mean(per_class_acc),
            'weighted_f1': np.average(f1, weights=support),
            'total_loss': total_loss / len(data_loader)
        }
    }

    # 打印评估结果
    print("\nDetailed Evaluation Results:")
    print(f"\nNumber of classes found: {n_classes}")
    print("\nPer-class Metrics:")
    print(f"{'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'Support':>10}")
    print("-" * 60)

    for i, label in enumerate(unique_labels):
        class_name = f"Class {label}" if class_names is None else class_names[label]
        print(f"{class_name:<6} {precision[i]:>10.4f} {recall[i]:>10.4f} "
              f"{f1[i]:>10.4f} {per_class_acc[i]:>10.4f} {support[i]:>10d}")

    print("\nOverall Metrics:")
    for metric, value in eval_results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")

    try:
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save evaluation results to file: {str(e)}")

    return eval_results


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    best_val_f1 = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 计算当前批次的指标
            batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
                labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)

            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'f1': batch_f1
            })

        # Validation phase
        model.eval()
        val_results = evaluate_model(model, val_loader, criterion, device)
        if val_results is not None:
            val_f1 = val_results['overall_metrics']['macro_f1']
            print(f'Validation Macro F1: {val_f1:.4f}')

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'best_classical_model.pth')

    return model


def load_data(train_pkl_path, test_pkl_path, val_split=0.2, batch_size=32):
    """Load and prepare data from pkl files"""
    # Load pkl files
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)

    # Extract paths and labels
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())

    # Split training data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split,
        random_state=42,
        stratify=train_labels
    )

    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels)
    val_dataset = ImageDataset(val_paths, val_labels)
    test_dataset = ImageDataset(test_paths, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def main():
    # Parameters
    num_epochs = 10
    batch_size = 32
    train_pkl = "pkls/train_dataset.pkl"
    test_pkl = "pkls/test_dataset.pkl"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(
        train_pkl_path=train_pkl,
        test_pkl_path=test_pkl,
        val_split=0.2,
        batch_size=batch_size
    )

    # Get number of classes
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    num_classes = len(set(train_data['labels'].values()))
    print(f"Number of classes: {num_classes}")

    # Initialize model
    model = ClassicalCNNWithConv(num_classes=num_classes).to(device)

    # Calculate class weights
    class_counts = np.bincount([label for label in train_data['labels'].values()])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    print("\nEvaluating on test set...")
    test_results = evaluate_model(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()