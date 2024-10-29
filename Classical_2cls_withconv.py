
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
        normalized = np.array(image) / 255.0  # 归一化图像到 [0,1] 区间
        if normalized.shape[2] == 3:  # 确保是三通道图像
            normalized = np.transpose(normalized, (2, 0, 1))  # 转换为 (C, H, W)
        return normalized
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

class ClassicalCNNWithConv(nn.Module):
    def __init__(self, num_classes=43):
        super(ClassicalCNNWithConv, self).__init__()

        # 卷积层和池化层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道3（RGB），输出通道32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出通道64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出尺寸: 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输出通道128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸: 128x4x4
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 展平后作为输入
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 卷积层处理
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)  # 全连接层
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
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

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': running_loss / total, 'acc': 100. * correct / total})

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
            torch.save(model.state_dict(), 'best_classical_model.pth')

    return model


def main():
    num_epochs = 15
    device = torch.device('cpu')

    print("Loading data...")
    train_df = pd.read_csv("Filtered_ClassIds_12.csv")
    test_df = pd.read_csv("Filtered_ClassIds_12_Test.csv")

    print("Processing data...")
    train_features = [preprocess_image(path) for path in tqdm(train_df['Path'])]
    train_labels = train_df['ClassId'].values -1

    test_features = [preprocess_image(path) for path in tqdm(test_df['Path'])]
    test_labels = test_df['ClassId'].values -1

    x_train, x_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )
    x_test = test_features
    y_test = test_labels

    train_dataset = ImageDataset(x_train, y_train)
    val_dataset = ImageDataset(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # model = ClassicalCNN(input_size=32*32*3, num_classes=2).to(device)
    model = ClassicalCNNWithConv(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    test_dataset = ImageDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')


if __name__ == "__main__":
    main()
