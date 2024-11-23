import os
import numpy as np
import pickle
from tqdm import tqdm
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from training_tools import TrainingTools, EvaluationTools, BaseDataset, ModelLoader, DataAugmentation

class CNNNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNNNet, self).__init__()
        
        # 调整输入层以适应3通道
        self.input_fc = nn.Linear(num_features, 32*32*3)
        
        # Classical layers after quantum features
        self.classical_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道改为3
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
        x = self.input_fc(x)
        x = x.view(-1, 3, 32, 32)  # 重塑为3通道
        x = self.classical_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def main():
    # 配置参数
    batch_size = 32
    num_epochs = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Classical_43cls"
    
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
    
    print("Loading data from pickle files...")
    # 加载训练数据
    with open("pkls/train_dataset.pkl", 'rb') as f:
        train_data = pickle.load(f)

    # 加载测试数据
    with open("pkls/test_dataset.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # 在这里初始化数据增强工具
    data_augmentor = DataAugmentation(size=(28, 28), aug_prob=0.5)
    
    print("\nProcessing training images...")
    train_paths = list(train_data['image_paths'].values())
    train_labels = list(train_data['labels'].values())
    train_features, train_labels = data_augmentor.process_dataset(
        train_paths, train_labels, is_training=True
    )
    train_features = train_features.reshape(train_features.shape[0], -1)

    print("\nProcessing test images...")
    test_paths = list(test_data['image_paths'].values())
    test_labels = list(test_data['labels'].values())
    test_features, test_labels = data_augmentor.process_dataset(
        test_paths, test_labels, is_training=False
    )
    test_features = test_features.reshape(test_features.shape[0], -1)

    # 分割训练和验证数据
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # 创建数据加载器
    train_dataset = BaseDataset(train_features, train_labels)
    val_dataset = BaseDataset(val_features, val_labels)
    test_dataset = BaseDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    num_features = train_features.shape[1]  # 获取特征维度
    num_classes = 43
    model = CNNNet(num_features=num_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 检查是否有检查点可以恢复
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

    # 训练模型
    print("Starting model training...")
    model, history = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        model_name=model_name,
        start_epoch=start_epoch,
        early_stopping_patience=None
    )

    # 加载最佳模型进行评估
    print("Loading best model for evaluation...")
    model = model_loader.load_for_inference(
        model=model,
        weights_path=os.path.join(save_dir, f'{model_name}/{model_name}_best.pth'),
        device=device
    )
    
    if model is not None:
        print("Starting model evaluation...")
        class_names = [str(i) for i in range(num_classes)]
        
        # 评估模型
        results, avg_loss, accuracy = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            class_names=class_names
        )
        
        print(f"\nFinal Test Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Failed to load best model for evaluation")

if __name__ == "__main__":
    main()
