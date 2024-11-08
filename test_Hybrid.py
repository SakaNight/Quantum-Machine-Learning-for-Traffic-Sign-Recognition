from Hybrid_43cls import (
    ImageDataset,
    preprocess_image,
    HybridNet,
    evaluate_model,
    load_model_weights
)

import os
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    model_path = 'model_zoos/Hybrid_43cls.pth'
    # 设置设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print(f"Using device: {device}")

    # 加载测试数据
    print("Loading test data...")
    try:
        with open("pkls/test_dataset.pkl", 'rb') as f:
            test_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return

    # 处理测试数据
    print("Processing test data...")
    try:
        test_paths = [test_data['image_paths'][i] for i in range(len(test_data['image_paths']))]
        test_labels = [test_data['labels'][i] for i in range(len(test_data['labels']))]
    except KeyError as e:
        print(f"Error accessing data from test_dataset.pkl. Missing key: {str(e)}")
        print("Available keys:", test_data.keys())
        return

    # 预处理图像
    try:
        test_features = []
        for path in tqdm(test_paths, desc="Processing images"):
            try:
                feature = preprocess_image(path)
                test_features.append(feature)
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                continue

        if not test_features:
            print("No images were successfully processed")
            return
    except Exception as e:
        print(f"Error during image preprocessing: {str(e)}")
        return

    # 创建测试数据加载器
    test_dataset = ImageDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    print("Initializing model...")
    num_features = len(test_features[0])
    model = HybridNet(num_features=num_features, num_classes=43).to(device)

    # 加载模型权重
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # 尝试加载模型
    if not load_model_weights(model, model_path, device):
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