import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import glob

class Config:
    def __init__(self):
        self.model_path = "model_zoos/Clssical_43cls_pqe.pth"
        self.num_classes = 43
        self.input_size = (32, 32)
        self.batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = "inference_data"
        self.output_dir = "inference_results"
        # 标签映射
        self.classes = {
            0:'University of Waterloo', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
            19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
            38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
            41:'End of no passing', 42:'End no passing veh > 3.5 tons'
        }

class ValidationDataset(Dataset):
    """验证数据集类"""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        # 使用与训练时相同的预处理方法
        processed_data = preprocess_image(img_path, None, self.transform)
        return processed_data, img_path

def preprocess_image(image_path: str, label: int = None, size: tuple = (32, 32)) -> np.ndarray:
    """使用与训练时完全相同的预处理方法"""
    try:
        image = Image.open(image_path)
        
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

        return torch.FloatTensor(normalized_amplitude)

    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def load_model(config: Config):
    """加载模型"""
    try:
        from classical_43cls import HybridNet, parameterized_quantum_encoding
        
        # 初始化模型 - 使用正确的特征数量
        num_features = config.input_size[0] * config.input_size[1] * 3
        model = HybridNet(
            num_features=num_features,
            num_classes=config.num_classes,
            encoding_method=parameterized_quantum_encoding
        ).to(config.device)
        
        # 加载模型权重
        checkpoint = torch.load(config.model_path, map_location=config.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_images(model: nn.Module, data_loader: DataLoader, config: Config) -> Dict[str, int]:
    """预测图片类别和置信度"""
    predictions = {}
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for inputs, paths in tqdm(data_loader, desc="Predicting"):
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            
            # 计算softmax概率
            probabilities = softmax(outputs)
            
            # 获取最高概率的类别和对应的概率值
            probs, predicted = probabilities.max(1)
            
            for path, pred, prob in zip(paths, predicted.cpu().numpy(), probs.cpu().numpy()):
                predictions[path] = (int(pred), float(prob))
    
    return predictions

def visualize_predictions(predictions: Dict[str, Tuple[int, float]], config: Config):
    """可视化预测结果，包含置信度"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置统一的输出图片大小
    OUTPUT_SIZE = (224, 224)
    FONT_SIZE = 16
    TEXT_MARGIN = 10
    
    try:
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", FONT_SIZE)
            except:
                font = ImageFont.load_default()
        
        for img_path, (pred_class, confidence) in tqdm(predictions.items(), desc="Visualizing"):
            # 读取原始图片
            image = Image.open(img_path).convert('RGB')
            
            # 计算调整大小后的图片尺寸，保持纵横比
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:
                new_width = OUTPUT_SIZE[0]
                new_height = int(OUTPUT_SIZE[0] / aspect_ratio)
            else:
                new_height = OUTPUT_SIZE[1]
                new_width = int(OUTPUT_SIZE[1] * aspect_ratio)
            
            # 调整图片大小
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 获取预测类别的标签
            label = config.classes[pred_class]
            # 添加置信度到文本中
            text = f"Class {pred_class}: {label}\nConfidence: {confidence:.2%}"
            
            # 计算多行文本的大小
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 为两行文本增加额外空间
            total_text_height = text_height + FONT_SIZE  # 为两行文本预留空间
            
            # 创建新的白色背景图像，高度加上文本高度和边距
            new_image = Image.new('RGB', (OUTPUT_SIZE[0], OUTPUT_SIZE[1] + total_text_height + 2*TEXT_MARGIN), 'white')
            
            # 计算图片在新画布上的位置（居中）
            paste_x = (OUTPUT_SIZE[0] - new_width) // 2
            paste_y = (OUTPUT_SIZE[1] - new_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            # 在新图像上绘制文本
            draw = ImageDraw.Draw(new_image)
            
            # 分别绘制类别和置信度（分两行）
            label_text = f"Class {pred_class}: {label}"
            conf_text = f"Confidence: {confidence:.2%}"
            
            # 计算每行文本的位置
            label_bbox = font.getbbox(label_text)
            conf_bbox = font.getbbox(conf_text)
            label_width = label_bbox[2] - label_bbox[0]
            conf_width = conf_bbox[2] - conf_bbox[0]
            
            # 绘制类别文本
            text_x = (OUTPUT_SIZE[0] - label_width) // 2
            text_y = OUTPUT_SIZE[1] + TEXT_MARGIN
            draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)
            
            # 绘制置信度文本
            conf_x = (OUTPUT_SIZE[0] - conf_width) // 2
            conf_y = text_y + FONT_SIZE + 5  # 5是行间距
            draw.text((conf_x, conf_y), conf_text, fill=(0, 0, 0), font=font)
            
            # 保存结果
            output_path = os.path.join(config.output_dir, f"pred_{os.path.basename(img_path)}")
            new_image.save(output_path)
            
    except Exception as e:
        print(f"Error visualizing predictions: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    config = Config()
    
    try:
        # 获取所有验证图片路径
        # 获取所有验证图片路径，包括 ppm 格式
        image_paths = glob.glob(os.path.join(config.data_dir, "*.jpg")) + \
                     glob.glob(os.path.join(config.data_dir, "*.png")) + \
                     glob.glob(os.path.join(config.data_dir, "*.ppm"))  # 添加 ppm 格式支持
        
        if not image_paths:
            raise Exception(f"No images found in {config.data_dir}")
        
        # 创建数据加载器 - 使用修改后的 Dataset
        dataset = ValidationDataset(
            image_paths=image_paths,
            transform=config.input_size
        )
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        # 加载模型
        print("Loading model...")
        model = load_model(config)
        
        # 进行预测
        print("Making predictions...")
        predictions = predict_images(model, data_loader, config)
        
        # 可视化结果
        print("Visualizing results...")
        visualize_predictions(predictions, config)
        
        print(f"Validation complete. Results saved to {config.output_dir}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == "__main__":
    main()