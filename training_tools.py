import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from PIL import Image, ImageEnhance

class BaseDataset(Dataset):
    def __init__(self, images, labels):
        if isinstance(images, list):
            images = np.array(images)
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class DataAugmentation:
    def __init__(self, size=(28, 28), aug_prob=0.5):
        self.size = size
        self.aug_prob = aug_prob

    def preprocess_image(self, image_path: str, label: int, is_training: bool = False):
        try:
            image = Image.open(image_path).convert('RGB')
            
            if is_training and np.random.random() < self.aug_prob:
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
                        
                        elif aug_type == 'flip' and label not in [33,34,36,37,38,39]:
                            if np.random.random() < 0.5:
                                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                        
                        elif aug_type == 'brightness':
                            factor = np.random.uniform(0.8, 1.2)
                            enhancer = ImageEnhance.Brightness(image)
                            image = enhancer.enhance(factor)
                        
                        elif aug_type == 'contrast':
                            factor = np.random.uniform(0.8, 1.2)
                            enhancer = ImageEnhance.Contrast(image)
                            image = enhancer.enhance(factor)
                        
                        elif aug_type == 'noise':
                            img_array = np.array(image)
                            noise = np.random.normal(0, 5, img_array.shape)
                            noisy_img = img_array + noise
                            noisy_img = np.clip(noisy_img, 0, 255)
                            image = Image.fromarray(noisy_img.astype(np.uint8))

            image = image.resize(self.size)

            img_array = np.array(image) / 255.0
            return img_array

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def process_dataset(self, image_paths, labels, is_training=False):
        features = []
        processed_labels = []
        
        for path, label in zip(image_paths, labels):
            img = self.preprocess_image(path, label, is_training)
            if img is not None:
                features.append(img)
                processed_labels.append(label)
                
        return np.array(features), np.array(processed_labels)
    
class ModelLoader:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def load_model_and_state(self, model, model_path, optimizer=None, device='cpu'):
        try:
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            start_epoch = 0
            best_val_acc = 0.0
            history = {
                'train_losses': [],
                'val_losses': [],
                'train_accs': [],
                'val_accs': []
            }
            
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                if 'best_val_acc' in checkpoint:
                    best_val_acc = checkpoint['best_val_acc']
                if 'history' in checkpoint:
                    history = checkpoint['history']
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Successfully loaded checkpoint from epoch {start_epoch}")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            
            return model, optimizer, start_epoch, best_val_acc, history
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return model, optimizer, 0, 0.0, None

    def load_for_inference(self, model, weights_path, device='cpu'):
        try:
            print(f"Loading model weights from {weights_path}")
            weights = torch.load(weights_path, map_location=device, weights_only=True)
            
            if isinstance(weights, dict):
                if 'model_state_dict' in weights:
                    model.load_state_dict(weights['model_state_dict'])
                elif 'state_dict' in weights:
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)
            else:
                model.load_state_dict(weights)
                
            model.eval()
            print("Model loaded successfully and set to evaluation mode")
            return model
            
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            return None

    def save_checkpoint(self, model, optimizer, epoch, best_val_acc, 
                       history, model_name, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'{model_name}_checkpoint.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, 
                f'{model_name}_best.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model checkpoint with accuracy: {best_val_acc:.2f}%")

class TrainingTools:
    def __init__(self, model_save_dir='model_checkpoints', tensorboard_dir='runs/training_logs'):
        self.model_save_dir = model_save_dir
        self.tensorboard_dir = tensorboard_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.model_loader = ModelLoader(checkpoint_dir=model_save_dir)
        
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, 
                   num_epochs=10, device='cpu', start_epoch=0, 
                   model_name="model", early_stopping_patience=None):
        writer = SummaryWriter(self.tensorboard_dir)
        best_val_acc = 0.0
        patience_counter = 0
        
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }
        
        print(f"Starting training...")
        try:
            for epoch in range(start_epoch, start_epoch + num_epochs):
                model.train()
                train_metrics = self._train_epoch(model, train_loader, criterion, 
                                                optimizer, device, epoch, writer)
                
                val_metrics = self._validate_epoch(model, val_loader, criterion, device)
                
                for key in history:
                    history[key].append(
                        train_metrics[key] if 'train' in key else val_metrics[key.replace('train', 'val')]
                    )
                
                self._log_metrics(writer, train_metrics, val_metrics, epoch)
                
                if val_metrics['val_accs'] > best_val_acc:
                    best_val_acc = val_metrics['val_accs']
                    self.model_loader.save_checkpoint(
                        model, optimizer, epoch, best_val_acc, 
                        history, model_name, is_best=True
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.model_loader.save_checkpoint(
                        model, optimizer, epoch, best_val_acc, 
                        history, model_name, is_best=False
                    )
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                print(f'Epoch {epoch + 1}/{start_epoch + num_epochs}:')
                print(f'Training Loss: {train_metrics["train_losses"]:.4f}, '
                    f'Training Acc: {train_metrics["train_accs"]:.2f}%')
                print(f'Validation Loss: {val_metrics["val_losses"]:.4f}, '
                    f'Validation Acc: {val_metrics["val_accs"]:.2f}%')  
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            writer.close()
            
        self._plot_training_curves(history, model_name)
        
        return model, history
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, device, epoch, writer):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
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
            
            writer.add_scalar('Loss/train_batch', 
                            loss.item(),
                            epoch * len(train_loader) + batch_idx)
            
            pbar.set_postfix({
                'loss': running_loss / total, 
                'acc': 100. * correct / total
            })
        
        return {
            'train_losses': running_loss / total,
            'train_accs': 100. * correct / total
        }
    
    def _validate_epoch(self, model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'val_losses': val_loss / total,
            'val_accs': 100. * correct / total
        }
    
    def _log_metrics(self, writer, train_metrics, val_metrics, epoch):
        writer.add_scalars('Loss/epoch', {
            'train': train_metrics['train_losses'],
            'val': val_metrics['val_losses']
        }, epoch)
        
        writer.add_scalars('Accuracy', {
            'train': train_metrics['train_accs'],
            'val': val_metrics['val_accs']
        }, epoch)
    
    def _plot_training_curves(self, history, model_name):
        plots_dir = os.path.join(self.model_save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        epochs = range(1, len(history['train_losses']) + 1)

        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, pad=15)
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, pad=15)
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(plots_dir, f'{model_name}_training_curves_{timestamp}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class EvaluationTools:
    def __init__(self, results_dir='evaluation_results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def evaluate_model(self, model, test_loader, criterion, device, class_names=None):
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        avg_loss = test_loss / total
        accuracy = 100. * correct / total
        results = self._calculate_metrics(all_labels, all_preds, class_names)
        self._plot_confusion_matrix(all_labels, all_preds, class_names)
        self._save_results(results, avg_loss, accuracy)
        
        return results, avg_loss, accuracy
    
    def _calculate_metrics(self, true_labels, predictions, class_names=None):
            if class_names is None:
                class_names = [str(i) for i in range(len(np.unique(true_labels)))]
                
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, labels=range(len(class_names)), zero_division=0
            )
            
            full_report = classification_report(
                true_labels, predictions,
                target_names=class_names,
                digits=4,
                output_dict=True,
                zero_division=0
            )
            
            class_accuracies = []
            for i in range(len(class_names)):
                mask = true_labels == i
                class_accuracy = (predictions[mask] == i).mean() if mask.any() else 0
                class_accuracies.append(class_accuracy)

            results_df = pd.DataFrame({
                'Class': class_names,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Accuracy': class_accuracies,
                'Support': support
            })
            
            avg_metrics = pd.DataFrame({
                'Class': ['macro avg', 'weighted avg'],
                'Precision': [full_report['macro avg']['precision'], 
                            full_report['weighted avg']['precision']],
                'Recall': [full_report['macro avg']['recall'],
                        full_report['weighted avg']['recall']],
                'F1': [full_report['macro avg']['f1-score'],
                    full_report['weighted avg']['f1-score']],
                'Accuracy': [full_report['macro avg']['recall'],
                            full_report['weighted avg']['recall']],
                'Support': [sum(support), sum(support)]
            })
            
            results_df = pd.concat([results_df, avg_metrics], ignore_index=True)
            return results_df
        
    def _plot_confusion_matrix(self, true_labels, predictions, class_names=None):
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(true_labels)))]
            
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.results_dir, f'confusion_matrix_{timestamp}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix has been saved to: {save_path}")

    def _save_results(self, results_df, avg_loss, accuracy):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.csv")
        
        results_df['Total Loss'] = avg_loss
        results_df['Total Accuracy'] = accuracy
        
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        
        print("\nEvaluation Results Summary:")
        print(f"Total Loss: {avg_loss:.4f}")
        print(f"Total Accuracy: {accuracy:.2f}%")
        print(f"\nDetailed results have been saved to: {csv_path}")

    def get_prediction(self, model, input_tensor, device='cpu'):
        model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
            
        return predicted.item(), confidence.item()

    def get_batch_predictions(self, model, data_loader, device='cpu'):
        model.eval()
        predictions = []
        confidences = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc='Generating predictions'):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)
                
                predictions.extend(predicted.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return predictions, confidences, true_labels