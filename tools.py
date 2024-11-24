import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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

def load_pretrained_model(model, pretrained_path, device):
    """
    加载预训练模型

    Args:
        model: 模型实例
        pretrained_path: 预训练模型路径
        device: 运行设备

    Returns:
        model: 加载了预训练权重的模型
        start_epoch: 开始的epoch
        optimizer_state: 优化器状态（如果存在）
        best_val_acc: 最佳验证准确率（如果存在）
    """
    if not os.path.exists(pretrained_path):
        print(f"No pretrained model found at {pretrained_path}")
        return model, 0, None, 0.0

    print(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)

    # 检查模型结构是否匹配
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return model, 0, None, 0.0

    start_epoch = checkpoint.get('epoch', 0)
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)

    print(f"Loaded checkpoint from epoch {start_epoch} with validation accuracy {best_val_acc:.2f}%")
    return model, start_epoch, optimizer_state, best_val_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=2, device='cpu',
                save_dir='checkpoints', pretrained_path=None, resume_training=False):
    """
    训练模型，支持从预训练模型继续训练和TensorBoard可视化
    """
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    start_epoch = 0
    best_val_acc = 0.0

    # 加载预训练模型
    if pretrained_path and os.path.exists(pretrained_path):
        model, start_epoch, optimizer_state, best_val_acc = load_pretrained_model(
            model, pretrained_path, device
        )
        if resume_training and optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            print("Resumed optimizer state")

    def log_metrics(phase, epoch, loss, accuracy, n_iter=None):
        """记录指标到TensorBoard"""
        try:
            if n_iter is not None:
                writer.add_scalar(f'{phase}/Loss_Step', loss, n_iter)
                writer.add_scalar(f'{phase}/Accuracy_Step', accuracy, n_iter)
            writer.add_scalar(f'{phase}/Loss_Epoch', loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy_Epoch', accuracy, epoch)
        except Exception as e:
            print(f"Warning: Failed to log metrics: {str(e)}")


    def log_model_stats(model, writer, epoch):
        """记录模型参数的统计信息"""
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 记录梯度的均值和标准差
                    writer.add_scalar(f'gradients/{name}/mean',
                                      param.grad.mean().item(), epoch)
                    writer.add_scalar(f'gradients/{name}/std',
                                      param.grad.std().item(), epoch)

                # 记录权重的均值和标准差
                writer.add_scalar(f'weights/{name}/mean',
                                  param.data.mean().item(), epoch)
                writer.add_scalar(f'weights/{name}/std',
                                  param.data.std().item(), epoch)
        except Exception as e:
            print(f"Warning: Failed to log model stats: {str(e)}")

    n_total_steps = len(train_loader)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # print(f'\nEpoch {epoch + 1}/{start_epoch + num_epochs}')

        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 更新统计
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # 计算当前batch的指标
            current_loss = running_loss / train_total
            current_acc = 100. * train_correct / train_total

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

            # 每100个batch记录一次训练指标
            if batch_idx % 100 == 0:
                n_iter = epoch * n_total_steps + batch_idx
                log_metrics('Training', epoch, current_loss, current_acc, n_iter)

        # 记录训练epoch总结
        train_loss = running_loss / train_total
        train_acc = 100. * train_correct / train_total
        log_metrics('Training', epoch, train_loss, train_acc)
        print(f'\nEpoch {epoch+1}/{num_epochs} - Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

        log_model_stats(model, writer, epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation')
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # 更新验证进度条
                current_val_loss = val_loss / val_total
                current_val_acc = 100. * val_correct / val_total
                pbar_val.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })

        # 记录验证epoch总结
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        log_metrics('Validation', epoch, val_loss, val_acc)
        print(f'\nEpoch {epoch+1}/{num_epochs} - Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')

        # 记录学习率
        try:
            writer.add_scalar('Training/Learning_Rate',
                              optimizer.param_groups[0]['lr'], epoch)
        except Exception as e:
            print(f"Warning: Failed to log learning rate: {str(e)}")

        # 保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')

    writer.close()
    return model


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