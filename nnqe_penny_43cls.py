import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# 配置参数
n_qubits = 4  # 使用4个量子比特
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def nnqe_circuit(inputs):
    """复杂版的4比特NNQE电路"""
    # 数据编码层
    for i in range(n_qubits):
        qml.RY(2 * np.pi * inputs[i], wires=i)
    
    # 三层纠缠结构
    for _ in range(3):
        # CZ纠缠层
        for i in range(n_qubits-1):
            qml.CZ(wires=[i, i+1])
        # 单量子比特旋转层
        for i in range(n_qubits):
            qml.RY(np.pi/4, wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def process_image_blocks(image):
    """使用2x2滑动窗口处理图像"""
    out = np.zeros((14, 14, 4))  # 与Pennylane QCNN相同的输出shape
    
    # 以2x2的窗口遍历图像
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # 提取2x2块
            block = [
                float(image[j, k, 0]),
                float(image[j, k + 1, 0]),
                float(image[j + 1, k, 0]),
                float(image[j + 1, k + 1, 0])
            ]
            # 使用复杂的4比特量子电路处理
            q_results = nnqe_circuit(block)
            # 存储结果
            for c in range(4):
                out[j // 2, k // 2, c] = float(q_results[c])
    
    return out

def MyModel():
    """与Pennylane QCNN相同的模型结构"""
    model = keras.models.Sequential([
        keras.layers.Input(shape=(14, 14, 4)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(43, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_results(history):
    """绘制训练结果图"""
    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    # 准确率曲线
    ax1.plot(history.history["accuracy"], "-ob", label="Training")
    ax1.plot(history.history["val_accuracy"], "-og", label="Validation")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # 损失曲线
    ax2.plot(history.history["loss"], "-ob", label="Training")
    ax2.plot(history.history["val_loss"], "-og", label="Validation")
    ax2.set_ylabel("Loss")
    ax2.set_ylim(top=2.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('nnqe_training_results.png')
    plt.close()


def main():
    # 加载数据
    print("Loading data...")
    with open("pkls/train_dataset_1k.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open("pkls/test_dataset_100.pkl", 'rb') as f:
        test_data = pickle.load(f)

    # 预处理图像
    print("\nProcessing training images...")
    train_paths = list(train_data['image_paths'].values())
    train_labels = np.array(list(train_data['labels'].values()))
    
    # 加载和预处理训练图像
    train_images = []
    for path in tqdm(train_paths, desc="Loading train images"):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(28, 28), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        train_images.append(img_array)
    train_images = np.array(train_images)

    # 对训练集进行量子处理
    q_train_images = []
    for img in tqdm(train_images, desc="Quantum processing train images"):
        q_features = process_image_blocks(img)
        q_train_images.append(q_features)
    q_train_images = np.array(q_train_images)

    # 同样处理测试集
    print("\nProcessing test images...")
    test_paths = list(test_data['image_paths'].values())
    test_labels = np.array(list(test_data['labels'].values()))
    
    test_images = []
    for path in tqdm(test_paths, desc="Loading test images"):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(28, 28), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        test_images.append(img_array)
    test_images = np.array(test_images)

    q_test_images = []
    for img in tqdm(test_images, desc="Quantum processing test images"):
        q_features = process_image_blocks(img)
        q_test_images.append(q_features)
    q_test_images = np.array(q_test_images)

    # 训练模型
    print("\nTraining model...")
    model = MyModel()
    history = model.fit(
        q_train_images, train_labels,
        validation_data=(q_test_images, test_labels),
        batch_size=32,
        epochs=80,
        verbose=1
    )
    
    # 绘制结果
    plot_results(history)
    
    # 评估最终结果
    test_loss, test_accuracy = model.evaluate(q_test_images, test_labels, verbose=0)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # 保存模型
    model.save('self_model_batch.h5')

if __name__ == "__main__":
    main()