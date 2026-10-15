"""
手写数字识别 - 训练脚本
使用sklearn的MLPClassifier实现神经网络训练
"""

import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os
import time


def load_mnist_images(filename):
    """
    加载MNIST图像数据
    参数:
        filename: MNIST图像文件的路径
    返回:
        numpy数组格式的图像数据, shape: (样本数, 784)
    """
    # 以二进制只读模式打开文件
    with open(filename, 'rb') as f:
        # 读取文件头: 魔数(4字节), 图像数量(4字节), 行数(4字节), 列数(4字节)
        # struct.unpack从二进制数据中解析出整数, '>IIII'表示大端序的4个无符号整数
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # 计算每张图像的总像素数
        pixel_count = rows * cols  # 28 * 28 = 784

        # 读取剩余的所有像素数据, 每个像素是1字节(0-255)
        # struct.unpack从二进制数据中解析出指定数量的无符号字节
        image_data = struct.unpack('>' + 'B' * (num_images * pixel_count), f.read())

        # 将一维数组转换为二维数组, 每行是一张图像的像素
        # 转换为float32类型以便后续归一化处理
        images = np.array(image_data, dtype=np.float32).reshape(num_images, pixel_count)

        # 打印加载信息
        print(f"成功加载图像: {filename}, 样本数: {num_images}, 形状: {images.shape}")
        return images


def load_mnist_labels(filename):
    """
    加载MNIST标签数据
    参数:
        filename: MNIST标签文件的路径
    返回:
        numpy数组格式的标签数据, shape: (样本数,)
    """
    # 以二进制只读模式打开文件
    with open(filename, 'rb') as f:
        # 读取文件头: 魔数(4字节), 标签数量(4字节)
        magic, num_labels = struct.unpack('>II', f.read(8))

        # 读取所有标签数据, 每个标签是1字节(0-9)
        label_data = struct.unpack('>' + 'B' * num_labels, f.read())

        # 转换为numpy数组
        labels = np.array(label_data, dtype=np.int32)

        # 打印加载信息
        print(f"成功加载标签: {filename}, 样本数: {num_labels}, 形状: {labels.shape}")
        return labels


def preprocess_data(images):
    """
    数据预处理: 归一化
    参数:
        images: 原始图像数据, 像素值范围0-255
    返回:
        归一化后的图像数据, 像素值范围0-1
    """
    # 将像素值除以255, 归一化到[0,1]区间
    # 这样可以加快训练收敛速度, 提高模型性能
    return images / 255.0


def main():
    """
    主函数: 训练神经网络模型
    """
    print("=" * 60)
    print("开始训练手写数字识别神经网络模型")
    print("=" * 60)

    # 1. 设置数据路径
    # 使用原始字符串或双反斜杠处理Windows路径
    data_dir = r"D:\Repositories\basic_algorithm\database\mnist"

    # 构建完整的文件路径
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    # 2. 加载训练集和测试集数据
    print("\n[1/5] 加载MNIST数据集...")
    print("-" * 40)

    # 加载训练集
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)

    # 加载测试集
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    # 3. 数据预处理 - 归一化
    print("\n[2/5] 数据预处理 - 归一化...")
    print("-" * 40)

    train_images = preprocess_data(train_images)
    test_images = preprocess_data(test_images)

    print(f"训练数据形状: {train_images.shape}")
    print(f"训练标签形状: {train_labels.shape}")
    print(f"测试数据形状: {test_images.shape}")
    print(f"测试标签形状: {test_labels.shape}")

    # 4. 创建并训练神经网络模型
    print("\n[3/5] 创建神经网络模型...")
    print("-" * 40)

    # 使用MLPClassifier创建多层感知机分类器
    # 参数说明:
    #   hidden_layer_sizes: 隐藏层结构, (100,)表示单层100个神经元, 可增加层数如(100,50)
    #   activation: 激活函数, 'relu'是修正线性单元, 常用且效果好
    #   solver: 优化器, 'adam'自适应矩估计, 适合大数据集
    #   alpha: L2正则化系数, 防止过拟合, 0.0001是常用值
    #   batch_size: 批量大小, 'auto'自动选择
    #   learning_rate: 学习率, 'adaptive'自适应调整
    #   learning_rate_init: 初始学习率, 0.001是常用值
    #   max_iter: 最大迭代次数, 200次通常足够
    #   random_state: 随机种子, 保证结果可复现
    #   verbose: 是否输出训练过程信息
    #   early_stopping: 是否使用早停法防止过拟合
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 两层隐藏层: 100个神经元和50个神经元
        activation='relu',  # 使用ReLU激活函数
        solver='adam',  # 使用Adam优化器
        alpha=0.0001,  # L2正则化系数
        batch_size='auto',  # 自动选择批量大小
        learning_rate='adaptive',  # 自适应学习率
        learning_rate_init=0.001,  # 初始学习率
        max_iter=200,  # 最大迭代次数
        random_state=42,  # 随机种子
        verbose=True,  # 输出训练过程
        early_stopping=True,  # 启用早停法
        validation_fraction=0.1,  # 10%数据用于验证
        n_iter_no_change=10  # 连续10次不提升则停止
    )

    print("模型结构:")
    print(f"  - 输入层: 784个神经元 (28x28像素)")
    print(f"  - 隐藏层1: 100个神经元 (ReLU)")
    print(f"  - 隐藏层2: 50个神经元 (ReLU)")
    print(f"  - 输出层: 10个神经元 (Softmax)")
    print(f"  - 正则化系数: 0.0001")
    print(f"  - 最大迭代次数: 200")

    # 5. 训练模型
    print("\n[4/5] 开始训练模型...")
    print("-" * 40)

    # 记录训练开始时间
    start_time = time.time()

    # 训练模型: 使用训练数据和标签进行拟合
    # fit方法会执行前向传播、反向传播和参数更新
    model.fit(train_images, train_labels)

    # 计算训练时间
    training_time = time.time() - start_time
    print(f"训练耗时: {training_time:.2f} 秒")

    # 6. 评估模型性能
    print("\n[5/5] 评估模型性能...")
    print("-" * 40)

    # 在训练集上进行预测
    # predict方法返回预测的类别标签
    train_predictions = model.predict(train_images)
    # 计算训练集准确率: 预测正确的样本数 / 总样本数
    train_accuracy = accuracy_score(train_labels, train_predictions)

    # 在测试集上进行预测
    test_predictions = model.predict(test_images)
    # 计算测试集准确率
    test_accuracy = accuracy_score(test_labels, test_predictions)

    # 输出准确率结果
    print("模型性能评估结果:")
    print(f"  - 训练集准确率: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"  - 测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # 7. 保存模型
    print("\n保存训练好的模型...")
    print("-" * 40)

    # 创建models目录用于存储模型文件
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # 如果目录不存在则创建

    # 模型文件路径
    model_filename = os.path.join(model_dir, "mnist_mlp_model.pkl")

    # 使用joblib保存模型, 比pickle更适合大型numpy数组
    # joblib.dump将模型序列化并保存到文件
    joblib.dump(model, model_filename)

    print(f"模型已保存到: {model_filename}")
    print("=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 返回模型和准确率, 便于其他函数使用
    return model, train_accuracy, test_accuracy


# 如果直接运行此脚本, 执行main函数
if __name__ == "__main__":
    main()