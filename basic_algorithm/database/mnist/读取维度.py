import struct
import numpy as np
import random


def read_mnist_images(filepath):
    """读取 MNIST 图像文件"""
    with open(filepath, 'rb') as f:
        # 读取文件头 (魔数, 数量, 行数, 列数)
        magic, num_items, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for image file {filepath}: {magic}")

        # 读取图像数据并转换为 numpy 数组
        # 原始形状: (样本数, 行数, 列数) -> (N, H, W)
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
    return data


def read_mnist_labels(filepath):
    """读取 MNIST 标签文件"""
    with open(filepath, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for label file {filepath}: {magic}")

        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 读取训练集数据 (你也可以改为 t10k 测试集)
    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'

    try:
        images = read_mnist_images(train_images_path)
        labels = read_mnist_labels(train_labels_path)

        print("=" * 60)
        print("📊 数据集整体概览")
        print("=" * 60)
        print(f"总条数 (Total Samples): {images.shape[0]}")
        print(f"原始数据维度 (Raw Shape): {images.shape[1:]} (即: 高 x 宽)")
        print(f"数据类型 (Data Type): {images.dtype}")
        print(f"像素值范围: {images.min()} - {images.max()}")

        # 2. 随机选择 3 个索引
        total_count = images.shape[0]
        random_indices = random.sample(range(total_count), 3)

        print("\n" + "=" * 60)
        print("🎲 随机抽取的 3 条数据详情")
        print("=" * 60)

        for idx in random_indices:
            raw_img = images[idx]
            label = labels[idx]

            # 手动添加通道维度 (模拟深度学习中的 C-H-W 格式)
            # MNIST 是灰度图，所以通道数 C=1
            img_with_channel = np.expand_dims(raw_img, axis=0)

            print(f"\n--- 第 {idx} 条数据 ---")
            print(f"【标签】: {label}")
            print(f"【原始数据维度】: {raw_img.shape} (H x W)")
            print(f"【通道维度】: 1 (灰度图)")
            print(f"【带通道的完整维度】: {img_with_channel.shape} (C x H x W)")

            print("【原始像素数据 (28x28 矩阵)】:")
            print(raw_img)

    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件。请确保代码目录下有 '{train_images_path}' 和 '{train_labels_path}'")
        print(e)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")