"""
手写数字识别 - 预测脚本
加载训练好的模型, 对本地图片进行数字识别
"""

import numpy as np
from PIL import Image
import joblib
import os
import sys


def load_model(model_path):
    """
    加载训练好的模型
    参数:
        model_path: 模型文件的路径
    返回:
        加载的模型对象
    """
    # 使用joblib加载之前保存的模型文件
    # joblib.load从文件中反序列化模型对象
    model = joblib.load(model_path)
    print(f"成功加载模型: {model_path}")
    return model


def preprocess_image(image_path):
    """
    预处理本地图片, 转换为模型可接受的格式
    参数:
        image_path: 图片文件路径
    返回:
        预处理后的图像数据, shape: (1, 784), 值范围0-1
    """
    try:
        # 使用PIL库打开图片
        # Image.open读取图片文件并返回Image对象
        img = Image.open(image_path)

        # 转换为灰度图
        # convert('L')将图片转换为灰度模式, 每个像素值0-255
        img = img.convert('L')

        # 调整大小到28x28像素 (MNIST标准尺寸)
        # resize方法调整图像大小, (28, 28)指定目标尺寸
        # Image.Resampling.LANCZOS是高质量的重采样滤波器
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 反色处理: MNIST是白底黑字, 如果图片是黑底白字则需反色
        # 将像素值转换为numpy数组, dtype=np.float32保证精度
        img_array = np.array(img, dtype=np.float32)

        # 检查图片是否为白底黑字
        # 计算黑色像素的平均值, 如果黑像素较多说明是白底黑字
        # 否则是黑底白字, 需要反色
        if np.mean(img_array) > 127:  # 如果平均亮度高, 说明是白底
            # 反色: 255 - 原像素值, 将白底黑字转为黑底白字
            img_array = 255 - img_array

        # 归一化: 将像素值从[0,255]缩放到[0,1]
        # 这步与训练时的预处理保持一致
        img_array = img_array / 255.0

        # 将二维图像(28,28)展平为一维数组(784,)
        # reshape(-1)将数组展开成一维
        img_flat = img_array.reshape(-1)

        # 将一维数组调整为二维数组(1,784), 因为模型需要批量输入
        # 这里只有一张图片, 所以第一维是1
        img_flat = img_flat.reshape(1, -1)

        print(f"图片预处理完成: {image_path}")
        print(f"  原始尺寸: {img.size[0]}x{img.size[1]}")
        print(f"  处理后形状: {img_flat.shape}")

        return img_flat

    except FileNotFoundError:
        print(f"错误: 找不到图片文件 '{image_path}'")
        return None
    except Exception as e:
        print(f"图片处理出错: {e}")
        return None


def predict_digit(model, image_data):
    """
    使用模型预测数字
    参数:
        model: 训练好的模型
        image_data: 预处理后的图像数据, shape: (1, 784)
    返回:
        预测的数字 (0-9)
    """
    # 使用模型的predict方法进行预测
    # 返回预测的类别标签
    prediction = model.predict(image_data)

    # 获取预测的类别 (第一个样本的预测结果)
    predicted_digit = prediction[0]

    # 如果需要获取预测概率, 可以使用predict_proba方法
    # proba = model.predict_proba(image_data)
    # 这返回每个类别的概率分布

    return predicted_digit


def predict_with_confidence(model, image_data):
    """
    使用模型预测数字并返回置信度
    参数:
        model: 训练好的模型
        image_data: 预处理后的图像数据, shape: (1, 784)
    返回:
        预测的数字和置信度
    """
    # 获取预测类别
    prediction = model.predict(image_data)
    predicted_digit = prediction[0]

    # 获取各类别的概率分布
    # predict_proba返回形状为(样本数, 类别原始分数)的数组
    probabilities = model.predict_proba(image_data)

    return predicted_digit, probabilities

def main():
    """
    主函数: 加载模型并预测图片
    """
    print("=" * 60)
    print("手写数字识别 - 预测程序")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/2] 加载训练好的模型...")
    print("-" * 40)

    # 模型文件路径
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "mnist_mlp_model.pkl")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 '{model_path}'")
        print("请先运行 train_mnist.py 训练模型")
        return

    # 加载模型
    model = load_model(model_path)
    print("模型信息:")
    print(f"  - 模型类型: {type(model).__name__}")
    print(f"  - 模型类: {model.__class__.__module__}")

    # 3. 预处理图片并进行预测
    print("\n[2/2] 进行数字识别...")
    print("-" * 40)

    image_path = r"D:\Repositories\basic_algorithm\database\mnist\2989.png"
    # 预处理图片
    image_data = preprocess_image(image_path)

    if image_data is None:
        print("图片预处理失败, 程序退出")
        return

    # 进行预测 (获取数字和置信度)
    predicted_digit, probabilities = predict_with_confidence(model, image_data)

    # 输出预测结果
    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"  预测数字: {predicted_digit}")
    print(f"  原始分数: {probabilities}")
    print("\n各类别概率分布:")
    print("-" * 40)

    # 显示所有类别的概率
    for digit in range(10):
        prob = probabilities[0][digit] * 100
        # 用柱状图字符表示概率大小
        bar_length = int(prob / 2)  # 缩放以便显示
        bar = '█' * bar_length
        # 标记最高概率
        marker = " ← 预测" if digit == predicted_digit else ""
        print(f"  数字 {digit}: {prob:6.2f}% {bar}{marker}")

    print("=" * 60)


# 如果直接运行此脚本, 执行main函数
if __name__ == "__main__":
    main()