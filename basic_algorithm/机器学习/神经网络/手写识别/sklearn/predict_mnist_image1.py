# 导入用于处理图像数据的 OpenCV 库
import cv2
# 导入用于处理数值数组的 numpy 库
import numpy as np
# 导入 joblib 库，用于从本地文件加载之前保存的模型
import joblib

# 使用 joblib 加载之前训练并保存的 MLP 模型文件
mlp_model = joblib.load('mnist_mlp_model.pkl')

# 定义本地待预测图片的路径 (请确保该路径下有一张名为 'test_digit.png' 的图片)
image_path = r'D:\Repositories\basic_algorithm\database\mnist\2989.png'

# 使用 OpenCV 读取本地图片，参数 0 表示以灰度模式 (Grayscale) 读取，去除颜色信息
img = cv2.imread(image_path, 0)

# 检查图片是否成功读取：如果图片路径错误或文件不存在，imread 会返回 None
if img is None:
    # 如果读取失败，在控制台打印错误提示信息
    print(f"错误：无法读取图片 '{image_path}'，请检查文件路径是否正确。")
# 如果图片读取成功，则继续执行以下预测逻辑
else:
    # 使用 OpenCV 将图片尺寸强制调整为 28x28 像素，因为 MNIST 模型只接受 28x28 大小的输入
    img_resized = cv2.resize(img, (28, 28))

    # 对调整大小后的图像进行归一化处理：将像素值除以 255.0，使其范围变为 0-1，与训练时的数据处理保持一致
    img_normalized = img_resized / 255.0

    # 将二维图像 (28, 28) 展平为一维向量 (784,)，以匹配 MLP 模型的输入维度要求
    img_flat = img_normalized.reshape(-1)

    # 使用 reshape(1, -1) 将一维向量增加一个批次维度，变成 (1, 784) 的二维数组，因为模型 predict 方法期望输入是二维的
    img_batch = img_flat.reshape(1, -1)

    # 调用模型的 predict 方法，传入处理好的单张图片数据，获取模型预测的数字类别
    prediction = mlp_model.predict(img_batch)

    # 在控制台打印输出模型预测的结果，prediction[0] 表示取出批次中的第一个（也是唯一一个）预测值
    print(f"模型预测该图片上的数字是: {prediction[0]}")