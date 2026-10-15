# -*- coding: utf-8 -*-
"""
CNN 手写数字识别 —— 加载模型 & 预测本地图片
用法：python predict_mnist.py [图片路径]
      若不指定路径则默认读取 test_digit.png
"""

# ======================== 导入依赖 ========================
import sys                                # 用于读取命令行参数 sys.argv
import os                                 # 用于检查文件是否存在
import numpy as np                        # 用于数组维度扩展、argmax 等数值操作
import keras                              # Keras 3 主模块，提供模型加载、图片工具等 API


# ======================== 1. 加载已训练的模型 ========================

model_path = 'mnist_cnn_model.keras'                          # 与 train_mnist.py 保存路径保持一致

if not os.path.isfile(model_path):                             # 检查模型文件是否存在于当前工作目录
    raise FileNotFoundError(                                   # 若模型文件不存在则抛出异常
        f"模型文件未找到: {model_path}，请先运行 train_mnist.py 完成训练并保存模型。"
    )

model = keras.saving.load_model(model_path)                    # 使用 Keras 3 的 saving 模块加载 .keras 格式模型（恢复架构+权重+优化器状态）

print("模型加载成功！")                                         # 提示模型加载成功
model.summary()                                                # 打印模型结构摘要，确认各层形状和参数量与训练时一致


# ======================== 2. 获取待预测图片路径 ========================

image_path = r"/database/mnist/MNIST_Sampled_Data/843.png"

if not os.path.isfile(image_path):                             # 检查图片文件是否存在
    raise FileNotFoundError(                                   # 若不存在则抛出异常
        f"图片文件未找到: {image_path}\n"
        f"请通过命令行指定正确的图片路径，例如: python predict_mnist.py path/to/digit.png"
    )

print(f"待预测图片: {image_path}")                              # 打印即将读取的图片路径


# ======================== 3. 读取图片并预处理 ========================

# 使用 Keras 内置工具加载图片（底层调用 Pillow）
img = keras.utils.load_img(                                    # 加载图片文件并返回 PIL Image 对象
    path=image_path,                                            # 参数 path：图片文件路径
    color_mode='grayscale',                                     # 参数 color_mode：以灰度模式加载，输出单通道图像
    target_size=(28, 28)                                        # 参数 target_size：将图片缩放到 (28, 28)，与 MNIST 训练数据尺寸一致
)

# 将 PIL Image 转为 numpy 数组
img_array = keras.utils.img_to_array(                          # 将 PIL Image 对象转换为 numpy 数组
    img                                                          # 参数 img：PIL Image 对象
)                                                                # 返回形状为 (28, 28, 1) 的 float32 数组，像素值范围 [0, 255]

print(f"图片数组形状: {img_array.shape}")                        # 应输出 (28, 28, 1)
print(f"像素值范围: [{img_array.min():.0f}, {img_array.max():.0f}]")  # 应输出 [0, 255]

# ---- 提示：若预测不准，可尝试取消下面一行的注释来反转颜色 ----
# MNIST 训练数据为白字黑底，若你的图片是黑字白底则需要反转
# img_array = 255.0 - img_array                                # 反转像素值：白底黑字 → 黑底白字，与 MNIST 风格一致

# 归一化像素值到 [0.0, 1.0]，与训练时的预处理保持一致
img_array = img_array.astype(np.float32) / 255.0               # 将像素值从 [0, 255] 线性映射到 [0.0, 1.0]，float32 类型

# 添加 batch 维度：模型期望输入形状为 (batch, H, W, C)，单张图片需扩展第 0 维
img_array = np.expand_dims(                                    # 在指定轴上扩展数组维度
    img_array,                                                   # 参数 a：待扩展的数组，当前形状 (28, 28, 1)
    axis=0                                                       # 参数 axis：在第 0 维（batch 维度）插入新维度
)                                                                # 结果形状：(1, 28, 28, 1)

print(f"预处理后形状: {img_array.shape}")                        # 应输出 (1, 28, 28, 1)


# ======================== 4. 执行预测 ========================

predictions = model.predict(                                   # 使用模型对输入数据进行前向推理，输出各类别概率
    x=img_array,                                                # 参数 x：输入数据，形状 (1, 28, 28, 1)
    verbose=0                                                    # 参数 verbose：静默模式，不打印进度条
)                                                                # 返回形状 (1, 10) 的二维数组，每行为 10 个类别的概率值

# 提取预测结果
predicted_class = np.argmax(predictions[0])                    # 取概率最大的类别索引作为预测数字（0~9）
confidence = np.max(predictions[0]) * 100                      # 取最大概率值并转为百分比，作为预测置信度


# ======================== 5. 输出预测结果 ========================

print("\n" + "=" * 60)                                         # 分隔线
print("【预测结果】")                                           # 提示信息
print(f"  预测数字: {predicted_class}")                          # 打印预测的数字
print(f"  置信度:   {confidence:.2f}%")                          # 打印置信度百分比，保留两位小数

# 打印所有 10 个类别的概率分布（带可视化柱状条）
print("\n【各类别概率分布】")                                     # 提示信息
for i in range(10):                                            # 遍历 0~9 共 10 个数字类别
    prob = predictions[0][i] * 100                             # 将第 i 个类别的概率从比例转为百分比
    bar_len = int(prob / 2)                                    # 将百分比除以 2 得到柱状条字符数（50% 对应 25 个方块，满格）
    bar = '█' * bar_len                                        # 用全方块字符 '█' 重复 bar_len 次生成可视化柱状条
    # 标记预测结果为箭头
    marker = ' ◄── 预测' if i == predicted_class else ''        # 对预测类别添加标记
    print(f"  数字 {i}: {prob:6.2f}% {bar}{marker}")            # 格式化输出每个数字的概率和柱状图