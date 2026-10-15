# ============================================================================
# 文件名：predict_alexnet.py
# 功能：加载训练好的AlexNet模型，读取本地图片进行分类预测
# 环境：Python 3.13.14 / Keras 3.15.0 / TensorFlow 2.21.0 / NumPy 2.5.1
# ============================================================================

# 导入numpy库，用于图像数组操作和预处理
import numpy as np
# 导入os模块，用于判断文件是否存在、获取文件路径等
import os
# 导入keras库（Keras 3.x独立包）
import keras
# 从keras.models中导入load_model函数，用于加载保存的.keras模型文件
from keras.models import load_model
# 导入PIL库中的Image模块，用于读取本地图片文件（支持jpg、png、bmp等格式）
# Pillow是Python图像处理标准库，需提前安装：pip install Pillow
from PIL import Image

# ============================================================================
# 第一部分：定义CIFAR-10类别名称映射
# ============================================================================

# CIFAR-10的10个类别名称列表，索引0-9分别对应以下类别
# 这个顺序与CIFAR-10官方定义一致
CLASS_NAMES = [
    'airplane',      # 索引0：飞机
    'automobile',    # 索引1：汽车
    'bird',          # 索引2：鸟
    'cat',           # 索引3：猫
    'deer',          # 索引4：鹿
    'dog',           # 索引5：狗
    'frog',          # 索引6：青蛙
    'horse',         # 索引7：马
    'ship',          # 索引8：船
    'truck',         # 索引9：卡车
]

# ============================================================================
# 第二部分：定义图像预处理函数
# ============================================================================

def preprocess_image(image_path, target_size=(32, 32)):
    """
    功能：读取本地图片并进行预处理，使其适配模型输入要求
    参数：
        image_path (str): 本地图片文件的完整路径，如 'D:\\images\\test.png'
        target_size (tuple): 目标尺寸(高, 宽)，默认(32, 32)与CIFAR-10一致
    返回：
        img_array (numpy.ndarray): 预处理后的图像数组，形状(1, 32, 32, 3)，float32类型
                                   第一维为batch维度（值为1表示单张图片）
    异常：
        FileNotFoundError: 当图片路径不存在时抛出
    """
    # 检查文件是否存在，不存在则抛出异常并给出提示
    if not os.path.exists(image_path):
        # 抛出文件未找到异常，提示用户检查路径
        raise FileNotFoundError(f'图片文件不存在: {image_path}')

    # 使用PIL的Image.open打开图片文件
    # 返回一个PIL Image对象，包含图像的像素数据和元信息
    img = Image.open(image_path)

    # 将图片转换为RGB模式（3通道）
    # 有些图片可能是RGBA(4通道)、L(灰度)或P(调色板)模式
    # 转换为RGB确保输入为3通道，与模型期望的输入一致
    img = img.convert('RGB')

    # 将图片缩放到目标尺寸(32, 32)
    # Image.BILINEAR是双线性插值方法，缩放效果平滑
    # 在Pillow 10+中推荐使用Image.Resampling.BILINEAR
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # 将PIL Image对象转换为numpy数组
    # 转换后形状为(32, 32, 3)，数据类型为uint8(0-255)
    img_array = np.array(img)

    # 将像素值从uint8(0-255)转换为float32，并归一化到[0, 1]
    # 与训练时的预处理保持一致，确保模型输入分布相同
    img_array = img_array.astype('float32') / 255.0

    # 扩展batch维度：将形状从(32, 32, 3)变为(1, 32, 32, 3)
    # 模型预测时需要4D输入(batch_size, height, width, channels)
    # np.expand_dims在axis=0位置插入一个大小为1的维度
    img_array = np.expand_dims(img_array, axis=0)

    # 返回预处理后的图像数组
    return img_array


# ============================================================================
# 第三部分：定义预测函数
# ============================================================================

def predict_image(model, image_path):
    """
    功能：使用加载的模型对单张图片进行分类预测
    参数：
        model (keras.Model): 已加载的Keras模型对象
        image_path (str): 待预测图片的完整路径
    返回：
        predicted_class (str): 预测的类别名称字符串
        confidence (float): 预测置信度（概率值，0-1之间）
        probabilities (numpy.ndarray): 所有类别的概率分布，形状(10,)
    """
    # 调用预处理函数，将图片转换为模型可接受的输入格式
    # 返回形状为(1, 32, 32, 3)的float32数组
    img_array = preprocess_image(image_path)

    # 调用模型的predict方法进行推理
    # 输入：形状(1, 32, 32, 3)的数组
    # 输出：形状(1, 10)的数组，每行是10个类别的概率分布（softmax输出）
    # verbose=0表示不打印预测进度
    predictions = model.predict(img_array, verbose=0)

    # 获取概率最大的类别索引
    # predictions[0]取出第一张（也是唯一一张）图片的预测结果，形状(10,)
    # np.argmax返回最大值的索引位置（0-9之间的整数）
    predicted_index = np.argmax(predictions[0])

    # 根据索引获取对应的类别名称
    predicted_class = CLASS_NAMES[predicted_index]

    # 获取该类别的置信度（即最大概率值）
    # predictions[0][predicted_index]取出预测类别对应的概率
    confidence = float(predictions[0][predicted_index])

    # 获取完整的概率分布（所有10个类别的概率）
    probabilities = predictions[0]

    # 返回预测类别名称、置信度和完整概率分布
    return predicted_class, confidence, probabilities


# ============================================================================
# 第四部分：主程序——加载模型并预测
# ============================================================================

if __name__ == '__main__':
    # 定义训练好的模型文件路径（与train_alexnet.py中保存路径一致）
    MODEL_PATH = 'alexnet_cifar10.keras'

    # 定义待预测的本地图片路径（用户需修改为实际图片路径）
    # 支持常见图片格式：.png、.jpg、.jpeg、.bmp等
    # 示例路径，请替换为实际图片路径
    IMAGE_PATH = r'D:\Repositories\basic_algorithm\database\test_image.png'

    # ---------- 步骤1：检查模型文件是否存在 ----------
    # 打印分隔线和提示信息
    print('=' * 60)
    print('步骤1：加载训练好的模型...')
    print('=' * 60)
    # 使用os.path.exists检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        # 如果模型文件不存在，打印错误信息并退出程序
        print(f'错误：模型文件不存在: {MODEL_PATH}')
        print('请先运行 train_alexnet.py 训练并保存模型！')
        # 调用exit()终止程序执行
        exit()

    # ---------- 步骤2：加载模型 ----------
    # 使用keras.models.load_model加载.keras格式的模型文件
    # 该函数会恢复模型结构、权重和优化器状态
    # 参数：模型文件路径字符串
    model = load_model(MODEL_PATH)
    # 打印模型加载成功提示
    print(f'模型加载成功: {MODEL_PATH}')
    # 打印模型名称（在build_alexnet中定义为'AlexNet_CIFAR10'）
    print(f'模型名称: {model.name}')

    # ---------- 步骤3：检查待预测图片是否存在 ----------
    # 打印分隔线和提示信息
    print('\n' + '=' * 60)
    print('步骤2：读取并预测图片...')
    print('=' * 60)
    # 检查图片文件是否存在
    if not os.path.exists(IMAGE_PATH):
        # 图片不存在时打印提示，并给出使用说明
        print(f'提示：图片文件不存在: {IMAGE_PATH}')
        print('请将 IMAGE_PATH 变量修改为实际的图片路径后重新运行。')
        print('支持的图片格式：.png, .jpg, .jpeg, .bmp')
        # 终止程序
        exit()

    # ---------- 步骤4：执行预测 ----------
    # 打印待预测图片的路径
    print(f'待预测图片: {IMAGE_PATH}')
    # 调用predict_image函数进行预测
    # 返回：预测类别名称、置信度、所有类别概率分布
    predicted_class, confidence, probabilities = predict_image(model, IMAGE_PATH)

    # ---------- 步骤5：输出预测结果 ----------
    # 打印分隔线
    print('\n' + '-' * 60)
    # 打印预测的类别名称
    print(f'预测类别: {predicted_class}')
    # 打印预测置信度，格式化为百分比（保留2位小数）
    print(f'置信度:   {confidence * 100:.2f}%')
    # 打印分隔线
    print('-' * 60)

    # 打印所有类别的概率分布，便于用户查看模型对各类别的判断
    print('\n各类别概率分布:')
    # 遍历10个类别，逐一打印类别名称和对应概率
    for i in range(len(CLASS_NAMES)):
        # 获取当前类别名称
        class_name = CLASS_NAMES[i]
        # 获取当前类别的预测概率
        prob = probabilities[i]
        # 格式化输出：类别名称左对齐12字符，概率以百分比显示保留2位小数
        print(f'  {class_name:<12s}: {prob * 100:.2f}%')

    # 打印完成提示
    print('\n预测完成！')