# ============================================================
# 文件名: predict_mnist.py
# 功能描述: 加载训练好的CNN模型，读取本地手写数字图片，预测其代表的数字
# 环境要求: Python 3.13.14, Keras 3.15.0, NumPy 2.5.1
# 使用前提: 需先运行train_mnist.py生成模型文件mnist_cnn_model.keras
# ============================================================

# 导入numpy库，用于图像数组的数值计算和形状变换操作
import numpy as np
# 导入keras库，用于加载已保存的模型文件和进行推理预测
import keras
# 导入Python标准库os模块，用于判断文件是否存在和路径操作
import os
# 导入Python标准库sys模块，用于在命令行参数缺失时给出提示并退出程序
import sys
# 导入PIL库(Pillow)用于读取本地图片文件
# Pillow是Python图像处理库，支持读取png、jpg、bmp等多种格式
from PIL import Image

# ============================================================
# 第二部分：定义图片预处理函数
# ============================================================

def preprocess_image(image_path):
    """
    函数功能: 读取本地图片文件，进行预处理使其符合CNN模型的输入要求
    参数说明:
        image_path (str): 本地图片文件的完整路径，支持png/jpg/bmp等常见格式，
                          图片内容应为手写数字（建议白字黑底或黑字白底均可）
    返回值:
        numpy.ndarray: 形状为(1, 28, 28, 1)的float32数组，
                       第一个维度1表示批次大小（单张图片），
                       28x28是MNIST标准尺寸，1是灰度通道数
    异常情况:
        如果文件不存在会打印错误信息并返回None
    """
    # 使用os.path.exists()检查图片文件是否存在
    if not os.path.exists(image_path):
        # 文件不存在时打印错误信息，包含具体路径方便用户排查
        print(f"错误: 图片文件不存在，路径: {image_path}")
        # 返回None表示预处理失败
        return None

    # 使用PIL的Image.open()方法打开图片文件，返回PIL Image对象
    # 参数image_path: 图片文件的完整路径字符串
    img = Image.open(image_path)

    # 将图片转换为灰度模式("L"模式):
    #   "L"表示8位灰度图，每个像素用0-255的整数表示亮度
    #   如果原图是RGB彩色图，会自动按加权公式转换: L = 0.299*R + 0.587*G + 0.114*B
    #   MNIST数据是灰度图，所以模型输入也必须是单通道灰度图
    img = img.convert("L")

    # 使用PIL的resize()方法将图片缩放到28x28像素:
    #   (28, 28): 目标尺寸元组，(宽度, 高度)，与MNIST训练数据尺寸一致
    #   Image.LANCZOS: 重采样滤波器，使用高质量Lanczos插值算法，
    #                  缩放效果比默认的NEAREST或BILINEAR更平滑
    img = img.resize((28, 28), Image.LANCZOS)

    # 使用numpy的array()方法将PIL Image对象转换为numpy数组:
    #   转换后形状为(28, 28)，数据类型为uint8，像素值范围[0, 255]
    img_array = np.array(img)

    # 将数组数据类型转换为float32，并除以255.0归一化到[0, 1]区间:
    #   与训练时的预处理保持一致，确保模型输入数据分布相同
    #   astype("float32"): 类型转换，神经网络计算需要浮点数
    #   / 255.0: 归一化操作，将[0,255]映射到[0.0, 1.0]
    img_array = img_array.astype("float32") / 255.0

    # 使用numpy的expand_dims在最后一个轴添加通道维度:
    #   axis=-1: 在最后添加新轴，形状从(28,28)变为(28,28,1)
    #   1表示灰度图的单通道，与Conv2D层要求的输入格式(height, width, channels)匹配
    img_array = np.expand_dims(img_array, axis=-1)

    # 再次使用expand_dims在第一个轴添加批次维度:
    #   axis=0: 在最前面添加新轴，形状从(28,28,1)变为(1,28,28,1)
    #   1表示批次大小为1（单张图片），model.predict()要求输入必须是批次格式
    img_array = np.expand_dims(img_array, axis=0)

    # 返回预处理完成的数组，形状(1, 28, 28, 1)，可直接传入model.predict()
    return img_array


# ============================================================
# 第三部分：定义预测函数
# ============================================================

def predict_digit(model, image_path):
    """
    函数功能: 对单张本地图片进行手写数字预测，返回预测结果和各类别概率
    参数说明:
        model (keras.Model): 已加载的Keras CNN模型对象，用于执行前向推理
        image_path (str): 待预测图片的本地完整路径
    返回值:
        tuple: (predicted_digit, confidence, probabilities)
            predicted_digit (int): 预测的数字类别(0-9)
            confidence (float): 预测的置信度（最大概率值），范围[0,1]
            probabilities (numpy.ndarray): 长度为10的概率数组，每个元素对应数字0-9的预测概率
        如果图片预处理失败则返回None
    """
    # 调用preprocess_image函数对图片进行预处理，返回形状(1,28,28,1)的数组
    img_array = preprocess_image(image_path)

    # 检查预处理结果是否为None（文件不存在等情况）
    if img_array is None:
        # 预处理失败，返回None表示无法预测
        return None

    # 调用model.predict()方法对输入图片进行前向推理（预测）:
    #   x=img_array: 输入数据，形状(1, 28, 28, 1)，float32类型
    #   verbose=0: 不显示预测进度信息，静默执行
    #   返回值: 形状(1, 10)的numpy数组，每行是10个类别的softmax概率分布
    predictions = model.predict(x=img_array, verbose=0)

    # 使用numpy的argmax()获取概率数组中最大值的索引:
    #   predictions[0]: 取批次中第一张（也是唯一一张）图片的预测结果，形状(10,)
    #   np.argmax(): 返回最大概率值对应的索引，即预测的数字类别(0-9)
    predicted_digit = int(np.argmax(predictions[0]))

    # 获取预测类别对应的置信度（最大概率值）:
    #   predictions[0][predicted_digit]: 取出预测类别对应的概率值
    #   float(): 将numpy浮点数转换为Python原生float类型
    confidence = float(predictions[0][predicted_digit])

    # 获取完整的10个类别概率分布，用于展示各类别的可能性
    probabilities = predictions[0]

    # 返回预测数字、置信度和完整概率分布的元组
    return predicted_digit, confidence, probabilities


# ============================================================
# 第四部分：主程序入口 - 执行预测
# ============================================================

# 使用if __name__ == "__main__"确保以下代码仅在直接运行此脚本时执行，
# 被其他模块import时不会执行，这是Python的标准实践
if __name__ == "__main__":
    # ============================================================
    # 加载训练好的模型
    # ============================================================

    # 定义训练好的模型文件路径，与train_mnist.py中保存路径一致
    # 使用原始字符串r""避免Windows路径中反斜杠的转义问题
    model_path = r".\mnist_cnn_model.keras"

    # 使用os.path.exists()检查模型文件是否存在于指定路径
    if not os.path.exists(model_path):
        # 如果模型文件不存在，打印错误信息提示用户先运行训练脚本
        print(f"错误: 模型文件不存在，路径: {model_path}")
        # 提示用户需要先运行train_mnist.py来生成模型文件
        print("请先训练并保存模型")
        # 终止程序执行
        sys.exit(1)

    # 使用keras.saving.load_model()加载完整的模型文件:
    #   filepath: 模型文件的完整路径（.keras格式）
    #   返回值: 完整的Keras Model对象，包含架构、权重和编译配置，可直接用于预测
    #   注意: Keras 3中使用keras.saving.load_model()而非旧版的keras.models.load_model()
    model = keras.saving.load_model(model_path)

    # 打印模型加载成功的提示信息
    print(f"模型加载成功: {model_path}")
    # 调用model.summary()打印模型结构，验证加载的模型架构是否正确
    model.summary()

    # 定义待预测图片的路径（用户需修改为实际的本地图片路径）
    # 示例路径：用户应替换为自己保存的手写数字图片路径
    # 支持常见图片格式: .png, .jpg, .jpeg, .bmp, .tiff等
    test_image_path = r"/database/mnist/MNIST_Sampled_Data/843.png"

    # 打印分隔线和提示信息，表示开始预测
    print(f"\n{'='*50}")
    print(f"待预测图片路径: {test_image_path}")
    print(f"{'='*50}")

    # 调用predict_digit函数执行预测:
    #   model: 前面加载的Keras CNN模型
    #   test_image_path: 待预测图片的路径
    result = predict_digit(model, test_image_path)

    # 检查预测结果是否为None（图片文件不存在或预处理失败）
    if result is None:
        # 预测失败，打印错误提示
        print("预测失败: 无法处理该图片，请检查文件路径和图片格式")
    else:
        # 预测成功，解包返回的元组为三个变量:
        #   predicted_digit: 预测的数字(0-9)
        #   confidence: 预测置信度(0-1之间的浮点数)
        #   probabilities: 10个类别的概率数组
        predicted_digit, confidence, probabilities = result

        # 打印预测结果：识别出的数字
        print(f"\n预测结果: 数字 {predicted_digit}")
        # 打印预测置信度，格式化为百分比形式，保留2位小数
        print(f"置信度: {confidence * 100:.2f}%")

        # 打印分隔线
        print(f"\n各类别预测概率分布:")
        print(f"{'-'*30}")

        # 使用for循环遍历0-9共10个数字类别，逐一打印每个类别的预测概率
        for digit in range(10):
            # 获取当前数字类别的预测概率值
            prob = probabilities[digit]
            # 生成概率条形图: 用"█"字符表示概率大小，每2%概率对应一个方块
            #   int(prob * 50): 将概率(0-1)映射到0-50个方块字符
            bar = "█" * int(prob * 50)
            # 格式化打印: 数字标签、概率百分比(保留2位小数)、概率条形图
            print(f"  数字 {digit}: {prob * 100:6.2f}% |{bar}")

        # 打印结束分隔线
        print(f"{'='*50}\n")