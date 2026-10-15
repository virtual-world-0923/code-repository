# 导入tensorflow，用于加载模型、图片张量预处理
import tensorflow as tf
# 导入opencv-python，读取本地图片文件
import cv2
# 导入numpy，数组数值运算
import numpy as np
# 导入os，路径处理
import os

# ===================== 全局配置参数 =====================
# 训练脚本保存的模型文件路径，和train文件统一
MODEL_LOAD_PATH = r"./mnist_mlp_model.keras"
# MNIST标准输入尺寸28*28灰度图
MNIST_IMG_W = 28
MNIST_IMG_H = 28


def preprocess_predict_image(image_file_path):
    """
    本地手写数字图片预处理函数，统一转换成模型要求输入格式
    :param image_file_path: 本地图片完整路径(str)
    :return: 模型可直接预测的一维归一化向量 [1,784]
    """
    # 1. cv2读取图片，参数0代表以灰度模式读取（单通道，匹配MNIST数据集）
    img_gray = cv2.imread(image_file_path, flags=0)
    # 判断图片是否读取成功，文件不存在/损坏则抛出异常
    if img_gray is None:
        raise FileNotFoundError(f"无法读取图片文件：{image_file_path}，检查路径是否正确")

    # 2. 将读取到的图片缩放至标准28*28像素，插值方式双线性
    img_resized = cv2.resize(
        src=img_gray,
        dsize=(MNIST_IMG_W, MNIST_IMG_H),
        interpolation=cv2.INTER_LINEAR
    )

    # 3. 转换数据类型为float32，神经网络浮点计算
    img_float = img_resized.astype(np.float32)
    # 4. 像素归一化0~255 → 0~1，和训练时预处理逻辑完全一致
    img_norm = img_float / 255.0

    # 5. 展平28*28二维矩阵为一维向量784
    img_flat = img_norm.reshape(-1, MNIST_IMG_W * MNIST_IMG_H)
    # 说明：reshape(-1,784)自动补充第一维为1，形状(1,784)适配模型批量输入

    # 返回预处理完成的图片张量数组
    return img_flat


def predict_mnist_number(model, image_path):
    """
    执行单张图片预测，输出预测数字与各类别置信概率
    :param model: 加载完成的keras训练模型
    :param image_path: 本地图片路径
    :return: pred_num(预测数字), pred_probs(各类0-9概率数组)
    """
    # 调用预处理函数，得到符合输入要求的图片向量
    input_tensor = preprocess_predict_image(image_path)
    # 模型前向推理预测，返回长度10的概率数组，总和≈1
    pred_probs = model.predict(x=input_tensor, verbose=1)
    # argmax取出概率最大值对应的索引，即预测数字0-9
    pred_num = np.argmax(pred_probs, axis=1)[0]
    return pred_num, pred_probs


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 第一步：加载训练好的完整模型
    print("正在加载训练完成的MNIST全连接模型...")
    # load_model读取h5模型文件，恢复网络结构、权重、优化器
    loaded_model = tf.keras.models.load_model(MODEL_LOAD_PATH)
    print("模型加载完成，网络结构：")
    # 打印模型结构查看层信息
    loaded_model.summary()

    # ========== 修改此处为你自己的本地手写数字图片路径 ==========
    # 示例：r"D:\test_digit.png"
    test_image_path = r"D:\Repositories\basic_algorithm\database\mnist\2989.png"

    try:
        # 调用预测函数，得到预测数字和概率
        result_digit, probability_array = predict_mnist_number(loaded_model, test_image_path)
        # 打印分割线
        print("=" * 50)
        # 输出最终预测数字
        print(f"图片预测手写数字结果：{result_digit}")
        # 打印每个数字0~9对应的预测置信度，保留4位小数
        print("各数字类别预测概率：")
        for digit in range(10):
            prob = probability_array[0][digit]
            print(f"数字{digit}：{prob:.4f}")
        print("=" * 50)

    # 捕获图片读取失败异常并友好提示
    except FileNotFoundError as err:
        print(f"预测失败：{err}")
        print("请修改代码中 test_image_path 变量为真实存在的手写数字图片路径")
    # 捕获其他未知运行错误
    except Exception as err:
        print(f"程序运行异常：{err}")