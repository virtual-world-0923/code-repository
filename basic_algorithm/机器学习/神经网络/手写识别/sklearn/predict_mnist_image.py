# 导入numpy用于数组处理、维度转换、归一化
import numpy as np
# joblib用于读取保存好的sklearn训练模型
import joblib
# PIL.Image用来打开、读取、缩放、灰度化本地图片
from PIL import Image

def preprocess_image(image_path):
    """
    函数功能：预处理本地手写数字图片，转换成和MNIST数据集格式一致的输入
    MNIST标准：28×28像素、灰度图、数字白色背景黑色，像素0~1
    参数：
        image_path: str 本地图片文件路径
    返回：
        img_vector: (1,784) 二维数组，符合模型输入格式
    """
    # 打开图片文件
    img = Image.open(image_path)
    # 将图片转为灰度模式 "L" = 单通道灰度图
    img = img.convert("L")
    # 缩放图片尺寸到标准28×28像素，匹配MNIST数据集大小
    img = img.resize((28, 28))

    # 将图片对象转为numpy数组，数值范围0~255 uint8
    img_array = np.array(img)

    # 【重要】MNIST数据集：数字线条是黑色(0)，背景白色(255)
    # 如果你的图片白底黑字不需要反转；黑底白字需要下面这行解除注释
    # img_array = 255 - img_array

    # 归一化像素值到 [0,1]，和训练时处理方式保持一致
    img_array = img_array / 255.0

    # 将28×28二维矩阵展平为一维向量 784
    img_flat = img_array.flatten()
    # 转换成二维数组 [样本数量,特征数]，sklearn模型输入必须是二维
    img_vector = img_flat.reshape(1, -1)

    return img_vector

# ----------------主程序入口----------------
if __name__ == "__main__":
    # 模型文件路径，和训练脚本保存名称保持一致
    model_file = "mnist_mlp_model.pkl"
    # 填写你要预测的手写数字图片路径！自行修改
    test_image_path = r"D:\Repositories\basic_algorithm\database\mnist\2989.png"

    # joblib加载本地保存的训练模型
    model = joblib.load(model_file)
    print("模型加载成功！")

    # 调用预处理函数，处理图片得到模型可识别特征向量
    input_data = preprocess_image(test_image_path)

    # 使用模型进行预测，predict返回预测标签
    pred_result = model.predict(input_data)
    # predict_proba 输出每个类别0-9的概率
    pred_proba = model.predict_proba(input_data)

    # 输出预测数字
    print(f"预测手写数字为：{pred_result[0]}")
    # 输出该数字对应的置信概率
    print(f"预测置信度：{np.max(pred_proba):.4f}")