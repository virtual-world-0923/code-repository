# ============================================================
# 文件名：predict.py
# 功能：加载训练好的CNN模型，读取本地图片文件，预测图片中的手写数字
# 环境：Python 3.13.14 / PyTorch 2.13.0+cpu / numpy 2.5.1
# 依赖：需要安装Pillow库用于读取图片 (pip install Pillow)
# ============================================================

# 导入Python标准库中的os模块，用于处理文件路径和检查文件是否存在
import os
# 导入Python标准库中的sys模块，用于获取命令行参数
import sys
# 导入numpy库并简写为np，用于图片数据的数值处理和数组操作
import numpy as np
# 导入PyTorch核心库并简写为torch，用于加载模型和进行张量推理
import torch
# 从torch中导入nn模块，用于定义与训练时相同的网络结构
import torch.nn as nn
# 从Pillow库中导入Image模块，用于读取和预处理本地图片文件
from PIL import Image

# ============================================================
# 第一部分：定义CNN网络架构（必须与训练时完全一致）
# ============================================================

class CNN(nn.Module):
    """
    卷积神经网络类，继承自nn.Module
    注意：此类定义必须与train_and_test.py中的CNN类完全一致，
          否则加载state_dict时会因参数名或形状不匹配而报错
    网络结构：
        输入 -> Conv2d(1,32) -> ReLU -> MaxPool2d
             -> Conv2d(32,64) -> ReLU -> MaxPool2d
             -> Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout
             -> Linear(128, 10) -> 输出
    """

    def __init__(self):
        """
        初始化网络层，定义所有可学习参数的层
        各层参数必须与训练时完全相同
        """
        # 调用父类nn.Module的__init__方法，初始化模块基础设施
        super(CNN, self).__init__()

        # 第一个卷积层：输入1通道，输出32通道，3x3卷积核，步长1，填充1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 第一个最大池化层：2x2窗口，步长2，将28x28降为14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层：输入32通道，输出64通道，3x3卷积核，步长1，填充1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 第二个最大池化层：2x2窗口，步长2，将14x14降为7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第一个全连接层：输入64*7*7=3136个特征，输出128个特征
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # ReLU激活函数
        self.relu = nn.ReLU()

        # Dropout层，训练时50%概率丢弃，评估时自动关闭
        self.dropout = nn.Dropout(p=0.5)

        # 第二个全连接层（输出层）：输入128个特征，输出10个类别
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        前向传播，与训练时完全一致
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, 1, 28, 28)
        返回值：
            x (torch.Tensor): 输出logits，形状为 (batch_size, 10)
        """
        # 第一个卷积层 + ReLU + 池化
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # 第二个卷积层 + ReLU + 池化
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层 + ReLU + Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc2(x)
        return x


# ============================================================
# 第二部分：图片预处理函数
# ============================================================

def preprocess_image(image_path):
    """
    读取本地图片文件并进行预处理，使其符合模型输入要求
    参数：
        image_path (str): 本地图片文件的完整路径，
                         支持常见格式如 .png, .jpg, .bmp 等
    返回值：
        tensor (torch.Tensor): 预处理后的图片张量，形状为 (1, 1, 28, 28)，
                               即 batch_size=1, channels=1, height=28, width=28
    异常：
        FileNotFoundError: 当指定路径的图片文件不存在时抛出
    """
    # 检查图片文件是否存在，不存在则抛出异常并给出提示
    if not os.path.exists(image_path):
        # 抛出文件未找到异常，附带具体的文件路径信息
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 使用Pillow的Image.open()打开图片文件，返回PIL Image对象
    # 此时图片可能是RGB(3通道)、RGBA(4通道)或灰度(1通道)等模式
    img = Image.open(image_path)

    # 将图片转换为灰度模式('L')，确保只有1个通道
    # 如果原图是彩色的，会按 luminosity 公式转换：L = 0.299*R + 0.587*G + 0.114*B
    img = img.convert('L')

    # 将图片缩放为28x28像素，这是MNIST模型要求的输入尺寸
    # Image.LANCZOS是高质量的下采样滤波器，能减少缩放时的锯齿
    img = img.resize((28, 28), Image.LANCZOS)

    # 将PIL Image对象转换为numpy数组，形状为(28, 28)，dtype为uint8
    # 每个像素值范围为0-255
    img_array = np.array(img)

    # 将像素值从[0, 255]归一化到[0.0, 1.0]区间
    # 除以255.0并转换为float32类型，与训练时的预处理保持一致
    img_array = img_array.astype(np.float32) / 255.0

    # 将numpy数组转换为PyTorch张量
    # 此时形状为 (28, 28)
    tensor = torch.from_numpy(img_array)

    # 使用unsqueeze(0)在最前面增加一个批次维度
    # 形状从 (28, 28) 变为 (1, 28, 28)
    tensor = tensor.unsqueeze(0)

    # 再次使用unsqueeze(0)增加通道维度
    # 形状从 (1, 28, 28) 变为 (1, 1, 28, 28)
    # 最终形状: (batch_size=1, channels=1, height=28, width=28)
    tensor = tensor.unsqueeze(0)

    # 返回预处理完成的张量
    return tensor


# ============================================================
# 第三部分：预测函数
# ============================================================

def predict_digit(model, image_tensor):
    """
    使用训练好的模型对单张图片进行数字预测
    参数：
        model (nn.Module): 已加载权重的CNN模型实例
        image_tensor (torch.Tensor): 预处理后的图片张量，形状为 (1, 1, 28, 28)
    返回值：
        predicted_digit (int): 预测的数字结果，范围0-9
        confidence (float): 预测的置信度（概率值），范围0.0-1.0
        probabilities (numpy.ndarray): 所有10个类别的概率分布，形状为(10,)
    """
    # 将模型设置为评估模式
    # 这会关闭Dropout（不再随机丢弃神经元），确保推理结果确定性
    model.eval()

    # 使用torch.no_grad()上下文管理器禁用梯度计算
    # 推理时不需要反向传播，禁用梯度可以节省内存并加速计算
    with torch.no_grad():
        # 将图片张量输入模型，得到10个类别的logits（未归一化的分数）
        # 输出形状: (1, 10)
        outputs = model(image_tensor)

        # 使用softmax函数将logits转换为概率分布
        # dim=1表示沿类别维度（第1维）进行softmax归一化
        # 输出形状: (1, 10)，每行元素之和为1.0
        probabilities = torch.softmax(outputs, dim=1)

        # 获取概率最大的类别索引和对应的概率值
        # torch.max返回 (最大值, 最大值的索引)
        # dim=1表示沿类别维度取最大值
        confidence, predicted = torch.max(probabilities, 1)

    # 将预测的类别索引从张量转为Python整数（0-9）
    predicted_digit = predicted.item()
    # 将置信度从张量转为Python浮点数
    confidence = confidence.item()
    # 将概率分布从张量转为numpy数组，便于查看各类别概率
    probabilities = probabilities.squeeze(0).numpy()

    # 返回预测数字、置信度和完整概率分布
    return predicted_digit, confidence, probabilities


# ============================================================
# 第四部分：主程序入口
# ============================================================

if __name__ == "__main__":
    # 定义模型文件保存的路径（与训练时保存路径一致）
    model_path = r"/深度学习/图像/mnist/pytorch/mnist_cnn_model.pth"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        # 如果模型文件不存在，打印错误信息并退出程序
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行 train_and_test.py 训练并保存模型。")
        # sys.exit(1)以非零状态码退出，表示程序异常终止
        sys.exit(1)

    # 实例化CNN模型对象（此时参数是随机初始化的）
    model = CNN()

    # 使用torch.load加载保存的模型参数字典(state_dict)
    # weights_only=True表示只加载张量数据，不执行任何pickle代码，更安全
    # 这是PyTorch 2.6+的推荐做法，防止恶意代码执行
    state_dict = torch.load(model_path, weights_only=True)

    # 将加载的参数字典载入模型中，覆盖随机初始化的参数
    # strict=True（默认值）要求键名完全匹配，不匹配会报错
    model.load_state_dict(state_dict)

    # 打印模型加载成功的提示信息
    print(f"模型加载成功: {model_path}")
    # 打印模型结构概览
    print("\n模型结构：")
    print(model)

    # 定义待预测图片的路径（用户需要修改为实际的图片路径）
    # 这里提供一个示例路径，用户应替换为自己要预测的图片
    image_path = r"D:\Repositories\basic_algorithm\database\mnist\test_digit.png"

    # 打印分隔线
    print("\n" + "=" * 50)
    # 打印待预测图片的路径
    print(f"待预测图片: {image_path}")
    # 打印分隔线
    print("=" * 50)

    # 检查待预测的图片文件是否存在
    if not os.path.exists(image_path):
        # 如果图片不存在，打印提示信息
        print(f"\n提示：图片文件 '{image_path}' 不存在。")
        print("请将一张手写数字图片(0-9)放置到该路径，或修改代码中的image_path变量。")
        print("\n图片要求：")
        print("  - 格式：png / jpg / bmp 等常见图片格式")
        print("  - 内容：单个手写数字(0-9)")
        print("  - 建议：白色背景、黑色数字，或黑色背景、白色数字")
        # 程序正常退出
        sys.exit(0)

    # 调用预处理函数，将图片转换为模型可接受的张量格式
    # 返回形状为 (1, 1, 28, 28) 的float32张量
    image_tensor = preprocess_image(image_path)
    # 打印预处理后张量的形状和数据类型，用于验证
    print(f"预处理后张量形状: {image_tensor.shape}, 数据类型: {image_tensor.dtype}")

    # 调用预测函数，获取预测结果
    # predicted_digit: 预测的数字(0-9)
    # confidence: 预测置信度(0.0-1.0)
    # probabilities: 10个类别各自的概率
    predicted_digit, confidence, probabilities = predict_digit(model, image_tensor)

    # 打印预测结果
    print(f"\n预测结果: {predicted_digit}")
    # 打印置信度，保留4位小数
    print(f"置信度: {confidence:.4f} ({confidence * 100:.2f}%)")

    # 打印所有类别的概率分布
    print("\n各类别概率分布:")
    print("-" * 30)
    # 遍历0-9共10个类别
    for digit in range(10):
        # 获取当前类别的概率值
        prob = probabilities[digit]
        # 生成概率条：用'█'字符可视化概率大小，每2%一个方块
        bar = "█" * int(prob * 50)
        # 打印类别编号、概率百分比和可视化条形图
        print(f"  数字 {digit}: {prob * 100:6.2f}% |{bar}")
    # 打印结束分隔线
    print("-" * 30)