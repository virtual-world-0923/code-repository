# -*- coding: utf-8 -*-
"""
====================================================================
文件名称：predict_cnn.py
功能说明：加载已训练的CNN模型，读取本地图片文件并预测手写数字
运行环境：Python 3.13.14 / PyTorch 2.13.0+cpu / NumPy 2.5.1 / Pillow
依赖安装：pip install Pillow
使用方式：
    python predict_cnn.py                          # 使用默认图片路径
    python predict_cnn.py D:\path\to\digit.png     # 指定图片路径
====================================================================
"""

# ==================== 第一部分：导入必要的库 ====================

import sys                                  # 导入sys模块，用于获取命令行参数(sys.argv)和程序退出(sys.exit)
import os                                   # 导入os模块，用于文件路径操作和文件存在性检查
import numpy as np                         # 导入NumPy库，用于数值计算和数组变换
import torch                               # 导入PyTorch主库，提供张量计算功能
import torch.nn as nn                      # 导入PyTorch神经网络模块，用于定义CNN网络结构
from PIL import Image                      # 从Pillow库导入Image模块，用于读取、转换和缩放图片文件

# ==================== 第二部分：定义文件路径 ====================

# 获取当前脚本文件所在目录的绝对路径
# os.path.abspath(__file__)：获取当前.py文件的绝对路径字符串
# os.path.dirname()：从完整文件路径中提取目录部分
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练好的模型参数文件路径（必须与train_cnn.py中保存的路径一致）
# os.path.join()参数：目录路径, 文件名，自动用系统路径分隔符拼接
MODEL_PATH = os.path.join(SCRIPT_DIR, "mnist_cnn_model.pth")

# 默认预测图片路径（可通过命令行参数覆盖）
# 用户可以将其修改为自己的图片路径
DEFAULT_IMAGE_PATH = r"D:\Repositories\basic_algorithm\test_digit.png"


# ==================== 第三部分：定义CNN模型结构 ====================
# 注意：必须与train_cnn.py中的模型定义完全一致，否则无法加载state_dict

class MNISTCNN(nn.Module):
    """
    CNN卷积神经网络模型（结构必须与训练时完全一致）

    重要说明：PyTorch的state_dict只保存参数张量的名称和数值，
    不保存模型结构。因此加载时必须先创建相同结构的模型，
    再用load_state_dict()将参数填入对应位置。
    """

    def __init__(self):
        """
        初始化网络层结构（与train_cnn.py中完全相同）
        参数：无
        """
        # 调用父类nn.Module的初始化方法，注册模块基础设施
        super(MNISTCNN, self).__init__()

        # 第一个卷积层：1通道输入→32通道输出，3×3卷积核，padding=1保持尺寸
        # 参数：in_channels=1, out_channels=32, kernel_size=3, padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # 第一个ReLU激活函数
        # 参数：inplace=False（不修改原张量，返回新张量）
        self.relu1 = nn.ReLU(inplace=False)

        # 第一个最大池化层：2×2窗口，步长2，空间尺寸减半（28→14）
        # 参数：kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层：32通道输入→64通道输出，3×3卷积核，padding=1
        # 参数：in_channels=32, out_channels=64, kernel_size=3, padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 第二个ReLU激活函数
        self.relu2 = nn.ReLU(inplace=False)

        # 第二个最大池化层：空间尺寸再减半（14→7）
        # 参数：kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 展平层：从第1维开始展平，保留batch维度
        # 参数：start_dim=1
        # 效果：(B, 64, 7, 7) → (B, 3136)
        self.flatten = nn.Flatten(start_dim=1)

        # 第一个全连接层：3136维→128维
        # 参数：in_features=3136(=64×7×7), out_features=128
        self.fc1 = nn.Linear(3136, 128)

        # 第三个ReLU激活函数
        self.relu3 = nn.ReLU(inplace=False)

        # 输出全连接层：128维→10维（对应数字0~9的10个类别）
        # 参数：in_features=128, out_features=10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播过程（与train_cnn.py中完全相同）

        参数：
            x (torch.Tensor): 输入张量，形状(B, 1, 28, 28)

        返回：
            torch.Tensor: 输出logits，形状(B, 10)
        """
        # 卷积块1：(B,1,28,28)→(B,32,14,14)
        x = self.pool1(self.relu1(self.conv1(x)))
        # 卷积块2：(B,32,14,14)→(B,64,7,7)
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平：(B,64,7,7)→(B,3136)
        x = self.flatten(x)
        # 全连接+激活：(B,3136)→(B,128)
        x = self.relu3(self.fc1(x))
        # 输出层：(B,128)→(B,10)
        x = self.fc2(x)
        return x


# ==================== 第四部分：图片预处理函数 ====================

def preprocess_image(image_path):
    """
    读取本地图片文件并预处理为CNN模型所需的输入格式

    完整预处理流程：
        1. 用PIL打开图片文件
        2. 转换为灰度图（单通道'L'模式）
        3. 缩放到28×28像素（MNIST标准尺寸）
        4. 转为float32 numpy数组并归一化到[0, 1]
        5. 自动检测背景颜色，必要时进行反转（适配MNIST白字黑底格式）
        6. 添加batch和channel维度，转为PyTorch张量

    参数：
        image_path (str): 图片文件的完整路径，支持PNG/JPG/BMP/TIFF等格式

    返回：
        torch.Tensor: 预处理后的输入张量，形状(1, 1, 28, 28)，dtype=float32
    """
    # 使用PIL的Image.open()读取本地图片文件
    # Image.open()参数：fp-文件路径字符串或文件对象
    # 自动根据文件头识别格式（PNG、JPEG、BMP等）
    img = Image.open(image_path)

    # 将图片转换为灰度模式
    # .convert()参数：mode-目标颜色模式
    # 'L'模式：8位灰度像素，0=纯黑，255=纯白，自动合并RGB三通道
    img = img.convert('L')

    # 将图片缩放到28×28像素（与MNIST训练数据尺寸一致）
    # .resize()参数说明：
    #   size: (width, height)元组，PIL使用(宽,高)顺序
    #   resampling: 重采样滤波器，Image.LANCZOS为高质量抗锯齿下采样算法
    img = img.resize((28, 28), Image.LANCZOS)

    # 将PIL Image对象转换为NumPy数组
    # np.array()参数：object-可转换为数组的对象（PIL Image支持）；dtype-目标数据类型
    # 转换后形状为(28, 28)，值范围[0, 255]，类型float32
    img_array = np.array(img, dtype=np.float32)

    # 像素值归一化：从[0, 255]线性缩放到[0.0, 1.0]
    # 与训练时的数据预处理方式保持一致
    img_array = img_array / 255.0

    # 自动检测是否需要颜色反转
    # MNIST数据格式：白色数字（高像素值）在黑色背景上（低像素值，均值偏低）
    # 常见手写照片/截图：黑色文字在白色背景上（均值偏高）
    # 判断方法：计算整幅图像的平均像素值，若>0.5说明是白底图片，需反转
    if img_array.mean() > 0.5:
        # 颜色反转：用1.0减去每个像素值
        # 效果：白底黑字(0.0=字, 1.0=底) → 黑底白字(1.0=字, 0.0=底)
        # 这样与MNIST训练数据的格式保持一致
        img_array = 1.0 - img_array

    # 在第0维（batch维度）插入大小为1的新轴
    # np.expand_dims()参数：arr-输入数组；axis=0-在第0维插入新轴
    # 效果：(28, 28) → (1, 28, 28)，表示batch中有1张图片
    img_array = np.expand_dims(img_array, axis=0)

    # 在第1维（channel维度）插入大小为1的新轴
    # 效果：(1, 28, 28) → (1, 1, 28, 28)，表示1个通道（灰度图）
    img_array = np.expand_dims(img_array, axis=0)

    # 将NumPy数组转换为PyTorch张量
    # torch.from_numpy()参数：ndarray-NumPy数组
    # 此时张量形状为(1, 1, 28, 28)，dtype为float32，满足CNN输入要求
    img_tensor = torch.from_numpy(img_array)

    # 返回预处理后的张量
    return img_tensor


# ==================== 第五部分：数字预测函数 ====================

def predict_digit(model, image_tensor, device):
    """
    使用已训练的CNN模型对单张预处理图片进行手写数字预测

    流程：
        1. 设置模型为评估模式
        2. 禁用梯度计算
        3. 前向传播获取logits
        4. 通过softmax转为概率分布
        5. 取概率最大值的类别作为预测结果

    参数：
        model (MNISTCNN):            已加载权重的CNN模型实例
        image_tensor (torch.Tensor): 预处理后的图片张量，形状(1, 1, 28, 28)
        device (torch.device):       计算设备（'cpu'或'cuda'）

    返回：
        tuple: (predicted_digit, probabilities)
            - predicted_digit (int): 预测的数字（0~9）
            - probabilities (numpy.ndarray): 10个类别的softmax概率，形状(10,)
    """
    # 将图片张量移动到目标设备
    image_tensor = image_tensor.to(device)

    # 将模型设置为评估模式
    # eval()：关闭Dropout层的随机丢弃，BatchNorm使用训练时的全局均值/方差
    model.eval()

    # 使用torch.no_grad()上下文管理器禁用梯度计算
    # 推理时不需要反向传播，禁用梯度可减少约50%的内存消耗并加速计算
    with torch.no_grad():

        # 前向传播：将图片输入模型获取原始输出分数
        # output形状：(1, 10)，包含10个类别的raw logits（未归一化的分数）
        output = model(image_tensor)

        # 将logits通过softmax函数转换为概率分布
        # torch.softmax()参数：input-输入张量；dim=1-沿类别维度计算softmax
        # softmax公式：P(class_i) = exp(logit_i) / Σexp(logit_j)
        # 特性：所有概率非负且总和为1
        # probabilities形状：(1, 10)
        probabilities = torch.softmax(output, dim=1)

        # 获取概率最大值对应的类别索引作为预测结果
        # torch.max()参数：input-输入张量(1,10)；dim=1-沿类别维度取最大值
        # 返回值：(最大值values, 最大值索引indices)元组
        # [1]：取索引元素（即预测的类别编号0~9）
        # .item()：将单元素张量转为Python标量整数
        predicted_digit = torch.max(probabilities, dim=1)[1].item()

    # 将概率张量从PyTorch格式转为NumPy数组（便于后续显示和处理）
    # [0]：取出batch维度，从(1,10)变为(10,)
    # .cpu()：确保张量在CPU上（从可能的GPU设备移回CPU）
    # .numpy()：将PyTorch张量转为NumPy ndarray
    probabilities = probabilities[0].cpu().numpy()

    # 返回预测的数字和10个类别的概率分布
    return predicted_digit, probabilities


# ==================== 第六部分：ASCII可视化函数 ====================

def display_ascii_image(image_tensor):
    """
    在终端中以ASCII字符可视化预处理后的28×28图像

    将像素亮度映射为不同密度的Unicode字符，在终端中呈现手写数字

    参数：
        image_tensor (torch.Tensor): 预处理后的图片张量，形状(1, 1, 28, 28)

    返回：无（直接打印到控制台）
    """
    # 从张量中提取28×28图像数据
    # [0, 0]：索引第0个batch和第0个通道，从(1,1,28,28)变为(28,28)
    # .numpy()：将PyTorch张量转为NumPy数组
    img_data = image_tensor[0, 0].numpy()

    # 打印ASCII图像标题
    print("\n预处理后图像可视化（终端ASCII）：")
    print("-" * 34)

    # 遍历图像的每一行（共28行）
    for row in img_data:

        # 初始化当前行的ASCII字符串
        line = ''

        # 遍历当前行中的每个像素（共28列）
        for pixel in row:

            # 根据像素亮度选择不同密度的Unicode字符
            # MNIST归一化后：0.0=黑色背景，1.0=白色笔画
            if pixel > 0.6:
                line += '██'       # 高亮度：用全填充Unicode块字符表示（笔画核心）
            elif pixel > 0.3:
                line += '▓▓'       # 中亮度：用3/4填充块表示（笔画边缘）
            elif pixel > 0.1:
                line += '░░'       # 低亮度：用1/4填充块表示（模糊区域）
            else:
                line += '  '       # 极低亮度/黑色背景：用空格表示

        # 打印当前行的ASCII表示
        print(line)

    # 打印结束分隔线
    print("-" * 34)


# ==================== 第七部分：主函数 ====================

def main():
    """
    主函数：加载模型 → 读取图片 → 预处理 → 预测 → 显示结果

    参数：无
    返回：无（结果打印到控制台）
    """
    # ---------- 1. 解析命令行参数获取图片路径 ----------

    # 检查是否通过命令行参数指定了图片路径
    # sys.argv：命令行参数列表，sys.argv[0]是脚本自身路径，后续元素是用户参数
    # len(sys.argv) > 1：表示用户至少传入了一个参数
    if len(sys.argv) > 1:
        # 使用命令行指定的第一个参数作为图片路径
        image_path = sys.argv[1]
    else:
        # 未指定参数时使用默认图片路径
        image_path = DEFAULT_IMAGE_PATH

    # 打印程序标题
    print("=" * 60)
    print("CNN手写数字识别 - 图片预测程序")
    print("=" * 60)

    # ---------- 2. 检查必要文件是否存在 ----------

    # 检查模型文件是否存在于指定路径
    # os.path.exists()参数：path-文件或目录路径字符串
    # 返回值：True表示存在，False表示不存在
    if not os.path.exists(MODEL_PATH):
        # 模型文件不存在，打印错误提示并退出
        print(f"\n错误：找不到模型文件 '{MODEL_PATH}'")
        print("请先运行 train_cnn.py 训练并保存模型！")
        # sys.exit()参数：exit_code-退出码，1表示异常退出，0表示正常退出
        sys.exit(1)

    # 检查图片文件是否存在于指定路径
    if not os.path.exists(image_path):
        # 图片文件不存在，打印错误提示并退出
        print(f"\n错误：找不到图片文件 '{image_path}'")
        print("请检查图片路径是否正确。")
        sys.exit(1)

    # ---------- 3. 设置计算设备 ----------

    # 创建CPU计算设备对象
    device = torch.device('cpu')
    print(f"\n计算设备：{device}")

    # ---------- 4. 加载已训练的模型 ----------

    # 实例化CNN模型（结构必须与训练时的定义完全一致，包括层顺序和参数）
    model = MNISTCNN()

    # 使用torch.load()加载保存的模型参数字典
    # torch.load()参数说明：
    #   f:             模型文件路径字符串或文件对象
    #   map_location:  指定参数张量映射到的设备，'cpu'确保无论原始设备是什么都加载到CPU
    #   weights_only:  True表示仅加载权重张量（安全模式），防止加载恶意pickle代码
    # 返回值：OrderedDict类型，键为参数名（如'conv1.weight'），值为参数张量
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)

    # 将加载的参数字典填入模型的对应层中
    # load_state_dict()参数：state_dict-包含参数名和张量的OrderedDict
    # 该方法会将字典中每个键对应的张量加载到模型中同名的参数里
    model.load_state_dict(state_dict)

    # 打印加载确认信息
    print(f"模型参数已从 '{MODEL_PATH}' 加载成功")

    # ---------- 5. 读取并预处理图片 ----------

    print(f"正在读取图片：{image_path}")

    # 调用预处理函数，将本地图片文件转为CNN输入张量
    # preprocess_image()参数：image_path-图片文件路径
    # 返回：形状为(1, 1, 28, 28)的float32张量
    image_tensor = preprocess_image(image_path)

    # 打印预处理后的张量形状确认信息
    # list(image_tensor.shape)：将Shape转为列表如[1, 1, 28, 28]
    print(f"预处理完成，输入张量形状：{list(image_tensor.shape)}")

    # ---------- 6. 可选：在终端可视化预处理后的图像 ----------

    # 调用ASCII可视化函数，在终端中用字符画展示预处理后的手写数字
    display_ascii_image(image_tensor)

    # ---------- 7. 执行数字预测 ----------

    print("\n正在执行预测...")

    # 调用预测函数，获取预测数字和各类别概率
    # predict_digit()参数：model-加载好权重的模型，image_tensor-预处理后的图片，device-设备
    # 返回：predicted_digit-预测数字(int, 0~9)，probabilities-概率数组(ndarray, shape=(10,))
    predicted_digit, probabilities = predict_digit(model, image_tensor, device)

    # ---------- 8. 显示预测结果 ----------

    # 打印预测结果区域
    print("\n" + "=" * 60)
    print(f"  ★ 预测结果：{predicted_digit}  ★")
    print("=" * 60)

    # 打印每个数字类别的详细概率分布
    print("\n各类别预测概率：")
    for digit in range(10):
        # 遍历0~9每个数字，打印其预测概率
        # digit:>2：数字右对齐占2字符宽
        # probabilities[digit]：该数字的预测概率值
        # :.4f：保留4位小数的概率值
        # probabilities[digit]*100：转为百分比
        # :.2f：百分比保留2位小数
        bar = '█' * int(probabilities[digit] * 30)  # 概率条形图（长度与概率成正比）
        print(f"  数字 {digit}: {probabilities[digit]:.4f}  ({probabilities[digit] * 100:>5.2f}%)  {bar}")

    # 打印最高置信度总结
    # predicted_digit：概率最高的数字类别
    # probabilities[predicted_digit]：该数字对应的概率值
    print(
        f"\n置信度最高的数字：{predicted_digit}，"
        f"概率：{probabilities[predicted_digit] * 100:.2f}%"
    )

    # 打印结束信息
    print("\n预测完成！")


# ==================== 程序入口 ====================

# 当该文件作为主程序直接运行时执行main函数
# 若被其他文件import导入则不会执行（此时__name__为模块名而非'__main__'）
if __name__ == '__main__':
    main()  # 调用主函数，启动完整的加载→预处理→预测→显示流程