# -*- coding: utf-8 -*-
"""
====================================================================
文件名称：train_cnn.py
功能说明：使用PyTorch搭建CNN卷积神经网络，在本地MNIST数据集上进行训练和测试
运行环境：Python 3.13.14 / PyTorch 2.13.0+cpu / NumPy 2.5.1
====================================================================
"""

# ==================== 第一部分：导入必要的库 ====================

import os                                    # 导入os模块，用于文件路径拼接与目录操作
import struct                                # 导入struct模块，用于将MNIST二进制idx文件解析为数值数据
import numpy as np                           # 导入NumPy库（别名np），用于高效数值计算与数组操作
import torch                                 # 导入PyTorch主库，提供张量计算、自动求导等核心功能
import torch.nn as nn                        # 导入PyTorch神经网络模块，提供Conv2d、Linear等层定义
import torch.optim as optim                  # 导入PyTorch优化器模块，提供Adam、SGD等优化算法
from torch.utils.data import (              # 从PyTorch数据工具中导入以下组件：
    DataLoader,                              #   DataLoader：数据加载器，支持自动批处理、打乱和并行加载
    TensorDataset                            #   TensorDataset：将张量包装为数据集对象，便于与DataLoader配合使用
)

# ==================== 第二部分：定义超参数与文件路径 ====================

# MNIST原始数据文件所在目录路径
# 作用：指定包含4个idx文件的文件夹位置
# 说明：使用原始字符串r""避免反斜杠转义问题
DATA_DIR = r"D:\Repositories\basic_algorithm\database\mnist"

# 训练好的模型保存路径
# os.path.abspath(__file__)：获取当前脚本文件的绝对路径
# os.path.dirname()：提取路径中的目录部分
# os.path.join()：拼接目录和文件名为完整路径
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # 当前脚本所在目录
    "mnist_cnn_model.pth"                        # 模型文件名（.pth为PyTorch标准格式）
)

# 训练超参数定义
BATCH_SIZE = 64       # 每个训练批次的样本数量，较大值训练更稳定但占用更多内存
EPOCHS = 3            # 整个训练集遍历的次数，更多轮次可提升精度但可能过拟合
LEARNING_RATE = 0.001 # 学习率，控制参数更新步长，Adam优化器默认推荐0.001


# ==================== 第三部分：读取MNIST原始idx文件 ====================

def read_mnist_images(file_path):
    """
    读取MNIST图像idx3-ubyte文件，返回图像像素数组

    idx3-ubyte文件格式：
        - 前16字节为头部：魔数(4B) + 图片数(4B) + 行数(4B) + 列数(4B)
        - 之后为连续的像素数据，每张图28×28=784字节

    参数：
        file_path (str): idx3-ubyte图像文件的完整路径

    返回：
        numpy.ndarray: 形状为(样本数, 28, 28)的uint8类型数组，像素值范围[0, 255]
    """
    # 使用with语句以二进制只读模式打开文件，确保文件用完后自动关闭
    # open()参数：file-文件路径字符串；mode-打开模式，'rb'表示二进制只读
    with open(file_path, 'rb') as f:

        # 从文件开头读取16字节头部信息，按大端字节序解析为4个无符号32位整数
        # struct.unpack()参数：format-格式字符串，'>IIII'表示大端序4个uint32；buffer-字节数据
        # '>' 大端字节序（网络字节序，MNIST标准格式）
        # 'I' 无符号32位整数（4字节）
        # f.read(16)：从文件当前位置读取16个字节
        # 返回值解包为：magic-魔数标识，num_images-图像总数，rows-行数，cols-列数
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))

        # 校验魔数是否为2051（MNIST图像文件的国际标准标识符）
        # assert：若条件为False则抛出AssertionError，用于数据完整性校验
        assert magic == 2051, f"图像文件魔数错误：期望2051，得到{magic}"

        # 读取文件剩余所有像素字节，通过numpy的frombuffer直接映射为uint8数组
        # frombuffer()参数：buffer-字节缓冲区对象；dtype-目标数据类型np.uint8（0~255）
        # 零拷贝方式，直接引用文件字节内存，效率极高
        data = np.frombuffer(f.read(), dtype=np.uint8)

        # 将一维像素数组重塑为三维结构(样本数, 高度28, 宽度28)
        # reshape()参数：newshape-目标形状元组(num_images, rows, cols)
        data = data.reshape(num_images, rows, cols)

    # 返回形状为(N, 28, 28)的图像数组
    return data


def read_mnist_labels(file_path):
    """
    读取MNIST标签idx1-ubyte文件，返回标签数组

    idx1-ubyte文件格式：
        - 前8字节为头部：魔数(4B) + 标签数(4B)
        - 之后为连续的标签字节，每个字节值为0~9

    参数：
        file_path (str): idx1-ubyte标签文件的完整路径

    返回：
        numpy.ndarray: 形状为(样本数,)的uint8类型数组，元素值范围[0, 9]表示数字类别
    """
    # 以二进制只读模式打开标签文件
    with open(file_path, 'rb') as f:

        # 读取8字节头部：魔数(4字节) + 标签总数(4字节)
        # struct.unpack()参数：'>II'表示大端序2个无符号32位整数
        magic, num_labels = struct.unpack('>II', f.read(8))

        # 校验魔数是否为2049（MNIST标签文件的标准标识符）
        assert magic == 2049, f"标签文件魔数错误：期望2049，得到{magic}"

        # 读取剩余字节并映射为uint8数组，每个字节即一个数字标签（0~9）
        data = np.frombuffer(f.read(), dtype=np.uint8)

    # 返回一维标签数组(N,)
    return data


# ==================== 第四部分：定义CNN卷积神经网络 ====================

class MNISTCNN(nn.Module):
    """
    手写数字识别CNN卷积神经网络模型

    网络架构（B = batch_size）：
        输入:      (B, 1, 28, 28)
        → Conv1:   (B, 1, 28, 28)  → (B, 32, 28, 28)   3×3卷积，padding=1
        → ReLU1:   (B, 32, 28, 28) → (B, 32, 28, 28)   非线性激活
        → Pool1:   (B, 32, 28, 28) → (B, 32, 14, 14)   2×2最大池化
        → Conv2:   (B, 32, 14, 14) → (B, 64, 14, 14)   3×3卷积，padding=1
        → ReLU2:   (B, 64, 14, 14) → (B, 64, 14, 14)   非线性激活
        → Pool2:   (B, 64, 14, 14) → (B, 64, 7, 7)     2×2最大池化
        → Flatten: (B, 64, 7, 7)   → (B, 3136)          展平为一维
        → FC1:     (B, 3136)       → (B, 128)            全连接隐藏层
        → ReLU3:   (B, 128)        → (B, 128)            非线性激活
        → FC2:     (B, 128)        → (B, 10)             输出层（10类）
    """

    def __init__(self):
        """
        初始化CNN网络的所有层结构
        参数：无（继承自nn.Module）
        """
        # 调用父类nn.Module的__init__方法，初始化PyTorch模块的内部基础设施
        # super()返回父类代理对象，调用其__init__完成注册参数等必需操作
        super(MNISTCNN, self).__init__()

        # ========== 第一个卷积块：提取低级特征（边缘、线条） ==========

        # 第一个二维卷积层
        # nn.Conv2d()参数说明：
        #   in_channels=1:  输入通道数，灰度图为1（RGB图像则为3）
        #   out_channels=32: 输出通道数（卷积核数量），提取32种不同局部特征
        #   kernel_size=3:   卷积核大小3×3，小感受野适合提取局部细节
        #   padding=1:       在输入边缘各填充1像素零值，使输出空间尺寸与输入相同
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, padding=1
        )

        # 第一个ReLU激活函数层
        # nn.ReLU()参数说明：
        #   inplace=False: 不在原地修改输入张量，而是返回新张量（更安全，占用略多内存）
        # 作用：f(x) = max(0, x)，将所有负值置零，引入非线性能力
        self.relu1 = nn.ReLU(inplace=False)

        # 第一个最大池化层
        # nn.MaxPool2d()参数说明：
        #   kernel_size=2: 池化窗口大小2×2
        #   stride=2:      步长为2（窗口无重叠），输出尺寸降为输入的一半
        # 作用：降低特征图分辨率，减少计算量，增强特征的平移不变性
        # 效果：空间尺寸 28×28 → 14×14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ========== 第二个卷积块：提取高级特征（形状、结构组合） ==========

        # 第二个二维卷积层
        # nn.Conv2d()参数说明：
        #   in_channels=32:  输入通道数，接收上一层输出的32个特征图
        #   out_channels=64: 输出通道数增加到64，捕捉更丰富的特征组合
        #   kernel_size=3:   3×3卷积核
        #   padding=1:       保持空间尺寸不变
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        # 第二个ReLU激活函数
        self.relu2 = nn.ReLU(inplace=False)

        # 第二个最大池化层（参数与pool1相同）
        # 效果：空间尺寸 14×14 → 7×7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ========== 展平层：多维特征图 → 一维向量 ==========

        # 展平层：将卷积输出的多维特征图展平为一维向量，供全连接层使用
        # nn.Flatten()参数说明：
        #   start_dim=1: 从第1维开始展平，保留第0维（batch维度）不变
        # 效果：(B, 64, 7, 7) → (B, 64×7×7) = (B, 3136)
        self.flatten = nn.Flatten(start_dim=1)

        # ========== 全连接层：特征映射与分类决策 ==========

        # 第一个全连接层（隐藏层）
        # nn.Linear()参数说明：
        #   in_features=3136:  输入特征维度，64×7×7=3136（卷积输出展平后的向量长度）
        #   out_features=128:  输出特征维度（隐藏层神经元数量），控制模型表达能力
        self.fc1 = nn.Linear(in_features=3136, out_features=128)

        # 第三个ReLU激活函数
        self.relu3 = nn.ReLU(inplace=False)

        # 第二个全连接层（输出层）
        # nn.Linear()参数说明：
        #   in_features=128:  输入特征维度，上一层输出128维
        #   out_features=10:  输出维度10，对应10个数字类别（0~9）
        # 输出值为原始logits分数（未经softmax归一化），CrossEntropyLoss内部会处理softmax
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        定义前向传播流程：数据在网络中的完整计算路径

        参数：
            x (torch.Tensor): 输入张量，形状为(batch_size, 1, 28, 28)
                - batch_size: 批次中的样本数量
                - 1: 通道数（灰度图为单通道）
                - 28, 28: 图像的高度和宽度（像素）

        返回：
            torch.Tensor: 输出张量，形状为(batch_size, 10)
                         每个样本输出10个类别的原始分数（logits）
        """
        # 第一个卷积块：卷积 → ReLU激活 → 最大池化
        # 数据流：(B,1,28,28) → conv1 → (B,32,28,28) → relu1 → (B,32,28,28) → pool1 → (B,32,14,14)
        x = self.pool1(self.relu1(self.conv1(x)))

        # 第二个卷积块：卷积 → ReLU激活 → 最大池化
        # 数据流：(B,32,14,14) → conv2 → (B,64,14,14) → relu2 → (B,64,14,14) → pool2 → (B,64,7,7)
        x = self.pool2(self.relu2(self.conv2(x)))

        # 展平操作：将三维特征图展平为一维向量
        # 数据流：(B,64,7,7) → flatten → (B,3136)
        x = self.flatten(x)

        # 第一个全连接层 + ReLU激活：将高维特征映射到低维隐藏空间
        # 数据流：(B,3136) → fc1 → (B,128) → relu3 → (B,128)
        x = self.relu3(self.fc1(x))

        # 输出全连接层：生成10个类别的原始分数（logits）
        # 数据流：(B,128) → fc2 → (B,10)
        x = self.fc2(x)

        # 返回模型最终输出
        return x


# ==================== 第五部分：打印网络各层维度信息 ====================

def print_model_layers(model):
    """
    逐层打印CNN模型的名称、输入维度和输出维度
    通过构建虚拟输入并逐层前向传播，精确获取每层的实际输入/输出尺寸

    参数：
        model (MNISTCNN): 已实例化的CNN模型对象

    返回：
        无（直接在控制台打印信息）
    """
    # 打印标题区域
    print("\n" + "=" * 70)
    print("CNN模型网络架构 —— 各层输入 / 输出维度详情")
    print("=" * 70)

    # 按网络前向传播顺序，手动列出每层的名称和对应模块对象
    # 使用列表保证顺序与forward()中的调用顺序一致
    # 每个元素为二元组：(层名称字符串, 层模块实例)
    layers_info = [
        ('conv1', model.conv1),       # 第一个卷积层
        ('relu1', model.relu1),       # 第一个ReLU激活
        ('pool1', model.pool1),       # 第一个最大池化层
        ('conv2', model.conv2),       # 第二个卷积层
        ('relu2', model.relu2),       # 第二个ReLU激活
        ('pool2', model.pool2),       # 第二个最大池化层
        ('flatten', model.flatten),   # 展平层
        ('fc1', model.fc1),           # 第一个全连接层
        ('relu3', model.relu3),       # 第三个ReLU激活
        ('fc2', model.fc2),           # 输出全连接层
    ]

    # 创建一个全零虚拟输入张量，形状模拟一个batch_size=1的灰度图像
    # torch.zeros()参数：size-张量形状元组(1,1,28,28)
    # 1=batch_size(单样本), 1=通道数(灰度), 28=高度, 28=宽度
    x = torch.zeros(1, 1, 28, 28)

    # 打印表头（使用f-string格式化对齐）
    # <12: 左对齐占12字符宽；<28: 左对齐占28字符宽
    print(f"{'层名称':<12} {'输入维度':<28} {'输出维度':<28}")
    print("-" * 68)

    # 遍历每一层，执行前向传播并记录输入输出维度
    for name, layer in layers_info:

        # 记录进入当前层之前的张量形状，即该层的输入维度
        # list(x.shape)：将张量的Shape对象转为Python列表，如[1, 1, 28, 28]
        input_shape = list(x.shape)

        # 执行当前层的前向计算，将输出结果覆盖x变量以传递给下一层
        # layer(x)：等价于调用layer.forward(x)，但通过__call__可触发hooks
        x = layer(x)

        # 记录当前层输出的张量形状
        output_shape = list(x.shape)

        # 格式化打印当前层信息：层名称、输入维度、输出维度
        # str(input_shape)：将列表转为字符串，如'[1, 1, 28, 28]'
        print(f"{name:<12} {str(input_shape):<28} {str(output_shape):<28}")

    # 打印结束分隔线
    print("=" * 70)

    # 统计模型总参数量
    # model.parameters()：返回模型所有可学习参数的迭代器（包括权重和偏置）
    # p.numel()：返回参数张量中元素的总个数（number of elements）
    # sum(生成器表达式)：累加所有参数的数量
    total_params = sum(p.numel() for p in model.parameters())

    # 统计可训练参数量（不包括冻结的参数，本模型全部可训练）
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # 打印参数统计信息
    # :, 格式说明符：使用千位分隔符，如421,642
    print(f"模型总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")
    print("=" * 70)


# ==================== 第六部分：训练与评估函数 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练模型一个完整的epoch（遍历一次全部训练数据）

    训练流程（每个batch）：
        1. 清零梯度
        2. 前向传播获取预测
        3. 计算损失
        4. 反向传播计算梯度
        5. 更新模型参数

    参数：
        model (MNISTCNN):          CNN模型实例
        train_loader (DataLoader):  训练数据加载器，自动按batch提供数据
        criterion (nn.Module):      损失函数，此处为交叉熵损失
        optimizer (Optimizer):      优化器，此处为Adam
        device (torch.device):      计算设备（'cpu'或'cuda'）

    返回：
        tuple: (平均训练损失值float, 训练准确率百分比float)
    """
    # 将模型设置为训练模式
    # train()：启用训练时特有的行为（如Dropout层的随机丢弃、BatchNorm使用批次统计量）
    # 本模型虽无这些层，但调用train()是PyTorch的标准最佳实践
    model.train()

    # 初始化统计变量
    running_loss = 0.0            # 累计损失值，用于计算epoch平均损失
    correct_predictions = 0       # 正确预测的样本数，用于计算准确率
    total_samples = 0             # 已处理的总样本数

    # 遍历训练数据加载器中的每个batch
    # train_loader每次迭代返回一个元组(batch_images, batch_labels)
    #   batch_images: (B, 1, 28, 28)的图像张量
    #   batch_labels: (B,)的标签张量，值为0~9的整数
    for batch_images, batch_labels in train_loader:

        # 将数据张量移动到指定计算设备
        # .to(device)：将张量迁移到目标设备（CPU或GPU），若已在目标设备则无操作
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # 清零优化器中所有参数的梯度缓存
        # 原因：PyTorch默认会累加梯度（方便RNN等场景），因此每个batch必须手动清零
        # optimizer.zero_grad()：将model中所有requires_grad=True的参数的.grad属性置零
        optimizer.zero_grad()

        # 前向传播：将批次图像输入模型，执行forward()获得预测输出
        # outputs形状：(B, 10)，每行包含该样本对10个数字类别的原始分数（logits）
        outputs = model(batch_images)

        # 计算损失值：衡量预测输出与真实标签之间的差距
        # criterion()参数：input-模型输出(B,10)的logits；target-真实标签(B,)的类别索引
        # nn.CrossEntropyLoss内部先对logits做softmax，再计算负对数似然的均值
        # 返回的loss是一个标量张量（requires_grad=True，用于反向传播）
        loss = criterion(outputs, batch_labels)

        # 反向传播：从loss开始，沿计算图自动计算每个参数的梯度
        # loss.backward()：利用链式法则计算 ∂loss/∂w 对所有叶子张量的梯度
        # 计算结果存储在每个参数的.grad属性中
        loss.backward()

        # 参数更新：根据梯度和学习率调整模型参数值
        # optimizer.step()：Adam优化器根据各参数的梯度自适应地更新参数
        # w = w - lr * m / (sqrt(v) + eps)，其中m和v是梯度的一阶和二阶矩估计
        optimizer.step()

        # 累加损失值（加权求和，权重为batch大小，后续除以总样本数得均值）
        # loss.item()：将单元素张量转为Python float标量
        # batch_images.size(0)：获取当前batch的实际样本数（最后一个batch可能小于BATCH_SIZE）
        running_loss += loss.item() * batch_images.size(0)

        # 获取预测类别：对每个样本取10个logits中最大值的索引
        # torch.max()参数：input-输入张量(B,10)；dim=1-沿类别维度（第1维）取最大值
        # 返回值：(values, indices)二元组，我们只需要indices即预测类别
        # predicted形状：(B,)，每个元素为0~9的预测数字
        _, predicted = torch.max(outputs, dim=1)

        # 统计正确预测数
        # (predicted == batch_labels)：逐元素比较，返回布尔张量(B,)
        # .sum()：统计True的数量（即预测正确的样本数）
        # .item()：将标量张量转为Python整数
        correct_predictions += (predicted == batch_labels).sum().item()

        # 累加总样本数
        total_samples += batch_labels.size(0)

    # 计算该epoch的平均训练损失（总损失 / 总样本数）
    avg_loss = running_loss / total_samples

    # 计算训练准确率（正确数 / 总数 × 100%）
    accuracy = 100.0 * correct_predictions / total_samples

    # 返回该epoch的平均损失和准确率
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    在给定数据集上评估模型性能（不更新模型参数）

    与train_one_epoch的主要区别：
        1. 调用model.eval()进入评估模式
        2. 使用torch.no_grad()禁用梯度计算
        3. 不执行loss.backward()和optimizer.step()

    参数：
        model (MNISTCNN):         CNN模型实例
        data_loader (DataLoader):  数据加载器（训练集或测试集均可）
        criterion (nn.Module):     损失函数
        device (torch.device):     计算设备

    返回：
        tuple: (平均损失值float, 准确率百分比float)
    """
    # 将模型设置为评估模式
    # eval()：关闭Dropout层的随机丢弃，BatchNorm使用训练时的全局统计量
    model.eval()

    # 初始化统计变量
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 使用torch.no_grad()上下文管理器禁用梯度追踪
    # 作用：内部所有Tensor操作不构建计算图，不存储中间结果，大幅减少内存消耗并加速计算
    with torch.no_grad():

        # 遍历数据加载器中的每个batch
        for batch_images, batch_labels in data_loader:

            # 将数据移到目标设备
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # 前向传播获取预测输出
            outputs = model(batch_images)

            # 计算当前batch的损失
            loss = criterion(outputs, batch_labels)

            # 累加损失值
            running_loss += loss.item() * batch_images.size(0)

            # 获取预测类别
            _, predicted = torch.max(outputs, dim=1)

            # 统计正确预测数
            correct_predictions += (predicted == batch_labels).sum().item()

            # 累加总样本数
            total_samples += batch_labels.size(0)

    # 计算平均损失
    avg_loss = running_loss / total_samples

    # 计算准确率百分比
    accuracy = 100.0 * correct_predictions / total_samples

    # 返回损失和准确率
    return avg_loss, accuracy


# ==================== 第七部分：主函数 ====================

def main():
    """
    主函数：协调数据加载、模型构建、训练、测试和保存的完整流程
    参数：无
    返回：无（结果打印到控制台，模型保存到文件）
    """
    # ---------- 1. 设备配置 ----------

    # 创建计算设备对象，指定为CPU
    # torch.device()参数：device-设备类型字符串，'cpu'表示使用CPU计算
    device = torch.device('cpu')
    print(f"使用计算设备：{device}")

    # ---------- 2. 设置随机种子以保证可复现性 ----------

    # 设置PyTorch的CPU随机种子，影响权重初始化、数据打乱等所有随机操作
    # torch.manual_seed()参数：seed-整数种子值
    torch.manual_seed(42)

    # 设置NumPy的随机种子，确保数据处理过程中的随机操作结果可复现
    # np.random.seed()参数：seed-整数种子值
    np.random.seed(42)

    # ---------- 3. 加载MNIST原始数据 ----------

    # 打印加载提示
    print("\n正在加载MNIST数据集...")

    # 构建各数据文件的完整路径
    # os.path.join()参数：*paths-多个路径组件，自动用系统分隔符拼接
    train_images_path = os.path.join(DATA_DIR, 'train-images-idx3-ubyte')  # 训练图像路径
    train_labels_path = os.path.join(DATA_DIR, 'train-labels-idx1-ubyte')  # 训练标签路径
    test_images_path = os.path.join(DATA_DIR, 't10k-images-idx3-ubyte')    # 测试图像路径
    test_labels_path = os.path.join(DATA_DIR, 't10k-labels-idx1-ubyte')    # 测试标签路径

    # 调用自定义函数读取MNIST原始idx二进制文件
    train_images = read_mnist_images(train_images_path)  # 训练图像(60000, 28, 28) uint8
    train_labels = read_mnist_labels(train_labels_path)  # 训练标签(60000,) uint8
    test_images = read_mnist_images(test_images_path)    # 测试图像(10000, 28, 28) uint8
    test_labels = read_mnist_labels(test_labels_path)    # 测试标签(10000,) uint8

    # 打印加载结果
    print(f"训练集：图像 shape={train_images.shape}，标签 shape={train_labels.shape}")
    print(f"测试集：图像 shape={test_images.shape}，标签 shape={test_labels.shape}")

    # ---------- 4. 数据预处理 ----------

    print("\n正在进行数据预处理...")

    # 将像素值从uint8 [0, 255] 转为float32 [0.0, 1.0]
    # .astype(np.float32)：转换数据类型为32位浮点数（神经网络要求浮点输入）
    # / 255.0：归一化，将像素值线性缩放到[0, 1]范围
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # 在通道维度（第1维）插入新轴，将(N,28,28)变为(N,1,28,28)
    # np.expand_dims()参数：arr-输入数组；axis=1-在第1维位置插入大小为1的新轴
    # 原因：PyTorch Conv2d要求输入格式为(batch, channels, height, width)
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

    # 将NumPy数组转换为PyTorch张量
    # torch.from_numpy()参数：ndarray-NumPy数组（共享底层内存，零拷贝）
    # 图像张量形状：(N, 1, 28, 28) float32
    train_images_tensor = torch.from_numpy(train_images)
    # 标签需要转为int64（即torch.long），CrossEntropyLoss要求target为long类型
    # .astype(np.int64)：将uint8转为64位整数
    train_labels_tensor = torch.from_numpy(train_labels.astype(np.int64))
    test_images_tensor = torch.from_numpy(test_images)
    test_labels_tensor = torch.from_numpy(test_labels.astype(np.int64))

    # 使用TensorDataset将图像和标签张量包装为数据集对象
    # TensorDataset()参数：*tensors-任意数量的张量，按第0维（样本索引）一一对应
    # 效果：可通过索引访问，如dataset[i]返回(images[i], labels[i])
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # 创建训练数据加载器
    # DataLoader()参数说明：
    #   dataset:    数据集对象（TensorDataset实例）
    #   batch_size: 每批加载的样本数（64）
    #   shuffle:    是否在每个epoch开始时随机打乱数据顺序（训练集应打乱以避免顺序偏差）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 创建测试数据加载器（不需要打乱顺序）
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 打印数据加载器信息
    # len(train_loader)：DataLoader的长度等于总批次数（向上取整）
    print(f"训练批次数：{len(train_loader)}，每批大小：{BATCH_SIZE}")
    print(f"测试批次数：{len(test_loader)}，每批大小：{BATCH_SIZE}")

    # ---------- 5. 创建CNN模型并打印架构 ----------

    # 实例化CNN模型（调用__init__初始化所有网络层）
    model = MNISTCNN()

    # 将模型的所有参数和缓冲区移动到目标设备
    # .to(device)：递归地将所有子模块的参数迁移到指定设备
    model = model.to(device)

    # 调用自定义函数打印模型各层的输入/输出维度详细信息
    print_model_layers(model)

    # ---------- 6. 定义损失函数和优化器 ----------

    # 定义交叉熵损失函数（多分类问题的标准损失函数）
    # nn.CrossEntropyLoss()：无参数，内部融合了LogSoftmax和NLLLoss
    # 输入要求：input为(N, C)的原始logits，target为(N,)的类别索引
    criterion = nn.CrossEntropyLoss()

    # 定义Adam优化器
    # optim.Adam()参数说明：
    #   params: 需要优化的参数，model.parameters()返回所有requires_grad=True的参数
    #   lr:     学习率（初始步长），控制每次参数更新的幅度
    # Adam特点：自适应调整每个参数的学习率，收敛快且对超参数不太敏感
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------- 7. 训练模型 ----------

    # 打印训练信息标题
    print("\n开始训练模型...")
    print(f"训练参数：epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print("-" * 70)

    # 主训练循环：遍历指定轮次
    # range(EPOCHS)：生成0到EPOCHS-1的整数序列
    for epoch in range(EPOCHS):

        # 执行一个epoch的训练
        # 参数：模型、训练加载器、损失函数、优化器、设备
        # 返回：该epoch的平均训练损失和训练准确率
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 在测试集上评估当前模型性能（不更新参数）
        # 参数：模型、测试加载器、损失函数、设备
        # 返回：平均测试损失和测试准确率
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 打印当前epoch的训练和测试指标
        # 格式化说明：epoch+1从1开始计数，:>2右对齐占2位，.4f保留4位小数，.2f保留2位小数
        print(
            f"Epoch [{epoch + 1:>2}/{EPOCHS}] "
            f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | "
            f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%"
        )

    print("-" * 70)

    # ---------- 8. 最终评估 ----------

    # 在全部训练集上计算最终准确率
    # _：用下划线丢弃不需要的返回值（损失值）
    _, final_train_acc = evaluate(model, train_loader, criterion, device)

    # 在全部测试集上计算最终准确率
    _, final_test_acc = evaluate(model, test_loader, criterion, device)

    # 打印最终结果
    print(f"\n========== 最终结果 ==========")
    print(f"训练集准确率：{final_train_acc:.2f}%")
    print(f"测试集准确率：{final_test_acc:.2f}%")

    # ---------- 9. 保存模型 ----------

    # 保存模型的状态字典（推荐方式：只保存可学习参数，不保存模型结构）
    # torch.save()参数说明：
    #   obj:  要保存的对象，model.state_dict()返回包含所有参数名和张量的OrderedDict
    #   f:    保存路径字符串或文件对象
    # 保存后文件大小约1.7MB（421,642个float32参数 × 4字节 ≈ 1.69MB）
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 打印保存确认信息
    print(f"\n模型已保存到：{MODEL_SAVE_PATH}")
    print("训练完成！请运行 predict_cnn.py 进行预测。")


# ==================== 程序入口 ====================

# 当该文件作为主程序直接运行时（而非被import时），执行main函数
# __name__是Python内置变量：直接运行时值为'__main__'，被import时值为模块名
if __name__ == '__main__':
    main()  # 调用主函数，启动完整的数据加载→模型构建→训练→测试→保存流程