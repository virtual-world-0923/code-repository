# ============================================================
# 文件名：train_and_test.py
# 功能：使用PyTorch搭建CNN网络，训练MNIST手写数字识别模型，
#       打印每层维度信息，输出准确率，并保存模型参数
# 环境：Python 3.13.14 / PyTorch 2.13.0+cpu / numpy 2.5.1
# ============================================================

# 导入Python标准库中的struct模块，用于解析IDX二进制文件中的字节数据
import struct
# 导入Python标准库中的os模块，用于处理文件路径拼接和路径存在性检查
import os
# 导入numpy库并简写为np，用于将二进制数据转换为多维数组进行数值计算
import numpy as np
# 导入PyTorch核心库并简写为torch，提供张量运算和自动求导功能
import torch
# 从torch中导入nn模块，该模块包含构建神经网络所需的所有层和容器类
import torch.nn as nn
# 从torch中导入optim模块，该模块包含各种优化器算法（如SGD、Adam等）
import torch.optim as optim
# 从torch.utils.data中导入Dataset基类，用于自定义数据集类必须继承该类
from torch.utils.data import Dataset
# 从torch.utils.data中导入DataLoader类，用于将数据集封装为可迭代的小批次加载器
from torch.utils.data import DataLoader

# ============================================================
# 第一部分：定义MNIST IDX文件读取函数
# ============================================================

def read_idx_images(file_path):
    """
    读取MNIST IDX格式的图片文件，返回numpy数组
    参数：
        file_path (str): IDX图片文件的完整路径，
                         例如 'D:\\Repositories\\basic_algorithm\\database\\mnist\\train-images-idx3-ubyte'
    返回：
        images (numpy.ndarray): 形状为 (num_images, 28, 28) 的uint8数组，
                                每个元素表示一个像素的灰度值(0-255)
    """
    # 以二进制只读模式('rb')打开IDX图片文件，返回文件对象f
    with open(file_path, 'rb') as f:
        # 读取文件前16个字节：前4字节是魔数(0x00000803)，
        # 接下来4字节是图片数量，再4字节是行数，最后4字节是列数
        # '>IIII'表示大端序(>)的4个无符号整型(I)，每个占4字节
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 读取剩余所有字节数据，即所有图片的像素原始数据
        # 每张图片占 rows*cols=784 字节，共 num_images*784 字节
        raw_data = f.read()
    # 使用numpy的frombuffer将原始字节数据解析为uint8类型的一维数组
    # dtype=np.uint8表示每个字节对应一个0-255的无符号整数
    images = np.frombuffer(raw_data, dtype=np.uint8)
    # 将一维数组重塑为三维数组 (num_images, rows, cols)，即(60000,28,28)或(10000,28,28)
    images = images.reshape(num_images, rows, cols)
    # 返回重塑后的图片数组
    return images


def read_idx_labels(file_path):
    """
    读取MNIST IDX格式的标签文件，返回numpy数组
    参数：
        file_path (str): IDX标签文件的完整路径，
                         例如 'D:\\Repositories\\basic_algorithm\\database\\mnist\\train-labels-idx1-ubyte'
    返回：
        labels (numpy.ndarray): 形状为 (num_labels,) 的uint8数组，
                                每个元素是对应图片的数字标签(0-9)
    """
    # 以二进制只读模式('rb')打开IDX标签文件，返回文件对象f
    with open(file_path, 'rb') as f:
        # 读取文件前8个字节：前4字节是魔数(0x00000801)，
        # 接下来4字节是标签数量
        # '>II'表示大端序(>)的2个无符号整型(I)，每个占4字节
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 读取剩余所有字节数据，即所有标签值，每个标签占1字节
        raw_data = f.read()
    # 使用numpy的frombuffer将原始字节数据解析为uint8类型的一维数组
    # 每个字节值即为对应图片的数字标签(0-9)
    labels = np.frombuffer(raw_data, dtype=np.uint8)
    # 返回标签数组，形状为 (num_labels,)
    return labels


# ============================================================
# 第二部分：自定义PyTorch数据集类
# ============================================================

class MNISTDataset(Dataset):
    """
    自定义MNIST数据集类，继承自torch.utils.data.Dataset
    作用：将numpy格式的图片数组和标签数组封装为PyTorch Dataset对象，
          使其能被DataLoader按批次加载
    """

    def __init__(self, images, labels):
        """
        初始化数据集对象
        参数：
            images (numpy.ndarray): 形状为 (N, 28, 28) 的图片数组，dtype为uint8
            labels (numpy.ndarray): 形状为 (N,) 的标签数组，dtype为uint8
        """
        # 将图片数组从uint8转换为float32，并除以255.0归一化到[0,1]区间
        # 然后增加一个通道维度，从(N,28,28)变为(N,1,28,28)，符合CNN输入要求
        # torch.from_numpy将numpy数组转为PyTorch张量，不复制内存
        self.images = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
        # 将标签数组从uint8转换为int64(long)类型，这是PyTorch交叉熵损失函数要求的标签类型
        # torch.from_numpy将numpy数组转为PyTorch张量
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        """
        返回数据集中样本的总数
        返回值：
            int: 样本总数，训练集为60000，测试集为10000
        """
        # 返回图片张量第0维的大小，即样本数量
        return len(self.images)

    def __getitem__(self, index):
        """
        根据索引返回单个样本（DataLoader内部调用此方法获取数据）
        参数：
            index (int): 样本索引，范围 [0, len(self)-1]
        返回值：
            image (torch.Tensor): 形状为 (1, 28, 28) 的float32张量，表示单张灰度图片
            label (torch.Tensor): 标量int64张量，表示该图片的数字标签(0-9)
        """
        # 根据索引取出对应的图片张量并返回
        image = self.images[index]
        # 根据索引取出对应的标签张量并返回
        label = self.labels[index]
        # 返回图片-标签对，DataLoader会自动将多个样本堆叠为批次
        return image, label


# ============================================================
# 第三部分：定义CNN网络架构
# ============================================================

class CNN(nn.Module):
    """
    卷积神经网络类，继承自nn.Module
    网络结构：
        输入 -> Conv2d(1,32) -> ReLU -> MaxPool2d
             -> Conv2d(32,64) -> ReLU -> MaxPool2d
             -> Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout
             -> Linear(128, 10) -> 输出
    """

    def __init__(self):
        """
        初始化网络层，定义所有可学习参数的层
        """
        # 调用父类nn.Module的__init__方法，这是PyTorch模块初始化的必要步骤
        super(CNN, self).__init__()

        # 定义第一个卷积层：输入通道1(灰度图)，输出通道32，卷积核大小3x3，步长1，填充1
        # 填充1使得经过3x3卷积后空间尺寸不变：28x28 -> 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 定义第一个ReLU激活函数，将负值置零引入非线性
        self.relu1 = nn.ReLU()

        # 定义第一个最大池化层：池化窗口2x2，步长2
        # 将空间尺寸减半：28x28 -> 14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义第二个卷积层：输入通道32，输出通道64，卷积核大小3x3，步长1，填充1
        # 空间尺寸不变：14x14 -> 14x14
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 定义第二个ReLU激活函数
        self.relu2 = nn.ReLU()

        # 定义第二个最大池化层：池化窗口2x2，步长2
        # 将空间尺寸减半：14x14 -> 7x7
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 【关键修复】定义显式的Flatten层，将多维特征图展平为一维向量
        # start_dim=1表示从第1维开始展平（保留第0维batch维度）
        # 输入形状 (batch, 64, 7, 7) -> 输出形状 (batch, 3136)
        # 这样在named_children()遍历时，Flatten会作为一个独立层被正确执行
        self.flatten = nn.Flatten(start_dim=1)

        # 定义第一个全连接层：输入特征数 64*7*7=3136，输出特征数 128
        # 64是第二个卷积层的输出通道数，7x7是经过两次池化后的空间尺寸
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)

        # 定义第三个ReLU激活函数，用于全连接层后
        self.relu3 = nn.ReLU()

        # 定义Dropout层，训练时以50%的概率随机将神经元输出置零，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

        # 定义第二个全连接层（输出层）：输入特征数 128，输出特征数 10（对应数字0-9共10个类别）
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        """
        定义前向传播过程，描述数据如何从输入流经各层到达输出
        参数：
            x (torch.Tensor): 输入张量，形状为 (batch_size, 1, 28, 28)
        返回值：
            x (torch.Tensor): 输出张量，形状为 (batch_size, 10)，即每个样本对10个类别的logits
        """
        # 将输入x通过第一个卷积层，提取低级特征（边缘、纹理等）
        # 输入形状: (batch, 1, 28, 28) -> 输出形状: (batch, 32, 28, 28)
        x = self.conv1(x)
        # 通过第一个ReLU激活函数，将负值置零，引入非线性变换能力
        x = self.relu1(x)
        # 通过第一个最大池化层，降低空间分辨率，减少计算量
        # 输入形状: (batch, 32, 28, 28) -> 输出形状: (batch, 32, 14, 14)
        x = self.pool1(x)

        # 将特征图通过第二个卷积层，提取更高级的特征（形状、结构等）
        # 输入形状: (batch, 32, 14, 14) -> 输出形状: (batch, 64, 14, 14)
        x = self.conv2(x)
        # 通过第二个ReLU激活函数
        x = self.relu2(x)
        # 通过第二个最大池化层，进一步降低空间分辨率
        # 输入形状: (batch, 64, 14, 14) -> 输出形状: (batch, 64, 7, 7)
        x = self.pool2(x)

        # 【关键修复】通过显式的Flatten层将三维特征图展平为一维向量
        # 输入形状: (batch, 64, 7, 7) -> 输出形状: (batch, 3136)
        # 保留第0维(batch维度)不变，其余维度全部展平
        x = self.flatten(x)

        # 通过第一个全连接层，将3136维特征映射到128维隐藏表示
        # 输入形状: (batch, 3136) -> 输出形状: (batch, 128)
        x = self.fc1(x)
        # 通过第三个ReLU激活函数
        x = self.relu3(x)
        # 通过Dropout层，训练时随机丢弃50%的神经元，增强泛化能力
        # 注意：在model.eval()模式下Dropout自动关闭
        x = self.dropout(x)

        # 通过第二个全连接层（输出层），将128维映射到10维（10个数字类别）
        # 输入形状: (batch, 128) -> 输出形状: (batch, 10)
        x = self.fc2(x)
        # 返回最终的logits输出（未经softmax，因为CrossEntropyLoss内部会处理）
        return x


# ============================================================
# 第四部分：打印网络每层的输入输出维度
# ============================================================

def print_layer_dimensions(model, input_size=(1, 1, 28, 28)):
    from torchinfo import summary
    summary(model, input_size=input_size)


# ============================================================
# 第五部分：训练函数
# ============================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    训练CNN模型
    参数：
        model (nn.Module): 待训练的CNN模型实例
        train_loader (DataLoader): 训练数据的批次加载器
        criterion (nn.Module): 损失函数，这里使用交叉熵损失
        optimizer (optim.Optimizer): 优化器，用于更新模型参数
        num_epochs (int): 训练的轮数（epoch数），默认为5
    """
    # 将模型设置为训练模式，启用Dropout和BatchNorm的训练行为
    model.train()

    # 外层循环：遍历每个训练轮次(epoch)
    for epoch in range(num_epochs):
        # 初始化当前epoch的累计损失值
        running_loss = 0.0
        # 初始化当前epoch正确预测的样本计数
        correct = 0
        # 初始化当前epoch处理的总样本计数
        total = 0

        # 内层循环：遍历训练数据加载器中的每个批次(batch)
        # 每次迭代返回一批图片images和对应标签labels
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 将优化器中所有参数的梯度清零，防止梯度累积
            # PyTorch默认会累积梯度，所以每次反向传播前必须清零
            optimizer.zero_grad()

            # 将一批图片输入模型，得到每个样本对10个类别的预测logits
            # images形状: (batch_size, 1, 28, 28)
            # outputs形状: (batch_size, 10)
            outputs = model(images)

            # 计算预测logits与真实标签之间的交叉熵损失
            # criterion内部会自动对logits做softmax再取负对数似然
            # labels形状: (batch_size,)，每个元素是0-9的整数
            loss = criterion(outputs, labels)

            # 反向传播：计算损失对所有模型参数的梯度
            # 梯度会存储在每个参数的.grad属性中
            loss.backward()

            # 根据计算出的梯度更新模型参数
            # 具体更新规则由优化器类型决定（如Adam、SGD等）
            optimizer.step()

            # 将当前批次的损失值（标量）累加到running_loss中
            # loss.item()将单元素张量转为Python浮点数
            running_loss += loss.item()

            # 使用torch.max获取每个样本预测概率最大的类别索引
            # outputs形状(batch,10)，dim=1表示沿类别维度取最大值
            # predicted形状: (batch_size,)，每个元素是预测的类别(0-9)
            _, predicted = torch.max(outputs, 1)

            # 累加当前批次的样本数到总计数
            total += labels.size(0)
            # 累加当前批次中预测正确的样本数
            # (predicted == labels)生成布尔张量，.sum()统计True的个数
            correct += (predicted == labels).sum().item()

        # 计算当前epoch的平均损失（总损失 / 批次数）
        avg_loss = running_loss / len(train_loader)
        # 计算当前epoch在训练集上的准确率（正确数 / 总数 * 100%）
        accuracy = 100.0 * correct / total
        # 打印当前epoch的训练信息：轮次、平均损失、准确率
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {avg_loss:.4f}, "
              f"Train Accuracy: {accuracy:.2f}%")


# ============================================================
# 第六部分：测试/评估函数
# ============================================================

def evaluate_model(model, data_loader, dataset_name="Test"):
    """
    在给定数据集上评估模型准确率
    参数：
        model (nn.Module): 已训练好的CNN模型实例
        data_loader (DataLoader): 评估数据的批次加载器（可以是训练集或测试集）
        dataset_name (str): 数据集名称字符串，用于打印显示，默认为"Test"
    返回值：
        accuracy (float): 该数据集上的分类准确率（百分比）
    """
    # 将模型设置为评估模式，关闭Dropout，固定BatchNorm的统计量
    model.eval()

    # 初始化正确预测的样本计数
    correct = 0
    # 初始化处理的总样本计数
    total = 0

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    # 评估时不需要计算梯度，禁用后可以节省内存和加速计算
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for images, labels in data_loader:
            # 将一批图片输入模型，得到预测logits
            outputs = model(images)
            # 获取每个样本预测概率最大的类别索引
            _, predicted = torch.max(outputs, 1)
            # 累加当前批次的样本数
            total += labels.size(0)
            # 累加当前批次中预测正确的样本数
            correct += (predicted == labels).sum().item()

    # 计算准确率百分比
    accuracy = 100.0 * correct / total
    # 打印该数据集的准确率信息
    print(f"{dataset_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    # 返回准确率数值
    return accuracy


# ============================================================
# 第七部分：主程序入口
# ============================================================

if __name__ == "__main__":
    # 定义MNIST数据文件所在的本地目录路径
    # 使用原始字符串r""避免反斜杠被解释为转义字符
    data_dir = r"D:\Repositories\basic_algorithm\database\mnist"

    # 拼接训练图片文件的完整路径
    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
    # 拼接训练标签文件的完整路径
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
    # 拼接测试图片文件的完整路径
    test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    # 拼接测试标签文件的完整路径
    test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    # 打印数据加载提示信息
    print("正在加载MNIST数据...")
    # 调用自定义函数读取训练图片，返回形状(60000, 28, 28)的numpy数组
    train_images = read_idx_images(train_images_path)
    # 调用自定义函数读取训练标签，返回形状(60000,)的numpy数组
    train_labels = read_idx_labels(train_labels_path)
    # 调用自定义函数读取测试图片，返回形状(10000, 28, 28)的numpy数组
    test_images = read_idx_images(test_images_path)
    # 调用自定义函数读取测试标签，返回形状(10000,)的numpy数组
    test_labels = read_idx_labels(test_labels_path)
    # 打印数据加载完成信息及数据形状
    print(f"训练集: 图片{train_images.shape}, 标签{train_labels.shape}")
    print(f"测试集: 图片{test_images.shape}, 标签{test_labels.shape}")

    # 创建训练集的Dataset对象，将numpy数据封装为PyTorch可用的格式
    train_dataset = MNISTDataset(train_images, train_labels)
    # 创建测试集的Dataset对象
    test_dataset = MNISTDataset(test_images, test_labels)

    # 创建训练集的DataLoader，设置批次大小为64，打乱数据顺序(shuffle=True)
    # num_workers=0表示在主进程中加载数据（Windows下推荐设为0避免多进程问题）
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    # 创建测试集的DataLoader，设置批次大小为64，不打乱顺序(shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 实例化CNN模型对象
    model = CNN()
    # 打印模型的完整结构（所有层及其参数）
    print("\n模型结构：")
    print(model)

    # 打印网络每层的输入输出维度信息
    print()
    print_layer_dimensions(model)

    # 定义损失函数：交叉熵损失，适用于多分类问题
    # 内部包含LogSoftmax + NLLLoss，输入为logits，标签为类别索引
    criterion = nn.CrossEntropyLoss()

    # 定义优化器：Adam优化器，学习率设为0.001
    # model.parameters()返回模型所有可学习参数的迭代器
    # Adam结合了动量和自适应学习率的优点，收敛速度快
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印开始训练的提示信息
    print("\n开始训练...")
    # 调用训练函数，训练5个epoch
    train_model(model, train_loader, criterion, optimizer, num_epochs=3)

    # 打印评估提示信息
    print("\n评估模型...")
    # 在训练集上评估准确率（检查是否过拟合或欠拟合）
    evaluate_model(model, train_loader, dataset_name="Train")
    # 在测试集上评估准确率（反映模型的真实泛化能力）
    evaluate_model(model, test_loader, dataset_name="Test")

    # 定义模型保存的文件路径
    model_save_path = os.path.join("./", "mnist_cnn_model.pth")
    # 使用torch.save保存模型的state_dict（参数字典）
    # state_dict是一个有序字典，键为参数名(如'conv1.weight')，值为参数张量
    # 这种方式比保存整个模型对象更推荐，因为不依赖类的定义路径
    torch.save(model.state_dict(), model_save_path)
    # 打印模型保存成功的提示信息及保存路径
    print(f"\n模型已保存至: {model_save_path}")