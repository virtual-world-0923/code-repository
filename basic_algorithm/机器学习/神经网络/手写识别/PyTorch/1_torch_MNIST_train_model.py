# ========================== 1. 导入必要的第三方库 ==========================
import os  # 导入操作系统接口库，用于处理文件路径、判断文件是否存在等
import pandas as pd  # 导入数据处理库，用于读取和处理 CSV 格式的标签文件
from PIL import Image  # 导入图像处理库，用于打开和处理 PNG 格式的图片文件
import torch  # 导入 PyTorch 核心库，提供张量计算和深度学习功能
import torch.nn as nn  # 导入神经网络模块，包含各种网络层（如全连接层、激活函数等）
import torch.optim as optim  # 导入优化器模块，用于更新网络权重参数
from torch.utils.data import Dataset, DataLoader  # 导入数据集基类和数据加载器，用于批量加载数据
from torchvision import transforms  # 导入图像变换模块，用于对图片进行预处理（如转张量、归一化）

# ========================== 3. 自定义数据集类 ==========================
class MNISTCustomDataset(Dataset):
    """
    自定义数据集类，继承自 PyTorch 的 Dataset 基类
    作用：将本地的 PNG 图片和 CSV 标签文件封装成 PyTorch 可识别的数据集对象
    """

    def __init__(self, img_dir, csv_path, transform=None):
        """
        初始化数据集，加载标签信息和图像预处理规则
        参数:
            img_dir (str): 存放 PNG 图片的文件夹路径
            csv_path (str): 存放图片标签的 CSV 文件路径
            transform (callable, optional): 图像预处理转换函数，默认为 None
        """
        self.img_dir = img_dir  # 保存图片目录路径
        self.transform = transform  # 保存图像预处理函数
        # 使用 pandas 读取 CSV 文件，获取文件名与标签的对应关系
        self.labels_df = pd.read_csv(csv_path)

    def __len__(self):
        """
        返回数据集中样本的总数量
        作用：DataLoader 会根据这个长度来计算总共需要多少个批次 (batches)
        """
        return len(self.labels_df)  # 返回 CSV 文件中的行数，即样本总数

    def __getitem__(self, idx):
        """
        根据索引获取单个样本（图片和对应的标签）
        参数:
            idx (int): 样本的索引值
        返回:
            image (Tensor): 预处理后的图像张量
            label (Tensor): 图像对应的数字标签（长整型张量）
        """
        # 从 DataFrame 中获取当前索引对应的文件名（Original_Index 列）和标签（Label 列）
        row = self.labels_df.iloc[idx]  # iloc 用于按位置索引获取一行数据
        img_name = str(row['Original_Index']) + ".png"  # 拼接文件名，如 "888.png"
        label = int(row['Label'])  # 获取对应的数字标签，并转为整数类型

        # 拼接完整的图片路径并使用 PIL 打开图片
        img_path = os.path.join(self.img_dir, img_name)  # 组合目录和文件名得到绝对路径
        image = Image.open(img_path).convert("L")  # 打开图片并强制转换为灰度图 (L 模式)

        # 如果定义了图像预处理转换，则对图片进行转换（如转为张量）
        if self.transform:
            image = self.transform(image)  # 应用预处理，例如 ToTensor() 会将像素值归一化到 [0,1]

        # 将标签转换为 PyTorch 的长整型张量，分类任务标签必须是 LongTensor
        label = torch.tensor(label, dtype=torch.long)
        return image, label  # 返回处理好的图像张量和标签张量


# ========================== 4. 定义神经网络模型 ==========================
class SimpleNN(nn.Module):
    """
    简单的全连接神经网络 (Fully Connected Network)
    作用：接收展平后的 28x28 图片像素，通过隐藏层提取特征，最终输出 10 个类别的概率
    """

    def __init__(self):
        """
        初始化网络层结构
        """
        super(SimpleNN, self).__init__()  # 调用父类 nn.Module 的初始化方法
        # 展平层：用于将二维图像数据展平为 784 维的一维向量 [Batch_Size, 1, 28, 28] --> [Batch_Size, 784]
        self.flatten = nn.Flatten()
        # 第一个全连接层：接收 784 个特征，输出 128 个神经元
        self.fc1 = nn.Linear(28 * 28, 128)
        # 第二个全连接层：输入 128 个神经元，输出 128 个神经元
        self.fc2 = nn.Linear(128, 128)
        # 输出层：输入 128 个神经元，输出 10 个神经元（对应 0-9 十个数字类别）
        self.fc3 = nn.Linear(128, 10)
        self.tanh = nn.Tanh()  # 设置网络的激活函数


    def forward(self, x):
        """
        定义前向传播过程，即数据在神经网络中的流动路径
        参数:
            x (Tensor): 输入的图像张量，形状为 [Batch_Size, 1, 28, 28]
        返回:
            logits (Tensor): 模型输出的原始预测值，形状为 [Batch_Size, 10]
        """
        x = self.flatten(x)  # 将图像展平为 [Batch_Size, 784]
        x = self.tanh(self.fc1(x))  # 通过第一层全连接层并应用激活函数
        x = self.tanh(self.fc2(x))  # 通过第二层全连接层并应用激活函数
        logits = self.fc3(x)  # 通过输出层，得到 10 个类别的原始得分（不需要在这里加 Softmax）
        return logits  # 返回预测结果


# ========================== 5. 模型训练函数 ==========================
def train_model(model, train_loader, criterion, optimizer, epoch):
    """
    执行一个 Epoch 的训练过程
    参数:
        model (nn.Module): 神经网络模型实例
        train_loader (DataLoader): 训练集数据加载器
        criterion (nn.Module): 损失函数，用于衡量预测值与真实标签的差距
        optimizer (optim.Optimizer): 优化器，用于更新模型参数
        epoch (int): 当前的训练轮数
    """
    model.train()  # 将模型切换为训练模式（这会开启 Dropout 和 BatchNorm 的训练状态）
    running_loss = 0.0  # 初始化累计损失值，用于计算平均损失
    correct = 0  # 初始化预测正确的样本数
    total = 0  # 初始化总样本数

    # 遍历每一个批次的数据
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # 将数据和标签移动到计算设备上（GPU/CPU）

        optimizer.zero_grad()  # 清空上一步累积的梯度，防止梯度叠加导致更新错误
        outputs = model(images)  # 前向传播：将图片输入模型，得到预测结果
        loss = criterion(outputs, labels)  # 计算预测结果与真实标签之间的交叉熵损失
        loss.backward()  # 反向传播：根据损失自动计算网络中每个参数的梯度
        optimizer.step()  # 优化器根据梯度更新网络参数（权重和偏置）

        running_loss += loss.item()  # 累加当前批次的损失值（.item() 将张量转为 Python 数字）
        _, predicted = torch.max(outputs.data, 1)  # 获取预测概率最大的类别索引
        total += labels.size(0)  # 累加当前批次的样本数
        correct += (predicted == labels).sum().item()  # 累加预测正确的样本数

    # 计算并打印当前 Epoch 的平均损失和准确率
    avg_loss = running_loss / len(train_loader)  # 平均损失 = 累计损失 / 批次总数
    accuracy = 100 * correct / total  # 准确率 = 正确数 / 总数 * 100%
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%")


# ========================== 6. 模型测试/评估函数 ==========================
def test_model(model, test_loader, criterion):
    """
    在测试集上评估模型性能
    参数:
        model (nn.Module): 训练好的神经网络模型实例
        test_loader (DataLoader): 测试集数据加载器
        criterion (nn.Module): 损失函数
    返回:
        accuracy (float): 测试集上的准确率百分比
    """
    model.eval()  # 将模型切换为评估模式（关闭 Dropout 等训练专属操作）
    test_loss = 0.0  # 初始化测试集累计损失
    correct = 0  # 初始化预测正确的样本数
    total = 0  # 初始化总样本数

    with torch.no_grad():  # 上下文管理器：在测试时禁用梯度计算，节省显存并加速推理
        for images, labels in test_loader:  # 遍历测试集的每一个批次
            images, labels = images.to(device), labels.to(device)  # 数据移至计算设备
            outputs = model(images)  # 前向传播获取预测结果
            loss = criterion(outputs, labels)  # 计算测试集损失（仅用于监控，不反向传播）

            test_loss += loss.item()  # 累加损失
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)  # 累加样本总数
            correct += (predicted == labels).sum().item()  # 累加正确预测数

    avg_loss = test_loss / len(test_loader)  # 计算平均测试损失
    accuracy = 100 * correct / total  # 计算测试集准确率
    print(f"[Test] Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")  # 打印测试结果
    return accuracy  # 返回准确率


# ========================== 7. 自定义图片预测函数 ==========================
def predict_single_image(model, image_path):
    """
    加载一张外部 PNG 图片并预测其数字类别
    参数:
        model (nn.Module): 训练好的模型
        image_path (str): 待预测图片的完整路径
    返回:
        predicted_label (int): 预测出的数字 (0-9)
    """
    model.eval()  # 切换为评估模式
    # 定义单张图片的预处理流程：转灰度 -> 转张量 -> 增加批次维度
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图片是单通道灰度图
        transforms.ToTensor(),  # 将 PIL 图片转为 Tensor，并归一化到 [0,1]
    ])

    image = Image.open(image_path)  # 使用 PIL 打开指定路径的图片
    image = transform(image)  # 应用预处理
    image = image.unsqueeze(0).to(device)  # unsqueeze(0) 增加批次维度 [1, 28, 28]，并移至设备

    with torch.no_grad():  # 预测时不需要计算梯度
        output = model(image)  # 前向传播
        _, predicted = torch.max(output, 1)  # 获取概率最大的类别索引
        predicted_label = predicted.item()  # 将张量转为普通 Python 整数

    print(f"图片 {image_path} 的预测数字是: {predicted_label}")  # 打印预测结果
    return predicted_label  # 返回预测的数字


# ========================== 2. 配置设备与全局参数 ==========================
# 自动检测当前可用的计算设备：优先使用 NVIDIA GPU (cuda)，其次苹果 M 系列芯片 (mps)，最后使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"当前使用的计算设备: {device}")  # 打印当前使用的设备信息，方便确认是否成功调用 GPU

# 定义训练集和测试集的图片文件夹路径
TRAIN_IMG_DIR = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/train_sampled/images"
TEST_IMG_DIR = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/test_sampled/images"

# 定义训练集和测试集的标签 CSV 文件路径
TRAIN_CSV_PATH = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/train_sampled/labels_summary.csv"
TEST_CSV_PATH = r"/database/torchMNIST/MNIST/raw/MNIST_Sampled_Data/test_sampled/labels_summary.csv"

# 定义超参数
BATCH_SIZE = 64  # 批次大小，每次训练喂给模型 64 张图片，平衡显存占用与训练速度
LEARNING_RATE = 0.001  # 学习率，控制模型参数更新的步长，过大可能不收敛，过小收敛慢
EPOCHS = 15  # 训练轮数，整个训练集被模型完整学习的次数

if __name__ == "__main__":
    # 1. 定义图像预处理：将 PIL 图片数据转换为 PyTorch的Tensor格式，并自动归一化像素值到 [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])

    # 2. 实例化自定义数据集，传入图片目录、CSV路径和预处理规则
    train_dataset = MNISTCustomDataset(TRAIN_IMG_DIR, TRAIN_CSV_PATH, transform=transform)
    test_dataset = MNISTCustomDataset(TEST_IMG_DIR, TEST_CSV_PATH, transform=transform)

    # 3. 创建数据加载器，实现数据的批量读取和打乱（shuffle=True 防止模型记住数据顺序）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 测试集不需要打乱

    # 4. 实例化神经网络模型，并将其移动到指定的计算设备（GPU/CPU）上
    model = SimpleNN().to(device)

    # 5. 定义损失函数：交叉熵损失，适用于多分类任务（内部已包含 Softmax）
    criterion = nn.CrossEntropyLoss()
    # 6. 定义优化器：Adam 优化器，自适应调整学习率，收敛速度快且稳定
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 开始循环训练
    for epoch in range(EPOCHS):
        train_model(model, train_loader, criterion, optimizer, epoch)  # 执行一轮训练

    # 8. 训练结束后，在测试集上评估最终准确率
    test_model(model, test_loader, criterion)

    # 9. 保存训练好的模型权重到本地文件，方便以后直接加载使用
    torch.save(model.state_dict(), "mnist_model.pth")
    print("模型已成功保存为 mnist_model.pth")

    # 10. 演示：预测一张自己保存的图片（请确保该路径下存在一张 28x28 的 PNG 图片）
    # 注意：运行此行前，请替换为您本地实际存在的图片路径
    # predict_single_image(model, "my_own_digit.png")