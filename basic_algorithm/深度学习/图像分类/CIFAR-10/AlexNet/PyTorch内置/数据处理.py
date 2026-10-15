import numpy as np                                     # 导入numpy模块，用于数值计算和多维数组操作
from torch.utils.data import Dataset, DataLoader       # 从PyTorch工具库导入Dataset基类和DataLoader批量加载器
from torchvision import transforms                     # 从torchvision导入数据预处理与增强的变换工具模块
import database.read_cifar as rc

#读取数据
d = rc.read_data()
train_images = d["train_images"]                     # b'data'键对应的值为形状(10000, 3072)的numpy数组，每行是一张32×32×3图片的展平像素
train_labels = d["train_labels"]        # b'labels'键对应标签列表，转换为numpy数组便于后续处理，shape为(10000,)
test_images = d["test_images"]                       # 形状为(10000, 3072)的numpy数组
test_labels = d["test_labels"]           # 形状为(10000,)的标签数组
label_names = d["label_names"]
"""
# 分离三个通道: 每张图片的前1024个值是红色通道，接着1024绿色，最后1024蓝色
r_channel = train_images[:, :1024].reshape(-1, 32, 32)  # (N, 32, 32)
g_channel = train_images[:, 1024:2048].reshape(-1, 32, 32)  # (N, 32, 32)
b_channel = train_images[:, 2048:].reshape(-1, 32, 32)  # (N, 32, 32)
# 沿最后一个维度堆叠为 (N, 32, 32, 3)，符合HWC格式，方便torchvision的ToTensor处理
data = np.stack([r_channel, g_channel, b_channel], axis=-1)
"""
# bCHW <-> bHWC
def _dimension_transpose(data, location:tuple=(0,2,3,1)):
    if data.ndim == 2:
        data = data.reshape(-1, 3, 32, 32)
    if data.ndim == 4:
        data = np.transpose(data, location)
    return data

#pytorch 处理图片的格式：bCHW，这里转换是为了下面transforms.Compose.ToPILImage 使用HWC格式，在ToTensor的时候会在转回CHW格式
train_images = _dimension_transpose(train_images)
train_labels = _dimension_transpose(train_labels)
test_images =  _dimension_transpose(test_images)
test_labels = _dimension_transpose(test_labels)

# --- 训练集预处理管道（包含数据增强） ---
# # Compose将多个变换操作串联为一个管道，按列表顺序依次执行
train_transform = transforms.Compose([
    transforms.ToPILImage(),                  # 将传入的值转换为PIL Image对象，因为后面的操作都需要这个对象类型。torch.Tensor=(C, H, W)，numpy.ndarray=(H, W, C)
    transforms.Resize((227, 227)),            # 将32×32的CIFAR-10图片缩放至227×227，匹配AlexNet网络的标准输入尺寸
    transforms.RandomHorizontalFlip(p=0.5),   # 以50%概率水平翻转（左右镜像）图片，作为数据增强手段减少过拟合
    transforms.ToTensor(),                    # 将PIL Image转回torch.Tensor：(HWC→CHW, [0,255]（uint8）→ [0.0,1.0]（float32）)
    transforms.Normalize(                     # 对图片张量进行标准化（减均值除标准差），使数据分布与ImageNet预训练时一致
        mean=[0.485, 0.456, 0.406],           # ImageNet训练集RGB三通道的均值
        std=[0.229, 0.224, 0.225]             # ImageNet训练集RGB三通道的标准差
    )
])

# --- 测试集预处理管道（不做数据增强，保持确定性） ---
test_transform = transforms.Compose([                  # 测试集不需要随机增强，仅做必要的尺寸转换和标准化
    transforms.ToPILImage(),                            # 将numpy数组转换为PIL Image
    transforms.Resize((227, 227)),                      # 缩放至227×227以匹配AlexNet输入要求
    transforms.ToTensor(),                              # 转换为torch.Tensor格式，像素值归一化到[0.0, 1.0]
    transforms.Normalize(                               # 使用与训练集相同的均值和标准差进行标准化
        mean=[0.485, 0.456, 0.406],                     # ImageNet RGB均值
        std=[0.229, 0.224, 0.225]                       # ImageNet RGB标准差
    )
])
# ==================== 第五部分：定义自定义CIFAR-10数据集类 ====================
class CIFAR10DataSet(Dataset):                           # 继承PyTorch的Dataset抽象基类，需实现__len__和__getitem__方法
    """
    自定义本地CIFAR-10数据集类
    将numpy数组格式的CIFAR-10数据封装为PyTorch Dataset，供DataLoader使用

    参数说明:
        images   (numpy.ndarray): 图片像素数据，形状为(N, 3072)，N为样本数
        labels   (numpy.ndarray): 标签数据，形状为(N,)，每个元素为0-9的整数
        transform (callable, 可选): 数据预处理/增强变换函数，应用于每张图片
    """

    def __init__(self, images, labels, transform=None):  # 构造函数，接收图片数据、标签数据和可选的变换函数
        self.images = images               # 将图片numpy数组保存为实例属性，shape为(N, 3072)
        self.labels = labels               # 将标签numpy数组保存为实例属性，shape为(N,)
        self.transform = transform         # 将变换函数保存为实例属性，可能为None

    def __len__(self):                     # 返回数据集的总样本数，DataLoader依赖此方法确定遍历长度
        return len(self.labels)            # 返回标签数组的长度即为样本总数

    def __getitem__(self, idx):            # 根据索引idx获取单个样本，DataLoader的每个worker都会调用此方法

        image = self.images[idx]

        # 应用数据预处理变换（如果已定义）
        if self.transform is not None:      # 检查transform是否已定义（非None）
            image = self.transform(image)   # 执行变换管道：可能包含ToPILImage→Resize→Augmentation→ToTensor→Normalize

        # 将标签转换为Python整数类型
        label = int(self.labels[idx])       # 将numpy整数转为Python原生int，CrossEntropyLoss需要long类型标签

        return image, label                 # 返回处理后的图片张量和标签整数，DataLoader会自动将它们组装为batch

# ==================== 第七部分：创建Dataset和DataLoader ====================
# 实例化训练数据集对象
train_dataset = CIFAR10DataSet(                          # 调用自定义数据集类的构造函数
    images=train_images,                                # 传入训练图片numpy数组，shape为(10000, 3072)
    labels=train_labels,                                # 传入训练标签numpy数组，shape为(10000,)
    transform=train_transform                           # 传入训练集预处理变换管道
)

# 实例化测试数据集对象
test_dataset = CIFAR10DataSet(                           # 调用自定义数据集类的构造函数
    images=test_images,                                 # 传入测试图片numpy数组，shape为(10000, 3072)
    labels=test_labels,                                 # 传入测试标签numpy数组，shape为(10000,)
    transform=test_transform                            # 传入测试集预处理变换管道
)

# 创建训练数据DataLoader，负责按batch批量加载训练数据
train_loader = DataLoader(                             # DataLoader封装Dataset，提供批量加载、多进程、打乱等功能
    dataset=train_dataset,                              # 传入训练数据集对象
    batch_size=64,                                      # 每个batch加载64张图片，平衡内存占用和训练效率
    shuffle=True,                                       # 每个epoch开始时打乱数据顺序，防止模型学习到数据的排列规律
    num_workers=0,                                      # 数据加载的工作进程数，0表示在主进程中加载（Windows系统下建议设为0避免多进程问题）
    drop_last=True                                      # 丢弃最后一个不完整batch（当样本数不能被batch_size整除时），保持所有batch大小一致
)

# 创建测试数据DataLoader
test_loader = DataLoader(                              # DataLoader封装测试数据集
    dataset=test_dataset,                               # 传入测试数据集对象
    batch_size=64,                                      # 测试时每个batch也加载64张图片
    shuffle=False,                                      # 测试集不需要打乱顺序，保持原始顺序便于结果分析
    num_workers=0,                                      # 主进程中加载数据
    drop_last=False                                     # 测试集保留所有样本，不丢弃最后不完整batch
)