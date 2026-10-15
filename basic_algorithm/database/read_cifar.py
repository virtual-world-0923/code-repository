import os
import pickle
import numpy as np
from pathlib import Path

# 当前文件路径
# current_file = Path(__file__).resolve()
# print(current_file)

# 当前文件所在目录
current_dir = Path(__file__).resolve().parent

# 拼接路径
meta_path = current_dir / "cifar-10-batches-py" / "batches.meta"
print(meta_path)
train_path = current_dir / "cifar-10-batches-py" / "data_batch_1"
print(train_path)
test_path = current_dir / "cifar-10-batches-py" / "test_batch"
print(test_path)

def read_data(size=None):
    """
    （32x32）的3通道彩色图片
    图片：形状(10000, 3072)，前1024红色通道值，中1024绿色通道值，后1024蓝色通道值
    标签：(10000,) 0~9 10分类
    :return:dict
        "train_images" = 训练图片
        "train_labels" = 训练标签
        "test_images" = 测试图片
        "test_labels" = 测试标签
        "label_names" = list,10类名称
    """
    # 以二进制读取模式打开pickle格式的元信息文件并反序列化
    with open(meta_path, "rb") as f:                       # "rb"表示以二进制只读模式打开文件；with语句确保文件最终被正确关闭
        meta = pickle.load(f, encoding="latin1")               # pickle.load从文件对象f中读取并反序列化，还原为原始Python字典对象

    # 从元信息字典中提取类别名称列表（pickle中键为字节字符串格式）
    label_names = meta["label_names"]                     # b"label_names"是字节字符串键，对应10个类别的名称列表
    print("CIFAR-10 十个类别名称:", label_names)              # 打印类别名称确认数据正确加载，如 ['airplane', 'automobile', ...]

    # ==================== 第三部分：加载训练数据 ====================

    # 以二进制读取模式打开训练数据文件并反序列化为Python字典
    with open(train_path, "rb") as f:                      # 以二进制只读模式打开文件
        train_data = pickle.load(f, encoding="latin1")                        # 反序列化得到字典，包含 'data'、'labels'、'filenames'、'batch_label' 等键

    # 从训练数据字典中提取图片数据和标签数据
    train_images = train_data["data"]                     # b'data'键对应的值为形状(10000, 3072)的numpy数组，每行是一张32×32×3图片的展平像素
    train_labels = np.array(train_data["labels"])         # b'labels'键对应标签列表，转换为numpy数组便于后续处理，shape为(10000,)

    # 以二进制读取模式打开测试数据文件并反序列化
    with open(test_path, "rb") as f:                       # 以二进制只读模式打开文件
        test_data = pickle.load(f, encoding="latin1")                          # 反序列化得到与训练数据格式相同的字典

    # 从测试数据字典中提取图片数据和标签数据
    test_images = test_data["data"]                       # 形状为(10000, 3072)的numpy数组
    test_labels = np.array(test_data["labels"])           # 形状为(10000,)的标签数组

    d = dict()
    d["train_images"] = train_images[:size]
    d["train_labels"] = train_labels[:size]
    d["test_images"] = test_images[:size]
    d["test_labels"] = test_labels[:size]
    d["label_names"] = label_names
    return d

def read_data1(size=None):
    """
    （32x32）的3通道彩色图片
    图片：形状(10000, 3072)，前1024红色通道值，中1024绿色通道值，后1024蓝色通道值
    标签：(10000,) 0~9 10分类
    :return:dict
        "train_images" = 训练图片
        "train_labels" = 训练标签
        "test_images" = 测试图片
        "test_labels" = 测试标签
        "label_names" = list,10类名称
    """
    # 以二进制只读模式打开指定路径的文件，with语句自动管理文件关闭
    with open(meta_path, 'rb') as fo:
        # 使用pickle加载二进制文件，指定encoding='bytes'兼容Python2生成的CIFAR-10原始文件
        # 若不指定编码，Python3加载Python2生成的pickle会出现解码错误
        dict_data = pickle.load(fo, encoding='bytes')
    class_names = [name.decode('utf-8') for name in dict_data[b'label_names']]

    with open(train_path, 'rb') as fo:
        # 使用pickle加载二进制文件，指定encoding='bytes'兼容Python2生成的CIFAR-10原始文件
        # 若不指定编码，Python3加载Python2生成的pickle会出现解码错误
        train_dict = pickle.load(fo, encoding='bytes')
    # 提取训练图像原始数据，形状为(10000, 3072)，3072 = 3通道 × 32高 × 32宽
    train_data = train_dict[b'data']
    # 提取训练标签列表，每个元素为0-9的整数，对应10个类别
    train_labels = train_dict[b'labels']

    with open(test_path, 'rb') as fo:
        # 使用pickle加载二进制文件，指定encoding='bytes'兼容Python2生成的CIFAR-10原始文件
        # 若不指定编码，Python3加载Python2生成的pickle会出现解码错误
        test_dict = pickle.load(fo, encoding='bytes')
    test_data = test_dict[b'data']
    # 提取测试标签列表
    test_labels = test_dict[b'labels']



    d = dict()
    d["train_data"] = train_data[:size]
    d["train_labels"] = train_labels[:size]
    d["test_data"] = test_data[:size]
    d["test_labels"] = test_labels[:size]
    d["class_names"] = class_names
    return d