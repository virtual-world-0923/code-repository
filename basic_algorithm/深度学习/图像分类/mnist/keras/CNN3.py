# -*- coding: utf-8 -*-
"""
CNN 手写数字识别 —— 模型训练与测试
使用 Keras 3 Functional API 搭建 CNN，在本地 MNIST IDX 数据上训练和评估
"""

# ======================== 导入依赖 ========================
import os                                # 用于拼接文件路径、检查文件是否存在
import struct                            # 用于解析 IDX 二进制文件头部（大端序整数）
import numpy as np                       # 用于数组运算、归一化、reshape 等数值操作
import keras                             # Keras 3 主模块，提供模型构建、训练、保存等 API
from keras import layers                 # 从 Keras 导入 layers 子模块，包含 Conv2D / MaxPooling2D 等层类


# ======================== 1. 定义 IDX 文件加载函数 ========================

def load_mnist_images(filepath):
    """
    读取 IDX 格式的图像二进制文件，返回四维 numpy 数组。
    IDX 图像文件头部结构：magic(4B) + 数量(4B) + 行数(4B) + 列数(4B) + 像素数据。
    参数:
        filepath (str): IDX 图像文件的完整路径，例如 'train-images-idx3-ubyte'。
    返回:
        np.ndarray: 形状为 (N, 28, 28, 1) 的 uint8 数组，N 为图像总数。
    """
    with open(filepath, 'rb') as f:                              # 以二进制只读模式打开文件
        magic = struct.unpack('>I', f.read(4))[0]                # 读取前 4 字节并按大端序无符号整数解码，得到魔数（用于校验文件类型）
        num_images = struct.unpack('>I', f.read(4))[0]           # 读取接下来 4 字节，解码为图像总数（训练集 60000，测试集 10000）
        rows = struct.unpack('>I', f.read(4))[0]                 # 读取接下来 4 字节，解码为每张图像的高度（像素行数，MNIST 为 28）
        cols = struct.unpack('>I', f.read(4))[0]                 # 读取接下来 4 字节，解码为每张图像的宽度（像素列数，MNIST 为 28）
        raw = np.frombuffer(f.read(), dtype=np.uint8)            # 将文件剩余全部字节一次性读入，按 uint8（0-255）解析为一维数组
        images = raw.reshape(num_images, rows, cols, 1)          # 将一维数组重塑为四维：(样本数, 高, 宽, 通道数)，通道数为 1 表示灰度图
    return images                                                # 返回形状为 (N, 28, 28, 1) 的图像数组


def load_mnist_labels(filepath):
    """
    读取 IDX 格式的标签二进制文件，返回一维 numpy 数组。
    IDX 标签文件头部结构：magic(4B) + 数量(4B) + 标签字节数据。
    参数:
        filepath (str): IDX 标签文件的完整路径，例如 'train-labels-idx1-ubyte'。
    返回:
        np.ndarray: 形状为 (N,) 的 uint8 数组，每个元素为 0~9 的数字标签。
    """
    with open(filepath, 'rb') as f:                              # 以二进制只读模式打开文件
        magic = struct.unpack('>I', f.read(4))[0]                # 读取前 4 字节并解码为魔数，用于校验文件类型
        num_labels = struct.unpack('>I', f.read(4))[0]           # 读取接下来 4 字节，解码为标签总数
        labels = np.frombuffer(f.read(), dtype=np.uint8)         # 将文件剩余全部字节一次性读入，按 uint8 解析为一维标签数组
    return labels                                                # 返回形状为 (N,) 的标签数组


# ======================== 2. 加载数据集 ========================

data_dir = r'/database/mnist'  # 指定本地 MNIST 数据集所在目录（原始字符串避免反斜杠转义）

# 检查数据目录是否存在
if not os.path.isdir(data_dir):                                 # 判断 data_dir 指向的目录是否存在
    raise FileNotFoundError(f"数据目录不存在: {data_dir}")        # 若目录不存在则抛出异常，提示用户检查路径

# 分别加载训练集和测试集的图像与标签
x_train = load_mnist_images(                                    # 加载训练集图像
    os.path.join(data_dir, 'train-images-idx3-ubyte')            # 拼接出训练集图像文件的完整路径
)
y_train = load_mnist_labels(                                    # 加载训练集标签
    os.path.join(data_dir, 'train-labels-idx1-ubyte')            # 拼接出训练集标签文件的完整路径
)
x_test = load_mnist_images(                                     # 加载测试集图像
    os.path.join(data_dir, 't10k-images-idx3-ubyte')             # 拼接出测试集图像文件的完整路径
)
y_test = load_mnist_labels(                                     # 加载测试集标签
    os.path.join(data_dir, 't10k-labels-idx1-ubyte')             # 拼接出测试集标签文件的完整路径
)

# 打印加载结果
print("=" * 60)                                                 # 打印 60 个等号作为分隔线
print("【数据加载完成】")                                         # 提示信息
print(f"  训练集图像: {x_train.shape}   标签: {y_train.shape}")   # 训练集形状：(60000, 28, 28, 1)，标签：(60000,)
print(f"  测试集图像: {x_test.shape}    标签: {y_test.shape}")    # 测试集形状：(10000, 28, 28, 1)，标签：(10000,)


# ======================== 3. 数据预处理 ========================

# 归一化：将像素值从整数 [0, 255] 映射到浮点数 [0.0, 1.0]
x_train = x_train.astype(np.float32) / 255.0                    # 训练集像素转为 float32 并除以 255 归一化，float32 兼顾精度和显存
x_test = x_test.astype(np.float32) / 255.0                      # 测试集同样归一化，保持与训练集相同的预处理方式

# 获取类别数
num_classes = 10                                                # MNIST 数据集包含 10 个类别（手写数字 0-9）

# 将整数标签转为 one-hot 编码（供 categorical_crossentropy 损失函数使用）
y_train_cat = keras.utils.to_categorical(                       # 将训练集标签转换为 one-hot 向量
    y_train,                                                     # 参数 1：整数标签数组，形状 (60000,)
    num_classes=num_classes                                      # 参数 2：类别总数，生成的 one-hot 向量长度为 10
)
y_test_cat = keras.utils.to_categorical(                        # 将测试集标签转换为 one-hot 向量
    y_test,                                                      # 参数 1：整数标签数组，形状 (10000,)
    num_classes=num_classes                                      # 参数 2：类别总数 10
)

# 打印预处理结果
print("=" * 60)                                                 # 分隔线
print("【数据预处理完成】")                                       # 提示信息
print(f"  像素值范围: [{x_train.min():.1f}, {x_train.max():.1f}]")  # 应输出 [0.0, 1.0]
print(f"  One-hot 标签形状: {y_train_cat.shape}")                # 应输出 (60000, 10)


# ======================== 4. 使用 Functional API 搭建 CNN 模型 ========================

print("=" * 60)                                                 # 分隔线
print("【搭建 CNN 模型】")                                       # 提示信息

# ---- 输入层 ----
inputs = keras.Input(                                           # 定义模型的输入张量（KerasTensor）
    shape=(28, 28, 1),                                           # 参数 shape：单个样本的空间维度 (高度, 宽度, 通道数)，不包含 batch 维度
    name='input'                                                 # 参数 name：该层的名称，便于在模型摘要中识别
)                                                                # 结果 inputs 的完整形状为 (None, 28, 28, 1)，None 为 batch 维度

# ---- 第一个卷积块 ----
x = layers.Conv2D(                                              # 二维卷积层，提取图像局部空间特征
    filters=32,                                                  # 参数 filters：卷积核/滤波器数量为 32，输出 32 个特征图
    kernel_size=(3, 3),                                          # 参数 kernel_size：卷积核尺寸 3×3
    activation='relu',                                           # 参数 activation：激活函数 ReLU，引入非线性并缓解梯度消失
    name='conv1'                                                 # 参数 name：层名称
)(inputs)                                                        # 将 inputs 张量传入该层，得到输出张量形状 (None, 26, 26, 32)

x = layers.MaxPooling2D(                                        # 二维最大池化层，降低空间维度并提取显著特征
    pool_size=(2, 2),                                            # 参数 pool_size：池化窗口 2×2，每个维度缩小为原来的一半
    name='pool1'                                                 # 参数 name：层名称
)(x)                                                             # 输入 (None, 26, 26, 32) → 输出 (None, 13, 13, 32)

# ---- 第二个卷积块 ----
x = layers.Conv2D(                                              # 第二个卷积层，提取更高层语义特征
    filters=64,                                                  # 参数 filters：卷积核数量 64，比第一层更多以捕获更丰富特征
    kernel_size=(3, 3),                                          # 参数 kernel_size：卷积核尺寸 3×3
    activation='relu',                                           # 参数 activation：ReLU 激活函数
    name='conv2'                                                 # 参数 name：层名称
)(x)                                                             # 输入 (None, 13, 13, 32) → 输出 (None, 11, 11, 64)

x = layers.MaxPooling2D(                                        # 第二个最大池化层
    pool_size=(2, 2),                                            # 参数 pool_size：池化窗口 2×2
    name='pool2'                                                 # 参数 name：层名称
)(x)                                                             # 输入 (None, 11, 11, 64) → 输出 (None, 5, 5, 64)

# ---- 展平层 ----
x = layers.Flatten(                                             # 展平层，将多维特征图拉平为一维向量以便输入全连接层
    name='flatten'                                               # 参数 name：层名称
)(x)                                                             # 输入 (None, 5, 5, 64) → 输出 (None, 1600)，即 5×5×64=1600

# ---- 全连接层 ----
x = layers.Dense(                                               # 全连接（密集）层
    units=128,                                                   # 参数 units：神经元数量 128
    activation='relu',                                           # 参数 activation：ReLU 激活函数
    name='fc1'                                                   # 参数 name：层名称
)(x)                                                             # 输入 (None, 1600) → 输出 (None, 128)

# ---- Dropout 正则化层 ----
x = layers.Dropout(                                             # Dropout 层，训练时随机将部分神经元输出置零以防止过拟合
    rate=0.5,                                                    # 参数 rate：丢弃概率 0.5，即每次训练约一半神经元被随机屏蔽
    name='dropout'                                               # 参数 name：层名称
)(x)                                                             # 输入输出形状不变 (None, 128)，推理时不做丢弃

# ---- 输出层 ----
outputs = layers.Dense(                                         # 输出全连接层，将特征映射为类别概率
    units=10,                                                    # 参数 units：10 个神经元对应 10 个数字类别
    activation='softmax',                                        # 参数 activation：Softmax 将输出归一化为概率分布（所有输出之和为 1）
    name='output'                                                # 参数 name：层名称
)(x)                                                             # 输入 (None, 128) → 输出 (None, 10)

# ---- 构建完整模型 ----
model = keras.Model(                                            # 通过 Functional API 将输入和输出连接为完整模型
    inputs=inputs,                                               # 参数 inputs：模型的输入张量
    outputs=outputs,                                             # 参数 outputs：模型的输出张量
    name='MNIST_CNN'                                             # 参数 name：模型名称
)

# 打印模型摘要（包含每层输出形状和参数量）
print("\n模型结构摘要:")                                         # 提示信息
model.summary()                                                 # 调用 Keras 内置方法打印表格形式的模型概览

# ======================== 5. 逐层打印输入/输出维度 ========================

print("=" * 60)                                                 # 分隔线
print("【各层 输入/输出 维度详情】")                               # 提示信息
print("-" * 60)                                                 # 虚线分隔

for i, layer in enumerate(model.layers):                       # 遍历模型中的每一层，i 为层索引，layer 为层对象
    layer_name = layer.name                                    # 获取当前层的名称字符串

    # ---- 获取输出维度 ----
    # Keras 3 中 layer.output 是该层在计算图中的输出 KerasTensor，
    # 通过 .shape 属性可获取其形状（含 batch 维度）
    out_shape = layer.output.shape                             # 例如 Conv2D 层输出形状为 (None, 26, 26, 32)

    # ---- 获取输入维度 ----
    # 当前层的输入维度 = 前一层的输出维度（单链模型）
    # InputLayer 是计算图的源头，不存在"前一层"，故标记为 N/A
    if i == 0:                                                 # 若当前层是第一层（即 InputLayer）
        in_shape_str = 'N/A'                                   # 输入层无上游输入，显示 N/A
    else:                                                      # 对于非输入层
        in_shape = model.layers[i - 1].output.shape            # 获取前一层输出形状，即当前层的输入形状
        in_shape_str = str(in_shape)                           # 将 TensorShape 转为字符串以便格式化打印

    print(                                                     # 格式化输出当前层的输入和输出维度
        f"  {layer_name:<15s} "                                # 左对齐层名称，占 15 字符宽度
        f"| 输入维度: {in_shape_str:<28s} "                    # 左对齐输入维度字符串，占 28 字符宽度
        f"| 输出维度: {out_shape}"                             # 输出维度
    )


# ======================== 6. 编译模型 ========================

print("=" * 60)                                                 # 分隔线
print("【编译模型】")                                            # 提示信息

model.compile(                                                  # 编译模型，配置优化器、损失函数和评估指标
    optimizer='adam',                                            # 参数 optimizer：使用 Adam 优化器（自适应学习率，默认 lr=0.001）
    loss='categorical_crossentropy',                             # 参数 loss：多分类交叉熵损失，适用于 one-hot 编码标签
    metrics=['accuracy']                                         # 参数 metrics：训练和评估时同时计算准确率
)

print("  编译完成。")                                            # 提示编译结束


# ======================== 7. 训练模型 ========================

print("=" * 60)                                                 # 分隔线
print("【开始训练】")                                            # 提示信息

history = model.fit(                                            # 训练模型，返回 History 对象包含训练过程的损失和指标记录
    x=x_train,                                                  # 参数 x：训练输入数据，形状 (60000, 28, 28, 1)
    y=y_train_cat,                                               # 参数 y：训练目标标签（one-hot），形状 (60000, 10)
    epochs=3,                                                   # 参数 epochs：训练总轮次，完整遍历训练集 10 次
    batch_size=128,                                              # 参数 batch_size：每批样本数 128，每次梯度更新使用 128 个样本
    validation_split=0.1,                                        # 参数 validation_split：从训练集末尾取 10% 作为验证集（6000 个样本）
    verbose=1                                                    # 参数 verbose：输出详细进度（每轮打印损失、准确率和进度条）
)


# ======================== 8. 评估模型准确率 ========================

print("\n" + "=" * 60)                                          # 分隔线
print("【模型评估结果】")                                        # 提示信息

# 在训练集上评估
train_loss, train_acc = model.evaluate(                         # 在训练集上评估模型性能
    x=x_train,                                                  # 参数 x：训练集图像数据
    y=y_train_cat,                                               # 参数 y：训练集 one-hot 标签
    verbose=0                                                    # 参数 verbose：静默模式，不打印进度条
)
print(f"  训练集  →  损失: {train_loss:.4f}   准确率: {train_acc:.4f}  ({train_acc * 100:.2f}%)")  # 格式化输出训练集评估结果

# 在测试集上评估
test_loss, test_acc = model.evaluate(                           # 在测试集上评估模型性能
    x=x_test,                                                   # 参数 x：测试集图像数据
    y=y_test_cat,                                                # 参数 y：测试集 one-hot 标签
    verbose=0                                                    # 参数 verbose：静默模式
)
print(f"  测试集  →  损失: {test_loss:.4f}   准确率: {test_acc:.4f}  ({test_acc * 100:.2f}%)")   # 格式化输出测试集评估结果


# ======================== 9. 保存模型 ========================

model_save_path = 'mnist_cnn_model.keras'                      # 定义模型保存路径，使用 Keras 3 原生 .keras 格式

model.save(model_save_path)                                     # 将模型架构、权重和训练配置保存为单一 .keras 文件

print("=" * 60)                                                 # 分隔线
print(f"【模型已保存至】: {os.path.abspath(model_save_path)}")   # 打印模型的绝对路径
print("【训练流程全部完成】")                                     # 最终提示