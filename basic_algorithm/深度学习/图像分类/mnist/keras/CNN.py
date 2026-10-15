# ============================================================
# 文件名: train_mnist.py
# 功能描述: 使用Keras搭建CNN网络，训练MNIST手写数字识别模型，
#           输出训练集和测试集准确率，并保存训练好的模型
# 环境要求: Python 3.13.14, Keras 3.15.0, NumPy 2.5.1
# ============================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 导入numpy库，用于数值计算和数组操作，版本2.5.1
import numpy as np
# 导入keras库，Keras 3.15.0是独立的高级神经网络API，不再从tensorflow导入
import keras
# 导入keras中的layers模块，包含所有神经网络层（卷积层、池化层、全连接层等）
from keras import layers
# 导入keras中的models模块，包含Sequential顺序模型和Model函数式模型
from keras import models
# 导入Python标准库struct模块，用于解析MNIST二进制文件中的整数数据
import struct
# 导入Python标准库os模块，用于文件路径拼接和目录操作
import os

# ============================================================
# 第一部分：定义读取MNIST本地数据的函数
# ============================================================

def load_mnist_images(file_path):
    """
    函数功能: 读取MNIST的images二进制文件（idx3-ubyte格式），返回图片数组
    参数说明:
        file_path (str): MNIST图片文件的本地完整路径，
                         例如 "D:\\Repositories\\basic_algorithm\\database\\mnist\\train-images-idx3-ubyte"
    返回值:
        numpy.ndarray: 形状为 (样本数量, 28, 28) 的uint8数组，每个元素是0-255的像素值
    """
    # 以二进制只读模式("rb")打开指定的MNIST图片文件
    with open(file_path, "rb") as f:
        # 读取文件的前4个字节，使用struct.unpack解析为大端序(>)无符号整数(I)，
        # 得到magic number，对于images文件其值应为2051(0x00000803)
        magic_number = struct.unpack(">I", f.read(4))[0]
        # 读取接下来的4个字节，解析为大端序无符号整数，得到图片总数量（训练集60000，测试集10000）
        num_images = struct.unpack(">I", f.read(4))[0]
        # 读取接下来的4个字节，解析为大端序无符号整数，得到每张图片的行数（固定为28）
        num_rows = struct.unpack(">I", f.read(4))[0]
        # 读取接下来的4个字节，解析为大端序无符号整数，得到每张图片的列数（固定为28）
        num_cols = struct.unpack(">I", f.read(4))[0]
        # 使用numpy的fromfile方法从文件对象中读取剩余所有字节数据，
        # dtype=np.uint8表示每个像素用1个无符号8位整数存储（范围0-255），
        # count=-1表示读取文件中所有剩余数据
        image_data = np.fromfile(f, dtype=np.uint8, count=-1)
    # 将一维数组reshape为三维数组(样本数, 行, 列)，即(num_images, 28, 28)
    image_data = image_data.reshape(num_images, num_rows, num_cols)
    # 返回reshape后的图片数组
    return image_data


def load_mnist_labels(file_path):
    """
    函数功能: 读取MNIST的labels二进制文件（idx1-ubyte格式），返回标签数组
    参数说明:
        file_path (str): MNIST标签文件的本地完整路径，
                         例如 "D:\\Repositories\\basic_algorithm\\database\\mnist\\train-labels-idx1-ubyte"
    返回值:
        numpy.ndarray: 形状为 (样本数量,) 的uint8数组，每个元素是0-9的数字标签
    """
    # 以二进制只读模式("rb")打开指定的MNIST标签文件
    with open(file_path, "rb") as f:
        # 读取文件的前4个字节，解析为大端序无符号整数，
        # 得到magic number，对于labels文件其值应为2049(0x00000801)
        magic_number = struct.unpack(">I", f.read(4))[0]
        # 读取接下来的4个字节，解析为大端序无符号整数，得到标签总数量（与图片数量一致）
        num_labels = struct.unpack(">I", f.read(4))[0]
        # 使用numpy的fromfile方法读取剩余所有字节，每个字节代表一个数字标签(0-9)
        label_data = np.fromfile(f, dtype=np.uint8, count=-1)
    # 返回一维标签数组，形状为(num_labels,)
    return label_data


# ============================================================
# 第二部分：加载本地MNIST数据
# ============================================================

# 定义MNIST数据文件所在的本地目录路径（Windows路径使用原始字符串r""避免转义问题）
data_dir = r"/database/mnist"

# 使用os.path.join拼接训练集图片文件的完整路径
train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
# 使用os.path.join拼接训练集标签文件的完整路径
train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
# 使用os.path.join拼接测试集图片文件的完整路径
test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
# 使用os.path.join拼接测试集标签文件的完整路径
test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

# 调用load_mnist_images函数读取训练集图片，返回形状(60000, 28, 28)的数组
train_images = load_mnist_images(train_images_path)
# 调用load_mnist_labels函数读取训练集标签，返回形状(60000,)的数组
train_labels = load_mnist_labels(train_labels_path)
# 调用load_mnist_images函数读取测试集图片，返回形状(10000, 28, 28)的数组
test_images = load_mnist_images(test_images_path)
# 调用load_mnist_labels函数读取测试集标签，返回形状(10000,)的数组
test_labels = load_mnist_labels(test_labels_path)

# 打印训练集图片数组的形状，验证数据加载是否正确，预期输出(60000, 28, 28)
print(f"训练集图片形状: {train_images.shape}")
# 打印训练集标签数组的形状，预期输出(60000,)
print(f"训练集标签形状: {train_labels.shape}")
# 打印测试集图片数组的形状，预期输出(10000, 28, 28)
print(f"测试集图片形状: {test_images.shape}")
# 打印测试集标签数组的形状，预期输出(10000,)
print(f"测试集标签形状: {test_labels.shape}")

# ============================================================
# 第三部分：数据预处理
# ============================================================

# 将训练集图片数据类型从uint8转换为float32，因为神经网络计算需要浮点数
# 同时除以255.0将像素值从[0,255]归一化到[0,1]区间，有助于模型更快收敛
train_images = train_images.astype("float32") / 255.0
# 将测试集图片数据类型从uint8转换为float32，并归一化到[0,1]区间
test_images = test_images.astype("float32") / 255.0

# 使用numpy的expand_dims在最后一个轴（axis=-1）添加一个维度，
# 将形状从(60000, 28, 28)变为(60000, 28, 28, 1)，
# 因为Conv2D层要求输入形状为(batch, height, width, channels)，灰度图通道数为1
train_images = np.expand_dims(train_images, axis=-1)
# 同样为测试集图片添加通道维度，形状从(10000, 28, 28)变为(10000, 28, 28, 1)
test_images = np.expand_dims(test_images, axis=-1)

# 打印归一化并扩展维度后的训练集形状，预期输出(60000, 28, 28, 1)
print(f"预处理后训练集图片形状: {train_images.shape}")
# 打印归一化并扩展维度后的测试集形状，预期输出(10000, 28, 28, 1)
print(f"预处理后测试集图片形状: {test_images.shape}")

# ============================================================
# 第四部分：搭建CNN神经网络架构
# ============================================================

# 创建Sequential顺序模型实例，Sequential模型是层的线性堆叠，适合逐层构建网络
model = models.Sequential(name="mnist_cnn")

# 添加第一个卷积层Conv2D:
#   filters=32: 使用32个卷积核（滤波器），提取32种不同的低级特征（如边缘、角点）
#   kernel_size=(3,3): 卷积核大小为3x3，是常用的选择，能有效捕获局部特征
#   activation="relu": 使用ReLU激活函数，将负值置零，引入非线性，公式f(x)=max(0,x)
#   input_shape=(28,28,1): 指定输入数据的形状（高28,宽28,通道1），仅第一层需要指定
#   padding="valid": 不进行零填充，卷积后尺寸会缩小，输出尺寸=(28-3+1)=26
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    input_shape=(28, 28, 1),
    padding="valid",
    name="conv2d_1"
))

# 添加第一个最大池化层MaxPooling2D:
#   pool_size=(2,2): 池化窗口大小为2x2，将特征图尺寸缩小一半
#   strides=None: 步长默认等于pool_size，即(2,2)，不重叠
#   作用: 降低特征图空间维度，减少计算量，同时保留最重要的特征（最大值）
#   输入(26,26,32) -> 输出(13,13,32)
model.add(layers.MaxPooling2D(
    pool_size=(2, 2),
    name="maxpool_1"
))

# 添加第二个卷积层Conv2D:
#   filters=64: 使用64个卷积核，提取更高级的组合特征
#   kernel_size=(3,3): 卷积核大小3x3
#   activation="relu": ReLU激活函数
#   padding="valid": 不进行零填充
#   输入(13,13,32) -> 卷积后(11,11,64)
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu",
    padding="valid",
    name="conv2d_2"
))

# 添加第二个最大池化层MaxPooling2D:
#   pool_size=(2,2): 池化窗口2x2
#   输入(11,11,64) -> 输出(5,5,64)
model.add(layers.MaxPooling2D(
    pool_size=(2, 2),
    name="maxpool_2"
))

# 添加Flatten展平层:
#   作用: 将多维特征图(5,5,64)展平为一维向量(1600,)，以便输入全连接层
#   不改变数据本身，只改变形状
model.add(layers.Flatten(name="flatten"))

# 添加第一个全连接层Dense:
#   units=128: 输出神经元数量为128，作为隐藏层进行高级特征组合
#   activation="relu": 使用ReLU激活函数引入非线性
#   输入(1600,) -> 输出(128,)
model.add(layers.Dense(
    units=128,
    activation="relu",
    name="dense_1"
))

# 添加Dropout正则化层:
#   rate=0.5: 训练时随机将50%的神经元输出置零，防止过拟合
#   注意: Dropout仅在训练阶段生效，推理/预测阶段自动关闭
model.add(layers.Dropout(
    rate=0.5,
    name="dropout_1"
))

# 添加输出全连接层Dense:
#   units=10: 输出神经元数量为10，对应数字0-9共10个类别
#   activation="softmax": 使用Softmax激活函数，将输出转换为概率分布，
#                          所有输出值之和为1，最大的那个即为预测类别
#   输入(128,) -> 输出(10,)
model.add(layers.Dense(
    units=10,
    activation="softmax",
    name="dense_output"
))

# 打印模型结构摘要，显示每一层的名称、输出形状和参数量
model.summary()

# ============================================================
# 第五部分：编译模型（配置训练参数）
# ============================================================

# 调用model.compile()方法配置模型的学习过程:
#   optimizer: 使用Adam优化器，learning_rate=0.001是默认学习率，
#              Adam结合了动量和自适应学习率的优点，收敛快且稳定
#   loss: 使用稀疏分类交叉熵损失函数SparseCategoricalCrossentropy，
#         "sparse"表示标签是整数形式(如0,1,2...9)而非one-hot编码形式，
#         适用于多分类问题且标签为整数的场景
#   metrics: 使用稀疏分类准确率SparseCategoricalAccuracy作为评估指标，
#            与sparse loss配套使用，计算预测类别与真实类别匹配的比例
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(), # 传入的y不需要强制onehot格式，可以直接传整数标签，也可是onehot
    # loss=keras.losses.categorical_crossentropy, # 传入的y需要onehot格式
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)
# 将整数标签转换为one-hot独热编码格式，适配多分类交叉熵损失函数的输入要求
# 例如标签3会转为[0,0,0,1,0,0,0,0,0,0]
y_train = keras.utils.to_categorical(train_labels, 10)
y_test = keras.utils.to_categorical(test_labels, 10)

# ============================================================
# 第六部分：训练模型
# ============================================================

# 调用model.fit()方法开始训练模型:
#   x=train_images: 训练输入数据，形状(60000, 28, 28, 1)，float32类型
#   y=train_labels: 训练标签数据，形状(60000,)，uint8类型，值为0-9的整数
#   batch_size=128: 每个批次包含128个样本，较大的batch_size训练更稳定但内存占用更大
#   epochs=10: 整个训练集迭代10轮，每轮模型会看到所有60000个训练样本
#   validation_data: 传入测试集作为验证集，每个epoch结束后在验证集上评估，
#                    用于监控模型是否过拟合（验证损失不再下降或开始上升）
#   verbose=1: 训练过程中显示进度条和每个epoch的损失/准确率信息
history = model.fit(
    x=train_images,
    y=train_labels,
    batch_size=128,
    epochs=3,
    validation_data=(test_images, test_labels),
    verbose=1
)

# ============================================================
# 第七部分：评估模型（输出训练集和测试集准确率）
# ============================================================

# 调用model.evaluate()在训练集上评估模型性能:
#   x=train_images: 训练集图片数据
#   y=train_labels: 训练集标签数据
#   batch_size=128: 评估时的批次大小
#   verbose=0: 不显示进度条，静默评估
#   返回值: [loss值, accuracy值] 的列表
train_loss, train_accuracy = model.evaluate(
    x=train_images,
    y=train_labels,
    batch_size=128,
    verbose=0
)

# 调用model.evaluate()在测试集上评估模型性能:
#   x=test_images: 测试集图片数据
#   y=test_labels: 测试集标签数据
#   batch_size=128: 评估时的批次大小
#   verbose=0: 不显示进度条
#   返回值: [loss值, accuracy值] 的列表
test_loss, test_accuracy = model.evaluate(
    x=test_images,
    y=test_labels,
    batch_size=128,
    verbose=0
)

# 打印训练集上的损失值和准确率，格式化保留4位小数
print(f"\n{'='*50}")
print(f"训练集准确率: {train_accuracy:.4f} (损失: {train_loss:.4f})")
# 打印测试集上的损失值和准确率，格式化保留4位小数
print(f"测试集准确率: {test_accuracy:.4f} (损失: {test_loss:.4f})")
print(f"{'='*50}\n")

# ============================================================
# 第八部分：保存训练好的模型
# ============================================================

# 定义模型保存的文件路径，使用.keras格式（Keras 3原生格式，推荐）
# .keras格式是Keras 3的默认保存格式，包含模型架构、权重和编译配置
model_save_path = r".\mnist_cnn_model.keras"

# 调用model.save()方法将完整模型保存到指定路径:
#   保存内容包括: 模型结构(层配置)、所有权重参数、优化器状态、编译配置
#   后续可用keras.saving.load_model()完整恢复模型，无需重新定义架构
model.save(model_save_path)

# 打印模型保存成功的提示信息，显示保存路径
print(f"模型已成功保存至: {model_save_path}")