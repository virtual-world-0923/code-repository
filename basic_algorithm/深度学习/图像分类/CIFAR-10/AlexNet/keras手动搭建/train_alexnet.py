# ============================================================================
# 文件名：train_alexnet.py
# 功能：使用Keras搭建AlexNet网络，在本地CIFAR-10数据集上训练并测试模型
# 环境：Python 3.13.14 / Keras 3.15.0 / TensorFlow 2.21.0 / NumPy 2.5.1
# ============================================================================

# 导入pickle模块，用于反序列化CIFAR-10的本地数据文件（.pkl格式的二进制文件）
import pickle
# 导入numpy库，用于数组操作、数据预处理（归一化、reshape等）
import numpy as np
# 导入os模块，用于拼接文件路径（跨平台兼容）
import os
# 导入keras库（Keras 3.x独立包，不再使用tf.keras）
import keras
# 从keras.layers中导入构建网络所需的各层组件
# Conv2D：二维卷积层，用于提取图像空间特征
# MaxPooling2D：二维最大池化层，用于下采样降低特征图尺寸
# Dense：全连接层，用于最终的分类决策
# Flatten：展平层，将多维特征图展平为一维向量以接入全连接层
# Dropout：随机失活层，训练时随机丢弃神经元防止过拟合
# BatchNormalization：批归一化层，加速训练收敛并提升稳定性
# ZeroPadding2D：零填充层，在特征图边缘补零以控制输出尺寸
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    ZeroPadding2D,
    Input,
)
# 从keras.models中导入Model类，用于构建函数式API模型
from keras.models import Model
# 从keras.optimizers中导入Adam优化器，用于模型参数更新
from keras.optimizers import Adam
# 从keras.utils中导入to_categorical函数，将整数标签转为one-hot编码
from keras.utils import to_categorical

# ============================================================================
# 第一部分：定义数据加载函数
# ============================================================================

def load_cifar10_batch(file_path):
    """
    功能：读取单个CIFAR-10 batch文件（pickle格式），返回图像数据和标签
    参数：
        file_path (str): batch文件的完整路径，如 'D:\\...\\data_batch_1'
    返回：
        data (numpy.ndarray): 形状为(10000, 3072)的uint8数组，每行是一张图片的像素值
        labels (list): 长度为10000的列表，每个元素是0-9的整数标签
    """
    # 以二进制只读模式打开文件，CIFAR-10的batch文件是二进制pickle格式
    with open(file_path, 'rb') as f:
        # 使用pickle.load反序列化文件内容，encoding='bytes'兼容Python2/3生成的文件
        # 返回一个字典，包含 'data'、'labels'、'batch_label'、'filenames' 等键
        batch = pickle.load(f, encoding='bytes')
    # 从字典中取出图像数据，键名为b'data'（bytes类型键名）
    # data的形状为(10000, 3072)，3072 = 32*32*3，按R、G、B通道顺序排列
    data = batch[b'data']
    # 从字典中取出标签列表，键名为b'labels'
    # labels是长度为10000的列表，每个元素为0-9的整数
    labels = batch[b'labels']
    # 返回图像数据和标签
    return data, labels


def load_cifar10_data(data_dir):
    """
    功能：从本地目录加载CIFAR-10训练集和测试集
    参数：
        data_dir (str): CIFAR-10数据目录路径，包含data_batch_1和test_batch文件
    返回：
        x_train (numpy.ndarray): 训练集图像，形状(10000, 32, 32, 3)，float32类型
        y_train (numpy.ndarray): 训练集标签，形状(10000, 10)，one-hot编码
        x_test (numpy.ndarray): 测试集图像，形状(10000, 32, 32, 3)，float32类型
        y_test (numpy.ndarray): 测试集标签，形状(10000, 10)，one-hot编码
    """
    # 拼接训练集文件路径：data_dir + 'data_batch_1'
    train_file = os.path.join(data_dir, 'data_batch_1')
    # 拼接测试集文件路径：data_dir + 'test_batch'
    test_file = os.path.join(data_dir, 'test_batch')

    # 调用load_cifar10_batch读取训练集，返回原始数据和标签
    train_data, train_labels = load_cifar10_batch(train_file)
    # 调用load_cifar10_batch读取测试集，返回原始数据和标签
    test_data, test_labels = load_cifar10_batch(test_file)

    # 将训练集数据reshape为(10000, 3, 32, 32)，即(样本数, 通道数, 高, 宽)
    # CIFAR-10原始存储格式是(10000, 3072)，前1024个是R通道，中间1024是G，后1024是B
    train_data = train_data.reshape(10000, 3, 32, 32)
    # 将测试集数据同样reshape为(10000, 3, 32, 32)
    test_data = test_data.reshape(10000, 3, 32, 32)

    # 将通道维度从第1维移动到最后一维，变为(10000, 32, 32, 3)
    # Keras默认使用channels_last格式（即NHWC：样本数, 高, 宽, 通道数）
    # np.transpose的axes参数(0,2,3,1)表示：第0维不变，原第2维→新第1维，原第3维→新第2维，原第1维→新第3维
    train_data = np.transpose(train_data, (0, 2, 3, 1))
    # 测试集同样转置为channels_last格式
    test_data = np.transpose(test_data, (0, 2, 3, 1))

    # 将像素值从uint8(0-255)转换为float32，并归一化到[0, 1]区间
    # 除以255.0使得神经网络输入在合理范围内，加速收敛
    x_train = train_data.astype('float32') / 255.0
    # 测试集同样归一化
    x_test = test_data.astype('float32') / 255.0

    # 将训练集整数标签转换为one-hot编码，num_classes=10表示10个类别
    # 例如标签3变为[0,0,0,1,0,0,0,0,0,0]
    y_train = to_categorical(train_labels, num_classes=10)
    # 测试集标签同样转为one-hot编码
    y_test = to_categorical(test_labels, num_classes=10)

    # 返回预处理后的训练集和测试集
    return x_train, y_train, x_test, y_test


# ============================================================================
# 第二部分：搭建AlexNet网络架构（适配32x32输入）
# ============================================================================

def build_alexnet(input_shape=(32, 32, 3), num_classes=10):
    """
    功能：构建适配CIFAR-10(32x32)的AlexNet网络
    参数：
        input_shape (tuple): 输入图像形状，默认(32, 32, 3)表示32x32的RGB图像
        num_classes (int): 分类类别数，CIFAR-10为10类
    返回：
        model (keras.Model): 编译前的Keras模型对象
    说明：
        原始AlexNet针对227x227输入设计，此处针对32x32做了适配调整：
        - 减小卷积核大小和步长，避免特征图过早缩小到0
        - 减少池化层数量
        - 保留AlexNet的核心思想：卷积→池化→卷积→池化→全连接→全连接→输出
    """
    # 定义输入层，shape参数指定单张图像的形状(高, 宽, 通道数)
    # Input层是函数式API的起点，返回一个张量对象
    inputs = Input(shape=input_shape)

    # ---------- 第1个卷积块 ----------
    # 第一层卷积：64个11x11卷积核（原始AlexNet为11x11，此处缩小为5x5适配小图）
    # 这里使用5x5卷积核，步长1，padding='same'保持空间尺寸不变
    # activation='relu'：ReLU激活函数，引入非线性，公式f(x)=max(0,x)
    x = Conv2D(
        filters=64,              # 卷积核数量（输出通道数）
        kernel_size=(5, 5),      # 卷积核大小，适配32x32输入
        strides=(1, 1),          # 步长，每次滑动1个像素
        padding='same',          # 填充方式，'same'表示输出与输入空间尺寸相同
        activation='relu',       # 激活函数，ReLU
        name='conv1'             # 层名称，便于调试和可视化
    )(inputs)                    # 将输入张量传入该层

    # 批归一化：对每个mini-batch的特征进行归一化，加速收敛
    # axis=-1表示沿最后一个轴（通道轴）进行归一化
    x = BatchNormalization(axis=-1, name='bn1')(x)

    # 最大池化：2x2窗口，步长2，将特征图尺寸减半（32x32→16x16）
    # pool_size=(2,2)：池化窗口大小
    # strides=(2,2)：池化步长
    x = MaxPooling2D(
        pool_size=(2, 2),        # 池化窗口大小
        strides=(2, 2),          # 池化步长
        padding='valid',         # 不填充，直接池化
        name='pool1'             # 层名称
    )(x)

    # ---------- 第2个卷积块 ----------
    # 第二层卷积：192个5x5卷积核（原始AlexNet为5x5，此处缩小为3x3）
    # 使用3x3卷积核，步长1，padding='same'保持尺寸
    x = Conv2D(
        filters=192,             # 卷积核数量
        kernel_size=(3, 3),      # 卷积核大小
        strides=(1, 1),          # 步长
        padding='same',          # 填充方式
        activation='relu',       # ReLU激活
        name='conv2'             # 层名称
    )(x)

    # 批归一化
    x = BatchNormalization(axis=-1, name='bn2')(x)

    # 最大池化：将16x16→8x8
    x = MaxPooling2D(
        pool_size=(2, 2),        # 池化窗口大小
        strides=(2, 2),          # 池化步长
        padding='valid',         # 不填充
        name='pool2'             # 层名称
    )(x)

    # ---------- 第3个卷积块 ----------
    # 第三层卷积：384个3x3卷积核
    x = Conv2D(
        filters=384,             # 卷积核数量
        kernel_size=(3, 3),      # 卷积核大小
        strides=(1, 1),          # 步长
        padding='same',          # 填充方式
        activation='relu',       # ReLU激活
        name='conv3'             # 层名称
    )(x)

    # 批归一化
    x = BatchNormalization(axis=-1, name='bn3')(x)

    # ---------- 第4个卷积块 ----------
    # 第四层卷积：256个3x3卷积核
    x = Conv2D(
        filters=256,             # 卷积核数量
        kernel_size=(3, 3),      # 卷积核大小
        strides=(1, 1),          # 步长
        padding='same',          # 填充方式
        activation='relu',       # ReLU激活
        name='conv4'             # 层名称
    )(x)

    # 批归一化
    x = BatchNormalization(axis=-1, name='bn4')(x)

    # ---------- 第5个卷积块 ----------
    # 第五层卷积：256个3x3卷积核
    x = Conv2D(
        filters=256,             # 卷积核数量
        kernel_size=(3, 3),      # 卷积核大小
        strides=(1, 1),          # 步长
        padding='same',          # 填充方式
        activation='relu',       # ReLU激活
        name='conv5'             # 层名称
    )(x)

    # 批归一化
    x = BatchNormalization(axis=-1, name='bn5')(x)

    # 最大池化：将8x8→4x4
    x = MaxPooling2D(
        pool_size=(2, 2),        # 池化窗口大小
        strides=(2, 2),          # 池化步长
        padding='valid',         # 不填充
        name='pool3'             # 层名称
    )(x)

    # ---------- 全连接层部分 ----------
    # 展平层：将4x4x256的三维特征图展平为一维向量（长度=4*4*256=4096）
    x = Flatten(name='flatten')(x)

    # 第一个全连接层：4096个神经元，ReLU激活
    # 原始AlexNet全连接层有4096个神经元
    x = Dense(
        units=4096,              # 输出神经元数量
        activation='relu',       # ReLU激活函数
        name='fc1'               # 层名称
    )(x)

    # Dropout层：训练时随机丢弃50%的神经元，防止过拟合
    # rate=0.5表示每次训练随机将50%的神经元输出置零
    x = Dropout(rate=0.5, name='dropout1')(x)

    # 第二个全连接层：4096个神经元，ReLU激活
    x = Dense(
        units=4096,              # 输出神经元数量
        activation='relu',       # ReLU激活函数
        name='fc2'               # 层名称
    )(x)

    # Dropout层：同样50%丢弃率
    x = Dropout(rate=0.5, name='dropout2')(x)

    # 输出层：10个神经元对应10个类别，softmax激活输出概率分布
    # softmax将输出转换为概率分布，所有输出之和为1
    outputs = Dense(
        units=num_classes,       # 输出神经元数量=类别数=10
        activation='softmax',    # softmax激活，输出各类别概率
        name='output'            # 层名称
    )(x)

    # 使用函数式API创建模型，指定输入和输出张量
    model = Model(inputs=inputs, outputs=outputs, name='AlexNet_CIFAR10')

    # 返回构建好的模型（尚未编译）
    return model


# ============================================================================
# 第三部分：主程序——训练、评估、保存模型
# ============================================================================

if __name__ == '__main__':
    # 定义本地CIFAR-10数据目录路径
    # 该目录下包含：batches.meta、data_batch_1、test_batch 三个文件
    DATA_DIR = r'/database/cifar-10-batches-py'

    # 定义模型保存路径，使用Keras 3默认的.keras格式
    MODEL_SAVE_PATH = 'alexnet_cifar10.keras'

    # 定义训练轮数（epoch），即整个训练集被完整遍历的次数
    EPOCHS = 3
    # 定义批次大小，每次训练使用的样本数量
    BATCH_SIZE = 128
    # 定义学习率，控制参数更新的步长大小
    LEARNING_RATE = 0.001

    # ---------- 步骤1：加载数据 ----------
    # 打印提示信息，表示开始加载数据
    print('=' * 60)
    print('步骤1：加载CIFAR-10本地数据...')
    print('=' * 60)
    # 调用数据加载函数，获取预处理后的训练集和测试集
    x_train, y_train, x_test, y_test = load_cifar10_data(DATA_DIR)
    # 打印训练集图像的形状，预期为(10000, 32, 32, 3)
    print(f'训练集图像形状: {x_train.shape}')
    # 打印训练集标签的形状，预期为(10000, 10)
    print(f'训练集标签形状: {y_train.shape}')
    # 打印测试集图像的形状，预期为(10000, 32, 32, 3)
    print(f'测试集图像形状: {x_test.shape}')
    # 打印测试集标签的形状，预期为(10000, 10)
    print(f'测试集标签形状: {y_test.shape}')

    # ---------- 步骤2：搭建模型 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print('步骤2：搭建AlexNet网络...')
    print('=' * 60)
    # 调用build_alexnet函数构建模型，输入形状为(32, 32, 3)，分类数为10
    model = build_alexnet(input_shape=(32, 32, 3), num_classes=10)

    # ---------- 步骤3：打印模型结构 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print('步骤3：模型结构如下：')
    print('=' * 60)
    # model.summary()打印每一层的名称、输出形状、参数量等信息
    # 输出包括：层名、输出维度、可训练参数数量、总参数量统计
    model.summary()

    # ---------- 步骤4：编译模型 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print('步骤4：编译模型...')
    print('=' * 60)
    # 创建Adam优化器实例
    # learning_rate：学习率，控制每次参数更新的幅度
    optimizer = Adam(learning_rate=LEARNING_RATE)
    # 编译模型，配置训练所需的优化器、损失函数和评估指标
    # optimizer：优化算法，Adam结合了动量和自适应学习率的优点
    # loss='categorical_crossentropy'：多分类交叉熵损失函数，适用于one-hot标签
    # metrics=['accuracy']：训练和评估时监控的指标，这里监控准确率
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # 打印编译完成提示
    print('模型编译完成！')

    # ---------- 步骤5：训练模型 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print(f'步骤5：开始训练（epochs={EPOCHS}, batch_size={BATCH_SIZE}）...')
    print('=' * 60)
    # 调用model.fit进行模型训练
    # x_train：训练集图像数据
    # y_train：训练集标签（one-hot编码）
    # batch_size：每个mini-batch的样本数
    # epochs：训练轮数
    # validation_data：验证集，每个epoch结束后在验证集上评估
    # verbose=1：训练过程中打印进度条和指标信息
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # ---------- 步骤6：评估模型 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print('步骤6：评估模型准确率...')
    print('=' * 60)
    # 在训练集上评估模型，返回[loss, accuracy]
    # verbose=0表示不打印进度条
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    # 打印训练集准确率，格式化为百分比
    print(f'训练集准确率: {train_accuracy * 100:.2f}%')
    # 打印训练集损失值
    print(f'训练集损失值: {train_loss:.4f}')

    # 在测试集上评估模型，返回[loss, accuracy]
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    # 打印测试集准确率
    print(f'测试集准确率: {test_accuracy * 100:.2f}%')
    # 打印测试集损失值
    print(f'测试集损失值: {test_loss:.4f}')

    # ---------- 步骤7：保存模型 ----------
    # 打印提示信息
    print('\n' + '=' * 60)
    print('步骤7：保存模型...')
    print('=' * 60)
    # Keras 3中model.save()默认保存为.keras格式（包含模型结构+权重+优化器状态）
    # 参数为保存的文件路径字符串
    model.save(MODEL_SAVE_PATH)
    # 打印保存成功提示及文件路径
    print(f'模型已保存至: {MODEL_SAVE_PATH}')
    print('训练流程全部完成！')