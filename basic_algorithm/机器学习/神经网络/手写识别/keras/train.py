# 导入tensorflow主模块，提供keras网络、张量运算、模型保存加载功能
import tensorflow as tf
# os模块：拼接Windows本地文件路径，判断文件是否存在
import os
# struct模块：解析MNIST二进制idx文件的头部数字（魔数、样本数、尺寸）
import struct
# numpy模块：二进制字节转像素数组、数值归一化、矩阵展平
import numpy as np

# ===================== 全局配置参数 =====================
# MNIST原生idx文件根目录，按你提供路径修改
DATASET_ROOT = r"D:\Repositories\basic_algorithm\database\mnist"
# 训练集图片二进制文件名
train_images_path = 'train-images-idx3-ubyte'
# 训练集标签二进制文件名
train_labels_path = 'train-labels-idx1-ubyte'
# 测试集图片二进制文件名
test_images_path = 't10k-images-idx3-ubyte'
# 测试集标签二进制文件名
test_labels_path = 't10k-labels-idx1-ubyte'
# 训练完成后模型保存路径
MODEL_SAVE_PATH = r"./mnist_mlp_model.keras"
# 单次训练批次大小，每次传入64张图片计算梯度
BATCH_SIZE = 64
# 完整遍历全部数据集的总轮数
EPOCHS = 5
# MNIST单张图片像素尺寸 28×28
IMG_SIZE = 28
# 手写数字分类总数：0~9共10类
CLASS_NUM = 10

# ===================== 新增：MNIST idx二进制文件读取函数 =====================
def read_mnist_images(file_full_path):
    """
    读取MNIST图片idx3-ubyte二进制文件，返回归一化、展平后的图片数组
    :param file_full_path: idx图片文件完整绝对路径字符串
    :return: np.float32数组，形状[样本总数, 784]，像素0~1
    """
    # 以二进制只读模式打开图片文件
    with open(file_full_path, 'rb') as f:
        # 读取前4个int头部信息：魔数、样本数量、图片高度、图片宽度
        # struct.unpack读取大端序4个无符号int，格式符>IIII代表大端4个int
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # 计算单张图片总像素：28*28=784
        image_pixel_count = rows * cols
        # 读取剩余全部字节，全部图片原始uint8像素数据
        image_buffer = f.read()
        # 将二进制字节转为uint8一维numpy数组
        images_raw = np.frombuffer(image_buffer, dtype=np.uint8)
        # 重塑数组维度：[样本数, 28, 28]
        images_2d = images_raw.reshape(num_images, rows, cols)
        # 展平二维图片为一维向量：[样本数, 784]，适配全连接网络输入
        images_flat = images_2d.reshape(num_images, image_pixel_count)
        # 转换数据类型为float32，神经网络浮点计算要求
        images_float = images_flat.astype(np.float32)
        # 像素归一化：0~255像素映射至0~1区间，加速模型收敛
        images_norm = images_float / 255.0
    # 返回处理完成的图片数据集
    return images_norm

def read_mnist_labels(file_full_path):
    """
    读取MNIST标签idx1-ubyte二进制文件，返回独热编码标签数组
    :param file_full_path: idx标签文件完整绝对路径字符串
    :return: np.float32独热标签数组，形状[样本总数, 10]
    """
    # 二进制只读打开标签文件
    with open(file_full_path, 'rb') as f:
        # 读取头部8字节：魔数、标签样本数量，格式>II为大端2个int
        magic, num_labels = struct.unpack('>II', f.read(8))
        # 读取全部标签字节数据
        label_buffer = f.read()
        # 字节转为uint8一维数组，每个元素是0~9原始数字标签
        labels_raw = np.frombuffer(label_buffer, dtype=np.uint8)
    # 使用tf.one_hot做独热编码，深度=10代表10个分类
    labels_onehot = tf.one_hot(labels_raw, depth=CLASS_NUM)
    # 转换为numpy数组返回
    return labels_onehot.numpy()

def create_dataset(images_np, labels_np, batch_size, shuffle_flag=True):
    """
    将numpy格式图片+标签转为tf.data训练数据集管道
    :param images_np: 归一化展平后的图片numpy数组 [N,784]
    :param labels_np: 独热编码标签numpy数组 [N,10]
    :param batch_size: 批次大小int
    :param shuffle_flag: True=打乱(训练集),False=不打乱(测试集)
    :return: 分批次、预加载优化后的tf.data.Dataset
    """
    # 从numpy数组创建tf数据集，每条样本(图片,标签)配对
    raw_ds = tf.data.Dataset.from_tensor_slices((images_np, labels_np))
    # 判断是否开启样本打乱，训练集打乱避免顺序过拟合
    if shuffle_flag:
        # 缓冲区10000随机打乱样本，AUTOTUNE自动分配CPU线程
        raw_ds = raw_ds.shuffle(buffer_size=10000)
    # 按指定批次打包样本，不足一批保留剩余数据
    batch_ds = raw_ds.batch(batch_size)
    # 预加载下一批数据，GPU训练时CPU同步读取，提升运行速度
    optimize_ds = batch_ds.prefetch(tf.data.AUTOTUNE)
    return optimize_ds

# ===================== 搭建传统全连接神经网络(MLP) =====================
def build_mlp_model():
    """
    搭建无卷积的传统多层感知机全连接网络
    :return: 编译完成的Sequential Keras模型
    """
    # Sequential线性堆叠网络层，适合简单全连接网络
    model = tf.keras.Sequential(name="MNIST_FullyConnected_MLP")
    # 显示设定输入层，不做任何操作，占位，接收网络输入维度
    model.add(tf.keras.layers.InputLayer(shape=(IMG_SIZE * IMG_SIZE,),name="input_layer"))
    # 展平层，输入接收28x28，展平成一维（已手动在读取数据时展平，如果没有手动展平，需要在这里添加展平层）
    # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 第一层隐藏层：输入维度784，256个神经元，ReLU激活
    model.add(tf.keras.layers.Dense(units=256,activation="relu",name="hidden_layer_1"))
    # Dropout层：随机丢弃30%神经元，抑制过拟合
    model.add(tf.keras.layers.Dropout(rate=0.3, name="dropout_1"))
    # 第二层隐藏层：128神经元，ReLU激活
    model.add(tf.keras.layers.Dense(units=128, activation="relu", name="hidden_layer_2"))
    # 第二层Dropout：丢弃20%神经元
    model.add(tf.keras.layers.Dropout(rate=0.2, name="dropout_2"))
    # 输出层：10神经元，Softmax输出各类别概率
    model.add(tf.keras.layers.Dense(units=CLASS_NUM, activation="softmax", name="output_layer"))

    model.loss
    # 模型编译配置优化器、损失函数、评估指标
    model.compile(
        # Adam自适应学习率优化器，学习率0.001
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # 独热标签专用分类交叉熵损失
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # 监控分类准确率指标
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    # 打印网络结构、每层参数数量
    model.summary()
    for i, layer in enumerate(model._layers):
        print('------------------------------------')
        print(f"{i}: {layer.name} - {layer.__class__.__name__}")
    return model

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 拼接训练集图片完整文件路径
    train_img_full_path = os.path.join(DATASET_ROOT, train_images_path)
    # 拼接训练集标签完整文件路径
    train_label_full_path = os.path.join(DATASET_ROOT, train_labels_path)
    # 拼接测试集图片完整文件路径
    test_img_full_path = os.path.join(DATASET_ROOT, test_images_path)
    # 拼接测试集标签完整文件路径
    test_label_full_path = os.path.join(DATASET_ROOT, test_labels_path)

    print("正在读取训练集二进制idx文件...")
    # 读取并处理训练图片
    train_images = read_mnist_images(train_img_full_path)
    # 读取并处理训练标签（独热编码）
    train_labels = read_mnist_labels(train_label_full_path)

    print("正在读取测试集二进制idx文件...")
    # 读取并处理测试图片
    test_images = read_mnist_images(test_img_full_path)
    # 读取并处理测试标签（独热编码）
    test_labels = read_mnist_labels(test_label_full_path)

    # 构建训练数据集：开启打乱，批次64
    train_ds = create_dataset(train_images, train_labels, BATCH_SIZE, shuffle_flag=True)
    # 构建测试数据集：不打乱，批次64
    test_ds = create_dataset(test_images, test_labels, BATCH_SIZE, shuffle_flag=False)

    # 初始化全连接神经网络模型
    mlp_model = build_mlp_model()

    # 开始训练模型，保存训练过程历史数据
    print("\n===== 开始模型训练 =====")
    train_history = mlp_model.fit(
        x=train_ds, #可以传入NumPy array数组，也可以传入ft.data。如果传入NumPy array，就要设置y标签，并和x一一对应。建议传ft.data，并设置好标签、批次、shuffle等信息，不需要单独传y参数
        epochs=EPOCHS,
        shuffle=False, #当传入x=tf.data.Dataset此参数不起作用，要在ft.data里shuffle。默认是true，所以会弹出一个警告，手动设置为false消除一下警告
        # validation_data=test_ds, #每轮训练结束，在指定数据集上评估。只参与评估，不更新权重
        verbose=1
    )

    # 在完整测试集上评估模型损失与准确率
    print("\n===== 测试集最终评估结果 =====")
    test_loss, test_acc = mlp_model.evaluate(x=test_ds, verbose=1)
    # 获取最后一轮训练集准确率
    train_final_acc = train_history.history["accuracy"][-1]

    # 打印训练、测试集准确率
    print(f"训练集最终准确率: {train_final_acc:.4f}")
    print(f"测试集最终准确率: {test_acc:.4f}")

    # 保存训练完成的模型，供预测脚本加载
    mlp_model.save(MODEL_SAVE_PATH)
    print(f"\n模型已保存至: {MODEL_SAVE_PATH}")