# 导入 TensorFlow 库，并设置别名为 tf，这是深度学习框架的核心库
import tensorflow as tf
# 导入 os 模块，用于处理本地文件路径和目录操作
import os

# ==================== 1. 数据加载与预处理 ====================

# 定义本地 MNIST 数据集所在的绝对路径
# 参数说明：字符串类型，指向包含 tfrecord 文件和 json 配置文件的本地文件夹路径
DATA_DIR = r"D:\Repositories\basic_algorithm\database\mnist"

# 拼接训练集 TFRecord 文件的完整绝对路径
# 参数说明：第一个参数是目录路径，第二个参数是文件名；os.path.join 会自动处理斜杠方向
TRAIN_TFRECORD_PATH = os.path.join(DATA_DIR, "mnist-train.tfrecord-00000-of-00001")

# 拼接测试集 TFRecord 文件的完整绝对路径
# 参数说明：同上，指向测试集的 tfrecord 文件
TEST_TFRECORD_PATH = os.path.join(DATA_DIR, "mnist-test.tfrecord-00000-of-00001")


# 定义一个解析 TFRecord 单条数据的函数
# 参数说明：serialized_example 是一个标量字符串张量（scalar string tensor），代表 TFRecord 中序列化后的一条数据
def parse_tfrecord_fn(serialized_example):
    # 定义 TFRecord 中每条数据的特征描述字典（Feature Description）
    # 参数说明：字典的键是特征名称，值是 tf.io 提供的解析描述对象
    feature_description = {
        # 描述 'image' 特征：它是一个固定长度的字节字符串，包含 28*28=784 个像素值
        # 参数说明：shape=(784,) 表示一维数组，dtype=tf.string 表示原始字节数据
        "image": tf.io.FixedLenFeature([], tf.string),
        # 描述 'label' 特征：它是一个整数标量，代表 0-9 的数字类别
        # 参数说明：shape=() 表示标量，dtype=tf.int64 表示 64 位整数类型
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    # 调用 tf.io.parse_single_example 解析单条序列化数据
    # 参数说明：serialized_example 是待解析的字节串，features 是上面定义的特征描述字典
    # 返回值：一个包含解析后张量的字典，键名与 feature_description 相同
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)

    # 将解析出的 'image' 字节字符串解码并转换为 float32 类型的张量
    # 参数说明：parsed_example["image"] 是原始字节数据；tf.io.decode_raw 将其按指定类型解码
    # 第二个参数 tf.float32 指定解码后的数据类型，便于后续神经网络计算
    image = tf.io.decode_raw(parsed_example["image"], tf.float32)

    # 将像素值从 [0, 255] 归一化到 [0, 1] 区间，加速神经网络收敛
    # 参数说明：image 是解码后的张量，除以 255.0 执行逐元素的浮点除法
    image = image / 255.0

    # 获取解析出的 'label' 标签值
    # 参数说明：直接从解析后的字典中取出，类型为 int64 标量张量
    label = parsed_example["label"]

    # 返回处理后的图像张量和对应的标签张量
    # 返回值：一个包含两个元素的元组 (image_tensor, label_tensor)
    return image, label


# 定义一个加载 TFRecord 数据集的函数
# 参数说明：file_path 是 tfrecord 文件的字符串路径；batch_size 是每批次的样本数量
def load_dataset(file_path, batch_size):
    # 使用 tf.data.TFRecordDataset 创建数据集对象
    # 参数说明：file_path 是 tfrecord 文件路径；该函数会惰性读取数据，不会一次性载入内存
    dataset = tf.data.TFRecordDataset(file_path)

    # 使用 map 函数对数据集中的每一条记录应用解析函数
    # 参数说明：parse_tfrecord_fn 是解析函数；num_parallel_calls=tf.data.AUTOTUNE
    # 表示自动根据 CPU 核心数开启多线程并行解析，提升数据读取效率
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # 对数据集进行批处理（Batching）
    # 参数说明：batch_size 是整数，指定每个批次包含的样本数；drop_remainder=True
    # 表示如果最后一个批次样本数不足 batch_size，则丢弃该批次，保证所有批次形状一致
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # 开启数据预取（Prefetching），在 GPU 训练当前批次时，CPU 提前准备下一批次数据
    # 参数说明：tf.data.AUTOTUNE 让 TensorFlow 自动决定预取的批次数量，以平衡内存和速度
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 返回构建好的、可直接用于训练/测试的 Dataset 对象
    return dataset


# ==================== 2. 搭建传统全连接神经网络 ====================

# 使用 Keras Sequential API 创建一个线性堆叠的神经网络模型
# 参数说明：传入一个层（Layer）对象的列表，模型会按列表顺序依次连接这些层
model = tf.keras.Sequential([
    # 第一层：输入层 / 隐藏层1（Dense 全连接层）
    # 参数说明：units=128 表示该层有 128 个神经元；activation='relu' 使用 ReLU 激活函数引入非线性
    # input_shape=(784,) 指定输入数据的形状为 784 维向量（即 28x28 展平后的图像）
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784,)),

    # 第二层：隐藏层2（Dense 全连接层）
    # 参数说明：units=64 表示该层有 64 个神经元；activation='relu' 继续使用 ReLU 激活函数
    tf.keras.layers.Dense(units=64, activation='relu'),

    # 第三层：输出层（Dense 全连接层）
    # 参数说明：units=10 表示输出 10 个神经元，对应 0-9 共 10 个数字类别
    # activation='softmax' 将输出转换为概率分布，所有输出值之和为 1
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# 编译模型，配置训练所需的优化器、损失函数和评估指标
# 参数说明：optimizer 指定优化算法；loss 指定损失函数；metrics 指定训练/测试时监控的指标列表
model.compile(
    # 参数说明：'adam' 是 Adam 优化器，自适应调整学习率，适合大多数深度学习任务
    optimizer='adam',

    # 参数说明：'sparse_categorical_crossentropy' 是稀疏分类交叉熵损失函数
    # 适用于标签为整数（0-9）而非 One-Hot 编码的多分类任务
    loss='sparse_categorical_crossentropy',

    # 参数说明：['accuracy'] 表示在训练和测试过程中计算并输出准确率
    metrics=['accuracy']
)

# 打印模型的结构摘要，查看各层名称、输出形状和参数量
# 参数说明：无参数，直接在控制台输出模型的层级结构信息
model.summary()

# ==================== 3. 加载数据、训练与测试 ====================

# 调用 load_dataset 函数加载训练集数据
# 参数说明：TRAIN_TFRECORD_PATH 是训练集 tfrecord 路径；batch_size=32 表示每批 32 张图像
train_dataset = load_dataset(TRAIN_TFRECORD_PATH, batch_size=32)

# 调用 load_dataset 函数加载测试集数据
# 参数说明：TEST_TFRECORD_PATH 是测试集 tfrecord 路径；batch_size=32 表示每批 32 张图像
test_dataset = load_dataset(TEST_TFRECORD_PATH, batch_size=32)

# 开始训练模型
# 参数说明：train_dataset 是训练数据；epochs=10 表示整个训练集遍历 10 次
# validation_data=test_dataset 表示每个 epoch 结束后在测试集上评估模型性能
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# 在测试集上进行最终评估，获取损失值和准确率
# 参数说明：test_dataset 是测试数据；verbose=1 表示在控制台打印评估进度
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

# 使用格式化字符串打印测试集的最终准确率
# 参数说明：test_accuracy 是 0~1 之间的浮点数，乘以 100 转换为百分比显示
print(f"测试集最终准确率: {test_accuracy * 100:.2f}%")

# ==================== 4. 保存训练好的模型 ====================

# 定义模型保存的目录路径
# 参数说明：字符串类型，指定模型保存的文件夹名称
MODEL_SAVE_PATH = "mnist_traditional_nn_model.keras"

# 调用 save 方法将完整的模型（结构、权重、优化器状态）保存到本地文件
# 参数说明：MODEL_SAVE_PATH 是保存路径；.keras 后缀表示使用 Keras 3 的原生格式保存
model.save(MODEL_SAVE_PATH)

# 打印模型保存成功的提示信息
# 参数说明：f-string 格式化输出，显示保存的具体路径
print(f"模型已成功保存至: {MODEL_SAVE_PATH}")