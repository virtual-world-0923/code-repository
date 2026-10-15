# 导入用于处理数值数组的 numpy 库
import numpy as np
# 导入用于解析二进制 MNIST 文件的 struct 库
import struct
# 导入 sklearn 中的多层感知机分类器，用于实现传统神经网络
from sklearn.neural_network import MLPClassifier
# 导入 sklearn 中的准确率评估指标
from sklearn.metrics import accuracy_score
# 导入 joblib 库，用于将训练好的模型保存到本地文件
import joblib

# 定义本地 MNIST 数据集的根目录路径
data_dir = r'D:\Repositories\basic_algorithm\database\mnist'

# 定义一个函数，用于读取 MNIST 图像的二进制文件
def load_mnist_images(filename):
    # 以二进制只读模式 ('rb') 打开指定的图像文件
    with open(filename, 'rb') as f:
        # 读取文件的前 16 个字节，解析出魔术数(4字节)、图片数量(4字节)、行数(4字节)、列数(4字节)
        # '>IIII' 表示大端模式，解析 4 个无符号整型
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # 从文件中读取剩余的字节数据，数据类型为无符号 8 位整数 (uint8)，即像素值 (0-255)
        # 然后使用 reshape 将一维数组重塑为 (图片数量, 行数, 列数) 的三维数组
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        # 返回解析出的图像三维数组
        return images

# 定义一个函数，用于读取 MNIST 标签的二进制文件
def load_mnist_labels(filename):
    # 以二进制只读模式 ('rb') 打开指定的标签文件
    with open(filename, 'rb') as f:
        # 读取文件的前 8 个字节，解析出魔术数(4字节)和标签数量(4字节)
        # '>II' 表示大端模式，解析 2 个无符号整型
        magic, num = struct.unpack(">II", f.read(8))
        # 从文件中读取剩余的字节数据，数据类型为无符号 8 位整数 (uint8)，即标签值 (0-9)
        labels = np.fromfile(f, dtype=np.uint8)
        # 返回解析出的标签一维数组
        return labels

# 调用函数加载训练集图像，传入训练集图像文件的完整路径
train_images = load_mnist_images(f'{data_dir}/train-images-idx3-ubyte')
# 调用函数加载训练集标签，传入训练集标签文件的完整路径
train_labels = load_mnist_labels(f'{data_dir}/train-labels-idx1-ubyte')
# 调用函数加载测试集图像，传入测试集图像文件的完整路径
test_images = load_mnist_images(f'{data_dir}/t10k-images-idx3-ubyte')
# 调用函数加载测试集标签，传入测试集标签文件的完整路径
test_labels = load_mnist_labels(f'{data_dir}/t10k-labels-idx1-ubyte')

# 对训练集图像进行数据预处理：将二维图像 (28, 28) 展平为一维向量 (784,)，因为 MLP 需要一维输入
# reshape(-1) 表示自动计算该维度的长度，保持总元素数量不变
train_images_flat = train_images.reshape(train_images.shape[0], -1)
# 对测试集图像进行同样的展平操作
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# 对训练集图像进行归一化处理：将像素值从 0-255 除以 255.0，缩放到 0-1 之间，有助于神经网络更快收敛
train_images_normalized = train_images_flat / 255.0
# 对测试集图像进行同样的归一化处理
test_images_normalized = test_images_flat / 255.0

# 实例化一个多层感知机分类器 (MLP) 神经网络模型
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 设置隐藏层结构：包含两个隐藏层，第一层 128 个神经元，第二层 64 个神经元
    activation='relu',             # 设置隐藏层的激活函数为 ReLU，引入非线性特征
    solver='adam',                 # 设置权重优化器为 Adam，这是一种自适应学习率的优化算法
    max_iter=10,                   # 设置最大训练迭代次数 (epochs) 为 10，防止训练时间过长
    random_state=42                # 设置随机种子为 42，保证每次运行代码时模型初始化结果一致，方便复现
)

# 开始训练模型：传入归一化后的训练集图像数据和对应的训练集标签
mlp_model.fit(train_images_normalized, train_labels)

# 使用训练好的模型对训练集数据进行预测，获取训练集上的预测结果
train_predictions = mlp_model.predict(train_images_normalized)
# 使用准确率评估函数，对比训练集真实标签和预测标签，计算训练集准确率
train_accuracy = accuracy_score(train_labels, train_predictions)
# 在控制台打印输出训练集的准确率，格式化为保留 4 位小数的百分比
print(f"训练集准确率: {train_accuracy * 100:.4f}%")

# 使用训练好的模型对测试集数据进行预测，获取测试集上的预测结果
test_predictions = mlp_model.predict(test_images_normalized)
# 使用准确率评估函数，对比测试集真实标签和预测标签，计算测试集准确率
test_accuracy = accuracy_score(test_labels, test_predictions)
# 在控制台打印输出测试集的准确率，格式化为保留 4 位小数的百分比
print(f"测试集准确率: {test_accuracy * 100:.4f}%")

# 使用 joblib 将训练好的 MLP 模型保存到本地文件中，文件名为 'mnist_mlp_model.pkl'
joblib.dump(mlp_model, 'mnist_mlp_model.pkl')
# 在控制台打印提示，告知用户模型已成功保存
print("模型已成功保存为 'mnist_mlp_model.pkl'")