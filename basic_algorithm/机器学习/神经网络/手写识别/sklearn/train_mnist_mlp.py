# 导入numpy库，用于数值数组运算、数据维度处理
import numpy as np
# 导入os模块，用于拼接本地文件路径，兼容Windows路径格式
import os
# 导入joblib，sklearn推荐用来保存/加载机器学习模型
import joblib
# 从sklearn.neural_network导入多层感知机分类器（传统全连接神经网络）
from sklearn.neural_network import MLPClassifier
# 导入sklearn指标模块，用于计算准确率
from sklearn.metrics import accuracy_score

def load_mnist(local_data_path):
    """
    函数作用：读取本地原始idx-ubyte格式MNIST二进制文件，解析图片与标签
    重点修复：idx文件数据为大端存储，numpy读取后需要字节反转byteswap()
    参数：
        local_data_path: str，mnist四个文件所在文件夹路径
    返回：
        X_train: 训练图片数组 (样本数,784)
        y_train: 训练标签数组
        X_test: 测试图片数组 (样本数,784)
        y_test: 测试标签数组
    """
    # =====================读取训练图片 train-images-idx3-ubyte=====================
    train_img_path = os.path.join(local_data_path, "train-images-idx3-ubyte")
    # 以二进制只读模式打开文件
    with open(train_img_path, "rb") as f:
        # 读取前16字节头部：4个uint32（魔数、样本数量、高度、宽度）
        header = np.frombuffer(f.read(16), dtype=np.uint32)
        # MNIST文件是大端序，当前系统读取字节顺序相反，执行字节交换
        header = header.byteswap()
        magic, num_train, rows, cols = header
        # 读取剩余所有像素字节，像素范围0~255
        train_img_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 将一维像素重塑为 [样本数量, 784]，28*28=784，展平成一维向量作为神经网络输入
    X_train = train_img_data.reshape(num_train, rows * cols)

    # =====================读取训练标签 train-labels-idx1-ubyte=====================
    train_label_path = os.path.join(local_data_path, "train-labels-idx1-ubyte")
    with open(train_label_path, "rb") as f:
        # 标签文件头部只占8字节：魔数、样本数量
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        header = header.byteswap()
        magic, num_label = header
        # 读取标签数据，0~9数字
        y_train = np.frombuffer(f.read(), dtype=np.uint8)

    # =====================读取测试图片 t10k-images-idx3-ubyte=====================
    test_img_path = os.path.join(local_data_path, "t10k-images-idx3-ubyte")
    with open(test_img_path, "rb") as f:
        header = np.frombuffer(f.read(16), dtype=np.uint32)
        header = header.byteswap()
        magic, num_test, rows, cols = header
        test_img_data = np.frombuffer(f.read(), dtype=np.uint8)
    X_test = test_img_data.reshape(num_test, rows * cols)

    # =====================读取测试标签 t10k-labels-idx1-ubyte=====================
    test_label_path = os.path.join(local_data_path, "t10k-labels-idx1-ubyte")
    with open(test_label_path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        header = header.byteswap()
        magic, num_label = header
        y_test = np.frombuffer(f.read(), dtype=np.uint8)

    # 将像素归一化：原始0~255 → 0~1浮点数，神经网络训练必须归一化，加速收敛
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

# ----------------主程序入口----------------
if __name__ == "__main__":
    # 定义本地MNIST数据集文件夹路径
    data_root = r"D:\Repositories\basic_algorithm\database\mnist"
    # 调用自定义函数加载mnist全部数据
    X_train, y_train, X_test, y_test = load_mnist(data_root)

    print("训练集形状：", X_train.shape)
    print("测试集形状：", X_test.shape)

    # =====================搭建传统全连接神经网络 MLPClassifier=====================
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),  #2层隐藏层，第一层256个神经元，第二层128个神经元
        activation="relu",  # 隐藏层激活函数
        solver="adam",  # 优化器，自适应学习率梯度下降
        batch_size=64,  # 批次大小：每次送入网络256个样本更新权重
        max_iter=5,  # 最大训练轮数(Epoch)，完整遍历数据集一次为一轮
        random_state=42,  # 随机种子，保证实验可复现
        early_stopping=True,  # 开启早停，验证集损失不再下降时提前终止，防止过拟合
        tol=1e-2,  # 损失变化小于该值判定收敛
        shuffle=True,   #打乱顺序训练
        verbose=True  # 打印训练日志

    )

    # 使用训练集数据训练神经网络模型
    # fit(X,y)：X特征矩阵，y对应标签
    model.fit(X_train, y_train)

    # 使用训练好模型分别预测训练集、测试集结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 计算训练集准确率：真实标签和预测标签对比
    train_acc = accuracy_score(y_train, y_train_pred)
    # 计算测试集准确率
    test_acc = accuracy_score(y_test, y_test_pred)

    # 格式化输出准确率
    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # 使用joblib保存训练完成的模型到本地文件
    # 参数1：要保存的模型对象；参数2：保存文件路径
    joblib.dump(model, "mnist_mlp_model.pkl")
    print("\n模型已保存为：mnist_mlp_model.pkl")