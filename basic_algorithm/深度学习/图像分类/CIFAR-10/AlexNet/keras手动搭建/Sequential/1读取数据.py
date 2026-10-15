import numpy as np            # 导入numpy库，用于高效的多维数组创建、变形和数值计算
import database.read_cifar as rc
import tensorflow as tf       # 导入TensorFlow框架，作为Keras 3.x的计算后端，提供张量运算和GPU支持
from keras.ops.image import resize

#读取数据
d = rc.read_data(1)
train_images = d["train_images"]    # b'data'键对应的值为形状(10000, 3072)的numpy数组，[R通道1024个像素][G通道1024个像素][B通道1024个像素]
train_labels = d["train_labels"]    # b'labels'键对应标签列表，转换为numpy数组便于后续处理，shape为(10000,)
test_images = d["test_images"]      # 形状为(10000, 3072)的numpy数组
test_labels = d["test_labels"]      # 形状为(10000,)的标签数组
label_names = d["label_names"]

# 分离三个通道: 每张图片的前1024个值是红色通道，接着1024绿色，最后1024蓝色
r_channel = train_images[:, :1024].reshape(-1, 32, 32)  # (N, 32, 32)
g_channel = train_images[:, 1024:2048].reshape(-1, 32, 32)  # (N, 32, 32)
b_channel = train_images[:, 2048:].reshape(-1, 32, 32)  # (N, 32, 32)
#在第一维度合并三通道 BCHW
train_images = np.stack([r_channel, g_channel, b_channel], axis=1)
train_labels = np.array(train_labels)

r_channel = test_images[:, :1024].reshape(-1, 32, 32)  # (N, 32, 32)
g_channel = test_images[:, 1024:2048].reshape(-1, 32, 32)  # (N, 32, 32)
b_channel = test_images[:, 2048:].reshape(-1, 32, 32)  # (N, 32, 32)
#在第一维度合并三通道 BCHW
test_images = np.stack([r_channel, g_channel, b_channel], axis=1)
test_labels = np.array(test_labels)


