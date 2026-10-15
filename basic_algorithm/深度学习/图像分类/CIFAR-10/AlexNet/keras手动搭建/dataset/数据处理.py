import numpy as np            # 导入numpy库，用于高效的多维数组创建、变形和数值计算
import database.read_cifar as rc
import tensorflow as tf       # 导入TensorFlow框架，作为Keras 3.x的计算后端，提供张量运算和GPU支持
from keras.ops.image import resize

#读取数据
d = rc.read_data(1)
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
# bCHW -> bHWC
def _dimension_transpose(data, location:tuple=(0,2,3,1)):
    if data.ndim == 2:
        data = data.reshape(-1, 3, 32, 32)
    if data.ndim == 4:
        data = np.transpose(data, location)
    return data

#keras 处理图片的格式：bHWC
train_images = _dimension_transpose(train_images)
train_labels = _dimension_transpose(train_labels)
test_images =  _dimension_transpose(test_images)
test_labels = _dimension_transpose(test_labels)


def preprocess(image, label):
    """
    输入: image shape = (H, W, C) 或 (B, H, W, C)，值域 0~255 uint8
    输出: image shape = (B, 227, 227, C)，值域标准化后 float32
    """
    # 1. Resize 到 227×227
    #method = tf.image.ResizeMethod.BILINEAR(默认) — 双线性插值方法
    #输入BHWC 或 HWC
    #输出BHWC 或 HWC
    image = tf.image.resize(image, [227, 227])
    # 或
    # resized = resize(image, size=(227, 227), interpolation="bilinear")

    # 将像素值归一化到[0, 1]范围
    # 除以255.0将原始uint8范围(0~255)映射到float32范围(0.0~1.0)
    # 归一化有助于: 加速梯度下降收敛、避免数值溢出、提升训练稳定性
    image = tf.cast(image, tf.float32) / 255.0

    # 3. ImageNet 均值方差标准化
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image - mean) / std

    return image, label

BATCH_SIZE = 64 #批次，一次训练的数据量
#把x和y切片对应起来，返回(x, y) 元组
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

#每条数据执行preprocess函数，取1000条放到缓冲，打乱顺序，每批次取出64条数据
train_dataset = train_dataset.map(preprocess).shuffle(1000).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(preprocess).shuffle(1000).batch(BATCH_SIZE)


#AlexNet原始输入为227×227，因此将32×32的CIFAR-10图像上采样到227×227
#Keras 默认是 channels_last（bHWC），如果你用 CHW 格式，需要告诉它channels_first
# model.add(layers.Resizing(227, 227, data_format="channels_last"),)


