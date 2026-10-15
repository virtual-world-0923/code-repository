# 导入 TensorFlow 库，并设置别名为 tf，用于加载模型和处理图像
import tensorflow as tf
# 导入 numpy 库并设置别名为 np，用于对图像数组进行数学运算和维度操作
import numpy as np
# 导入 matplotlib.pyplot 用于显示图片（可选，方便直观查看预测的图片）
import matplotlib.pyplot as plt

# ==================== 1. 加载训练好的模型 ====================

# 定义已保存模型的本地路径（需与 train_and_test.py 中保存的路径一致）
# 参数说明：字符串类型，指向 .keras 格式的模型文件
MODEL_PATH = "mnist_traditional_nn_model.keras"

# 使用 tf.keras.models.load_model 加载完整的 Keras 模型
# 参数说明：MODEL_PATH 是模型文件路径；该函数会自动恢复模型结构、权重和编译配置
model = tf.keras.models.load_model(MODEL_PATH)

# 打印模型加载成功的提示信息
# 参数说明：无参数，直接在控制台输出确认信息
print("模型加载成功！")

# ==================== 2. 读取并预处理本地图片 ====================

# 定义待预测的本地手写数字图片路径
# 参数说明：字符串类型，指向一张 28x28 像素的手写数字图片（支持 png/jpg 等格式）
IMAGE_PATH = "test_digit.png"

# 使用 tf.io.read_file 读取本地图片的原始字节内容
# 参数说明：IMAGE_PATH 是图片文件路径；返回一个包含图片二进制数据的标量字符串张量
img_raw = tf.io.read_file(IMAGE_PATH)

# 使用 tf.io.decode_image 将原始字节解码为图像张量
# 参数说明：img_raw 是原始字节数据；channels=1 表示强制解码为单通道（灰度图）
# 因为 MNIST 是灰度图，指定 channels=1 可避免 RGB 三通道带来的维度不匹配问题
img = tf.io.decode_image(img_raw, channels=1)

# 将图像数据类型转换为 float32，以便进行后续的浮点运算
# 参数说明：img 是 uint8 类型的图像张量；tf.cast 执行类型转换，目标类型为 tf.float32
img = tf.cast(img, tf.float32)

# 将像素值从 [0, 255] 归一化到 [0, 1] 区间
# 参数说明：与训练时的预处理保持一致，除以 255.0 使输入分布与训练时相同
img = img / 255.0

# 将 28x28 的二维图像展平为一维的 784 维向量
# 参数说明：img 当前形状为 (28, 28, 1)；tf.reshape 将其重塑为 (784,) 以匹配模型输入层
img = tf.reshape(img, shape=(784,))

# 在 batch 维度上增加一个维度，将形状从 (784,) 变为 (1, 784)
# 参数说明：tf.expand_dims 在 axis=0 的位置插入一个大小为 1 的维度
# 因为模型期望的输入是批次数据 (batch_size, 784)，单张图片需要凑成 batch 格式
img_batch = tf.expand_dims(img, axis=0)

# ==================== 3. 模型预测与结果输出 ====================

# 使用模型对预处理后的图片进行预测
# 参数说明：img_batch 是形状为 (1, 784) 的输入张量；返回值是形状为 (1, 10) 的概率数组
predictions = model.predict(img_batch)

# 使用 np.argmax 获取概率数组中最大值的索引，即预测的数字类别
# 参数说明：predictions[0] 取出批次中的第一个（也是唯一一个）样本的 10 个类别概率
# axis=0 表示在第 0 轴（即 10 个类别）上寻找最大值的索引
predicted_digit = np.argmax(predictions[0])

# 获取预测数字对应的置信度（概率值）
# 参数说明：predictions[0][predicted_digit] 通过索引取出最大概率值
confidence = predictions[0][predicted_digit]

# 在控制台打印预测结果和置信度
# 参数说明：predicted_digit 是 0-9 的整数；confidence 是 0~1 的浮点数，乘 100 转为百分比
print(f"预测数字: {predicted_digit}")
print(f"置信度: {confidence * 100:.2f}%")

# （可选）使用 matplotlib 显示原始图片及预测结果，方便人工核验
# 参数说明：img 是归一化后的张量，需乘 255 转回 uint8 显示；cmap='gray' 指定灰度色彩映射
plt.imshow(tf.reshape(img, (28, 28)) * 255, cmap='gray')

# 设置图表标题，显示预测数字和置信度
# 参数说明：f-string 格式化字符串，将预测结果嵌入标题中
plt.title(f"Prediction: {predicted_digit} ({confidence * 100:.2f}%)")

# 关闭坐标轴刻度，使图片显示更美观
# 参数说明：无参数，隐藏 x 轴和 y 轴的刻度线和标签
plt.axis('off')

# 显示图片窗口
# 参数说明：无参数，阻塞式弹出 matplotlib 窗口展示图片
plt.show()