import numpy as np            # 导入numpy库，用于高效的多维数组创建、变形和数值计算
import keras                  # 导入独立Keras 3.x库，用于构建网络模型、编译训练和保存加载
import runpy
from PIL import Image    # 从PIL(Pillow)库导入Image模块，用于打开、转换和缩放本地图片文件

# ==================== 设置随机种子以确保结果可复现 ====================
# keras.utils.set_random_seed: 一次性设置所有后端(TensorFlow、NumPy、Python内置random)的全局随机种子
# 参数: seed (int) - 随机种子值，相同种子保证每次运行产生相同的随机序列，便于结果复现
keras.utils.set_random_seed(42)
# 定义模型保存文件路径，使用Keras 3.x原生格式(.keras)
model_save_path = '../alexnet_cifar10.keras'

#################################  数据处理  #######################################
result = runpy.run_path("数据处理.py")
train_dataset = result["train_dataset"]
test_dataset = result["test_dataset"]
label_names = result["label_names"]
BATCH_SIZE = result["BATCH_SIZE"]

##################################  网络搭建  #################################
result = runpy.run_path("网络搭建.py")
model = result["model"]

#训练模型
EPOCHS = 1  # 训练总轮数，每轮完整遍历训练集一次

# 打印训练开始信息
print("\n" + "=" * 70)
print(f"开始训练 AlexNet (共 {EPOCHS} 轮, 批次大小 {BATCH_SIZE})...")
print("=" * 70)

##################################  训练模型  #################################
result = runpy.run_path("训练模型.py")
train_model = result["train_model"] #获取模块级函数
train_model(model, train_dataset, test_dataset, EPOCHS)

##################################  评估模型  #################################
result = runpy.run_path("评估模型.py")
evaluate_model = result["evaluate_model"]
evaluate_model(train_model, train_model, test_dataset)

###################################  保存模型  ##################################

# model.save: 将完整的Keras模型保存到磁盘文件
# 保存内容包括:
#   1. 模型结构(网络拓扑和层配置)
#   2. 模型权重(所有可训练和不可训练参数)
#   3. 优化器状态(用于断点续训)
#   4. 编译配置(loss、metrics等)
# 参数:
#   filepath (str) — 模型保存路径
#     .keras格式是Keras 3.x推荐的原生保存格式(底层为ZIP压缩包)
#     也支持.h5(HDF5)格式，但.keras格式更轻量且功能更完整
model.save(model_save_path)

# 打印保存成功提示和文件大小
# os.path.getsize: 获取文件大小(字节数)
# 除以(1024*1024)将字节转换为兆字节(MB)
print(f"\n模型已保存到: {model_save_path}")

##################################  预测模型  #################################
result = runpy.run_path("预测模型.py")
predicted_model = result["predicted_model"] #获取模块级函数

# 加载保存的Keras模型
load_model = keras.models.load_model(model_save_path)
print("模型加载成功！")

#获取待预测图片的路径
image_path = 'test_image.jpg'

# Image.open: PIL库的核心图片加载函数
# 参数:
#   fp (str 或 file object) - 图片文件路径(字符串)或已打开的文件对象
# 返回: PIL.Image.Image 对象，包含图片的像素数据、尺寸、颜色模式等属性
# 支持格式: JPEG, PNG, BMP, TIFF, GIF, WebP等常见图片格式
img = Image.open(image_path)

# 打印原始图片信息
# img.size: 返回图片尺寸元组(宽度, 高度)，注意PIL的顺序是(宽, 高)
# img.mode: 返回颜色模式字符串，如'RGB'(3通道彩色), 'RGBA'(4通道含透明度), 'L'(灰度)
print(f"原始图片尺寸: {img.size}")
print(f"原始图片颜色模式: {img.mode}")

# Image.convert: 将图片转换为指定的颜色模式
# 参数:
#   mode (str) - 目标颜色模式
#     'RGB': 红绿蓝三通道彩色模式(不含透明度通道)
#     'L': 灰度模式(单通道)
#     'RGBA': 红绿蓝+透明度四通道模式
# 这一步是必要的，因为用户提供的图片可能是:
#   RGBA(含透明通道)、L(灰度)、P(调色板)等非标准RGB格式
# AlexNet期望3通道RGB输入，因此统一转为'RGB'
img_rgb = img.convert('RGB')

# Image.resize: 调整图片到指定尺寸
# 参数:
#   size (tuple) - 目标尺寸，格式为(宽度, 高度)
#     AlexNet的标准输入为227×227像素
#   resample - 重采样(插值)算法
#     Image.BILINEAR(值为2): 双线性插值，平衡质量和速度
#     其他选项: Image.NEAREST(最近邻，速度快但质量差),
#              Image.BICUBIC(双三次，质量好但稍慢)
# 返回: 调整尺寸后的新PIL.Image对象(原图不变)
img_resized = img_rgb.resize((227, 227), resample=Image.BILINEAR)

# np.array: 将PIL Image对象转换为numpy多维数组
# 参数: object - PIL Image对象(或任何可转换为数组的对象)
# 返回: numpy ndarray，形状(高度, 宽度, 通道)=(227, 227, 3)
#        数据类型dtype=uint8，像素值范围[0, 255]
img_array = np.array(img_resized)

# 将像素值转换为float32类型并归一化到[0, 1]范围
# .astype('float32'): 将uint8(0~255整数)转为float32浮点数
# / 255.0: 将值域从[0, 255]缩放到[0.0, 1.0]
# 注意: 必须与训练时的预处理方式保持一致(训练数据也是除以255.0归一化的)
img_array = img_array.astype('float32') / 255.0

# np.expand_dims: 在指定位置增加一个维度
# 参数:
#   a (ndarray) - 输入数组
#   axis (int) - 新增维度的位置
#     axis=0 表示在第0维(最前面)添加一个维度
# 作用: 将单张图片(227, 227, 3)扩展为批次格式(1, 227, 227, 3)
# 模型的输入要求是(N, H, W, C)的批次格式，N为批次大小
# 单张图片预测时N=1
img_batch = np.expand_dims(img_array, axis=0)

# 打印预处理后的张量信息
print(f"预处理后张量形状: {img_batch.shape}")     # 期望: (1, 227, 227, 3)
print(f"预处理后数据类型: {img_batch.dtype}")       # 期望: float32
print(f"预处理后像素值范围: [{img_batch.min():.3f}, {img_batch.max():.3f}]")  # [0.000, 1.000]

predictions = predicted_model(load_model, img_batch)

# predictions[0]: 取出第一张图片的预测结果
# 形状从(1, 10)变为(10,)的一维数组
predicted_probabilities = predictions[0]

# np.argmax: 返回数组中最大值所在位置的索引
# 参数:
#   a (ndarray) - 输入数组
#   axis (int, 可选) - 沿哪个轴查找最大值，默认为None(展平后查找)
# 返回: int - 最大值的索引，即预测类别的编号(0~9)
predicted_class_index = np.argmax(predicted_probabilities)

# 获取预测类别的置信度(即该类别的softmax概率值)
# predicted_probabilities[predicted_class_index]: 取预测概率向量中最大值
# 该值越接近1.0，表示模型对预测结果越确信
confidence = predicted_probabilities[predicted_class_index]

# 根据预测类别索引获取对应的类别名称
# label_names[predicted_class_index]: 用索引从列表中取出类别名字符串
predicted_class_name = label_names[predicted_class_index]

print("预测结果:")

# 打印预测类别信息
# predicted_class_index: 类别编号
# predicted_class_name: 类别名称
print(f"预测类别: [{predicted_class_index}] {predicted_class_name}")

# 打印置信度
# :.2% — 格式化为百分比形式保留2位小数(如0.9856 → 98.56%)
print(f"置信度: {confidence:.2%}")

# 打印所有10个类别的详细概率分布
print("\n各类别预测概率:")
# enumerate: 枚举函数，同时获取元素的索引和值
# 参数:
#   iterable - 可迭代对象(这里为zip对象)
#   start=0 - 起始索引编号(默认为0)
# zip: 将多个可迭代对象按位置配对
# zip(label_names, predicted_probabilities):
#   将类别名称列表和对应概率数组配对为元组序列
#   例如: [('airplane', 0.01), ('automobile', 0.02), ...]
for i, (name, prob) in enumerate(zip(label_names, predicted_probabilities)):
    # 打印每个类别的编号、名称和预测概率
    # name:12s — 将字符串左对齐并填充到12字符宽度，保证对齐
    # :.4f — 保留4位小数的浮点数
    # :.2% — 保留2位小数的百分比格式
    print(f"  [{i}] {name:12s}: {prob:.4f} ({prob:.2%})")

# 标记最高概率的类别(用箭头指示)
print(f"\n  >>> 预测结果: {predicted_class_name} (概率最高)")
