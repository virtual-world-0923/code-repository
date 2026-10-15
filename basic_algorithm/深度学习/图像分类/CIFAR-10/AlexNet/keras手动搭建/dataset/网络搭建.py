import keras                  # 导入独立Keras 3.x库，用于构建网络模型、编译训练和保存加载
from keras import layers      # 从Keras中导入layers模块，提供Conv2D、Dense、MaxPooling2D等网络层定义

# keras.Sequential: 创建顺序模型，各层按添加顺序线性堆叠
# 参数: name='AlexNet' — 模型名称，用于打印和保存时标识
model = keras.Sequential(name='AlexNet')

# 添加输入层，明确指定模型接收的输入张量形状
# keras.Input: 创建一个Keras符号张量(SymbolicTensor)作为模型的输入占位符
# 参数:
#   shape=(227, 227, 3) — 输入数据的形状(不含batch维度)
#     227=高度, 227=宽度, 3=RGB通道数
#   name='input' — 该层的名称标识
# 有了Input层后，后续各层无需再指定input_shape参数
model.add(keras.Input(shape=(227, 227, 3), name='input'))

# ======================== 第1卷积层 (Conv1) ========================
# layers.Conv2D: 二维卷积层，使用可学习的卷积核对输入特征图进行卷积运算
# 参数说明:
#   filters=96 — 卷积核(滤波器)的数量，决定输出特征图的通道数
#   kernel_size=(11, 11) — 每个卷积核的空间尺寸为11×11像素
#   strides=(4, 4) — 卷积核在水平和垂直方向的滑动步长为4个像素
#   padding='valid' — 不进行零填充(无padding)，输出尺寸会小于输入
#   activation='relu' — ReLU激活函数: f(x) = max(0, x)
#     将所有负值置零，引入非线性变换能力
#   name='conv1' — 层名称
# 输入形状: (227, 227, 3) → 卷积输出形状: (55, 55, 96)
# 计算: (227 - 11) / 4 + 1 = 55
model.add(layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                        padding='valid', activation='relu', name='conv1'))

# layers.BatchNormalization: 批归一化层
# 对每个mini-batch内每个通道的特征进行标准化(减均值除标准差)，然后进行可学习的仿射变换
# 作用: 加速训练收敛、缓解梯度消失/爆炸、允许使用更大的学习率、起到正则化效果
# 参数:
#   axis=-1 — 归一化轴，-1表示对最后一个维度(即通道维度)进行归一化
#   momentum=0.99 — 指数移动平均的动量系数(用于计算全局均值和方差)
#   epsilon=0.001 — 防止除零的小常数，加在分母方差上
# 可训练参数: 每通道2个(gamma缩放因子和beta偏移因子)
model.add(layers.BatchNormalization(axis=-1, name='bn1'))

# layers.MaxPooling2D: 二维最大池化层
# 在每个不重叠的池化窗口内取最大值，用于降低特征图的空间分辨率
# 作用: 减少参数量和计算量、增加特征的平移不变性、扩大感受野
# 参数:
#   pool_size=(3, 3) — 池化窗口大小为3×3
#   strides=(2, 2) — 池化窗口滑动步长为2
#   padding='valid' — 不进行零填充
# 输入: (55, 55, 96) → 输出: (27, 27, 96)
# 计算: (55 - 3) / 2 + 1 = 27
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', name='pool1'))


# ======================== 第2卷积层 (Conv2) ========================
# 第2卷积层: 256个5×5卷积核，使用same填充保持空间尺寸不变
# 参数:
#   filters=256 — 输出256个特征图(通道)
#   kernel_size=(5, 5) — 5×5卷积核
#   strides=(1, 1) — 步长为1(默认值，可省略)
#   padding='same' — 零填充使输出空间尺寸与输入相同
#     具体填充量由框架自动计算: pad = kernel_size - 1 (对称两侧各一半)
# 输入: (27, 27, 96) → 输出: (27, 27, 256)
model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                        padding='same', activation='relu', name='conv2'))

# 第2层批归一化
model.add(layers.BatchNormalization(axis=-1, name='bn2'))

# 第2次最大池化: 进一步压缩特征图的空间尺寸
# 输入: (27, 27, 256) → 输出: (13, 13, 256)
# 计算: (27 - 3) / 2 + 1 = 13
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', name='pool2'))


# ======================== 第3卷积层 (Conv3) ========================
# 第3卷积层: 384个3×3卷积核，same填充
# 更小的卷积核(3×3)可以捕获更精细的局部特征
# 输入: (13, 13, 256) → 输出: (13, 13, 384)
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu', name='conv3'))

# 第3层批归一化
model.add(layers.BatchNormalization(axis=-1, name='bn3'))


# ======================== 第4卷积层 (Conv4) ========================
# 第4卷积层: 384个3×3卷积核，same填充
# 输入: (13, 13, 384) → 输出: (13, 13, 384)
model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu', name='conv4'))

# 第4层批归一化
model.add(layers.BatchNormalization(axis=-1, name='bn4'))


# ======================== 第5卷积层 (Conv5) ========================
# 第5卷积层: 256个3×3卷积核，same填充
# 输入: (13, 13, 384) → 输出: (13, 13, 256)
model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu', name='conv5'))

# 第5层批归一化
model.add(layers.BatchNormalization(axis=-1, name='bn5'))

# 第3次最大池化: 最后一次空间下采样
# 输入: (13, 13, 256) → 输出: (6, 6, 256)
# 计算: (13 - 3) / 2 + 1 = 6
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='valid', name='pool5'))


# ======================== 展平层 (Flatten) ========================
# layers.Flatten: 将多维特征图展平为一维向量
# 这是卷积层到全连接层之间的桥梁
# 输入: (6, 6, 256) — 三维特征图
# 输出: (9216,) — 一维向量
# 计算: 6 × 6 × 256 = 9216
model.add(layers.Flatten(name='flatten'))


# ======================== 第6层: 全连接层 (FC6) ========================
# layers.Dense: 全连接(密集)层，每个输入节点与每个输出节点之间都有连接权重
# 参数:
#   units=4096 — 输出神经元数量
#   activation='relu' — ReLU激活函数
# 可训练参数: 9216 × 4096 + 4096(偏置) = 37,752,832
# 输入: (9216,) → 输出: (4096,)
model.add(layers.Dense(units=4096, activation='relu', name='fc6'))

# layers.Dropout: 随机失活正则化层
# 训练时: 以rate概率随机将神经元输出置为0(即"关闭"该神经元)
#   这迫使网络学习冗余表示，减少对特定神经元的依赖，防止过拟合
# 推理时: 所有神经元均参与计算，输出按(1-rate)缩放以保持期望值一致
# 参数:
#   rate=0.5 — 随机丢弃50%的神经元
model.add(layers.Dropout(rate=0.5, name='dropout6'))


# ======================== 第7层: 全连接层 (FC7) ========================
# 第2个全连接层，结构与FC6相同
# 可训练参数: 4096 × 4096 + 4096 = 16,781,312
# 输入: (4096,) → 输出: (4096,)
model.add(layers.Dense(units=4096, activation='relu', name='fc7'))

# 对FC7层输出应用Dropout正则化，丢弃率同样为50%
model.add(layers.Dropout(rate=0.5, name='dropout7'))


# ======================== 第8层: 输出层 ========================
# 输出全连接层: 神经元数量等于分类类别数(10)
# 参数:
#   units=10 — CIFAR-10数据集包含10个类别
#   activation='softmax' — Softmax激活函数
#     将原始输出(logits)转换为概率分布: softmax(z_i) = exp(z_i) / Σexp(z_j)
#     所有输出值在[0,1]之间，且总和为1，每个值表示对应类别的预测概率
# 输入: (4096,) → 输出: (10,)
model.add(layers.Dense(units=10, activation='softmax', name='output'))


# --- 2.2 编译模型 ---
# model.compile: 配置模型的训练过程，指定优化器、损失函数和评估指标
# 参数:
#   optimizer='adam' — Adam优化器(Adaptive Moment Estimation)
#     结合了Momentum(动量)和RMSProp(自适应学习率)的优点
#     内部维护梯度的一阶矩(均值)和二阶矩(方差)的指数移动平均
#     默认学习率lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
#   loss='sparse_categorical_crossentropy' — 稀疏分类交叉熵损失函数
#     适用于标签为整数形式(如0,1,2,...,9)的多分类问题
#     损失公式: L = -Σ y_true * log(y_pred)，其中y_true为one-hot真实标签
#     "sparse"版本自动将整数标签转为one-hot编码，无需手动转换
#     如果标签已经是one-hot编码，应使用'categorical_crossentropy'
#   metrics=['accuracy'] — 评估指标列表
#     'accuracy'计算分类准确率 = 正确预测数 / 总样本数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ==============================================================================
# 第三部分：打印模型结构
# ==============================================================================

# 打印分隔线和标题
print("\n" + "=" * 70)
print("AlexNet 模型结构摘要:")
print("=" * 70)

# model.summary(): 打印模型的详细结构摘要表格
# 表格包含: 层名称(Type)、输出形状(Output Shape)、参数数量(Param #)
# 可选参数:
#   line_length (int) — 打印表格的总字符宽度(默认根据终端自动调整)
#   print_fn (callable) — 自定义打印函数(默认使用print)
#   show_trainable (bool) — 是否显示每层的可训练状态(Keras 3新增)
model.summary(line_length=90)

# 尝试将模型结构可视化为图片文件
# keras.utils.plot_model: 将Keras模型结构绘制成图形并保存
# 参数:
#   model — 要可视化的Keras模型对象
#   to_file (str) — 输出图片文件的保存路径
#   show_shapes=True — 在图中显示每层的输入和输出张量形状
#   show_layer_names=True — 在图中显示每层的名称
# 注意: 此函数依赖graphviz系统库和pydot Python库
#   若未安装会抛出异常，因此用try-except捕获
try:
    keras.utils.plot_model(model, to_file='alexnet_structure.png',
                           show_shapes=True, show_layer_names=True)
    print("\n模型结构图已保存到: alexnet_structure.png")
except Exception as e:
    # 捕获graphviz/pydot缺失等异常，打印提示但不中断后续训练流程
    print(f"\n模型结构图保存失败(可能需要安装 graphviz 和 pydot): {e}")

