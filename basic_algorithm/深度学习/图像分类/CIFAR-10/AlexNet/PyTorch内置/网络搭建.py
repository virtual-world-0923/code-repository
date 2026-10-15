import torch.nn as nn                                  # 导入PyTorch神经网络模块，提供层定义、损失函数等组件
from torchvision.models import alexnet                 # 从torchvision导入PyTorch内置的AlexNet模型构建函数
from torchvision.models import AlexNet_Weights         # 导入AlexNet预训练权重枚举类，用于指定加载哪种预训练权重

print("\n正在加载PyTorch内置AlexNet预训练模型...")           # 打印模型加载开始的提示信息

# 使用torchvision内置的alexnet()函数构建AlexNet网络，并加载ImageNet预训练权重
# weights=AlexNet_Weights.IMAGENET1K_V1 表示加载在ImageNet-1K数据集上预训练的V1版本权重
model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1) # 创建AlexNet模型实例并自动下载/缓存ImageNet预训练权重参数

print("AlexNet预训练模型加载完成！")                        # 打印模型加载完成的确认信息

# ==================== 第九部分：修改最后全连接层以适配CIFAR-10的10分类任务 ====================
# AlexNet原始设计用于ImageNet的1000类分类，最后一层classifier[6]输出维度为1000
# CIFAR-10只有10个类别，因此需要将最后一层替换为输出维度为10的全连接层

# 获取原始最后一层的输入特征维度（AlexNet中为4096）
num_features = model.classifier[6].in_features         # in_features属性返回该全连接层的输入神经元数量，AlexNet中为4096

# 用新的全连接层替换原始最后一层，将输出从1000改为10
model.classifier[6] = nn.Linear(                       # nn.Linear创建全连接层（线性变换层 y = xW^T + b）
    in_features=num_features,                           # 输入特征维度：4096，保持与前一层输出一致
    out_features=10                                     # 输出特征维度：10，对应CIFAR-10的10个类别（airplane, automobile, bird, ...）
)

print("已将AlexNet最后一个全连接层修改为10分类输出")        # 打印网络结构修改完成的提示信息

# ==================== 第九-2部分：冻结除最后一层外的所有层 ====================
# 遍历模型的每一个参数，按名称判断是否属于最后一层
# 不属于最后一层的参数：冻结（requires_grad=False），训练时不会更新
# 属于最后一层的参数：保持可训练（requires_grad=True），训练时正常更新
for name, param in model.named_parameters():
    if "classifier.6" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 统计冻结与可训练参数数量
frozen_params = sum(p.numel() for p in model.parameters()
                    if not p.requires_grad)
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
print(f"冻结参数数量: {frozen_params:,}")
print(f"可训练参数数量（仅最后一层）: {trainable_params:,}")

# ==================== 第十部分：打印完整的模型网络结构 ====================
print("\n" + "=" * 70)                                   # 打印由70个等号组成的分隔线
print("                    AlexNet 完整模型结构")          # 打印模型结构的标题
print("=" * 70)                                         # 打印分隔线
print(model)                                            # 直接打印model对象，PyTorch会自动格式化输出所有层的名称、类型和参数信息
print("=" * 70)                                         # 打印分隔线

# 统计并打印模型的总参数量和可训练参数量
total_params = sum(p.numel() for p in model.parameters())           # 计算模型所有参数的总数量（包括冻结参数）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算requires_grad=True的可训练参数数量
print(f"模型总参数量: {total_params:,}")                              # 打印总参数量，使用千位分隔符格式化
print(f"可训练参数量: {trainable_params:,}")                          # 打印可训练参数量
print("=" * 70)                                                      # 打印分隔线
