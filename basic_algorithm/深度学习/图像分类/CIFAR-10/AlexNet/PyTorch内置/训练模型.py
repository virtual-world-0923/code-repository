import torch                                           # 导入PyTorch主模块，提供张量计算、自动求导等核心功能
import torch.nn as nn  # 导入PyTorch神经网络模块，提供层定义、损失函数等组件
import torch.optim as optim  # 导入PyTorch优化器模块，提供SGD、Adam等优化算法

# 定义损失函数：交叉熵损失，适用于多分类任务
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss内部包含LogSoftmax和NLLLoss，输入为未归一化的logits

def train_model(model, train_loader ,num_epochs=3):
    """
    训练AlexNet模型的完整函数

    参数说明:
        model        (nn.Module)  : 待训练的AlexNet模型实例
        train_loader (DataLoader) : 训练数据加载器，提供批量(图片, 标签)数据
        criterion    (nn.Module)  : 损失函数，用于计算预测输出与真实标签之间的误差
        optimizer    (Optimizer)  : 优化器，负责根据梯度更新模型参数
        scheduler    (Scheduler)  : 学习率调度器，按策略调整学习率
        num_epochs   (int)        : 训练的总轮数，默认为10

    返回值:
        model (nn.Module): 训练完成后的模型
    """
    # 自动检测是否有可用的CUDA GPU，有则使用GPU加速，无则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()检测CUDA是否可用
    print(f"\训练---当前计算设备: {device}")  # 打印所选的计算设备名称

    # 定义优化器：随机梯度下降(SGD)，带动量项
    # optimizer = optim.SGD(                                  # 创建SGD优化器实例
    #     model.parameters(),                                 # 传入模型的所有可训练参数，优化器将更新这些参数以最小化损失
    #     lr=0.001,                                           # 学习率(learning rate)：控制每次参数更新的步长大小，微调预训练模型时使用较小值
    #     momentum=0.9                                        # 动量因子：引入历史梯度的惯性，加速收敛并减少训练过程中的震荡
    # )
    # 【修改】只传入requires_grad=True的参数，即只优化最后一层
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,  # ← 【修改】只训练最后一层，可用稍大的学习率
        momentum=0.9
    )

    # 定义学习率调度器：每隔固定epoch将学习率按比例衰减
    scheduler = optim.lr_scheduler.StepLR(  # StepLR按固定间隔将学习率乘以衰减因子
        optimizer=optimizer,  # 关联的优化器对象，调度器将修改其学习率
        step_size=5,  # 衰减间隔：每经过5个epoch触发一次学习率衰减
        gamma=0.1  # 衰减因子：每次衰减后新学习率 = 原学习率 × 0.1
    )

    # 将模型的所有参数和缓冲区移动到目标设备上
    model = model.to(device)  # .to(device)将模型整体转移到指定设备，确保模型参数与输入数据在同一设备上
    model.train()   # 将模型切换为训练模式：启用Dropout层的随机丢弃，BatchNorm使用当前batch的统计量

    # 记录所有epoch的平均损失和准确率，用于后续分析
    train_losses = []   # 创建空列表，用于存储每个epoch的平均训练损失
    train_accs = []     # 创建空列表，用于存储每个epoch的训练准确率

    # 外层循环：遍历每一个训练epoch
    for epoch in range(num_epochs):  # range(num_epochs)生成0到num_epochs-1的整数序列
        running_loss = 0.0           # 初始化当前epoch的累计损失为0，用于计算平均损失
        correct = 0                  # 初始化当前epoch的正确预测计数为0
        total = 0                    # 初始化当前epoch的已处理样本总数为0

        # 内层循环：遍历当前epoch中的每一个mini-batch
        for batch_idx, (images, labels) in enumerate(train_loader):
            # enumerate返回(batch序号, (图片张量, 标签张量))的元组
            # images.shape: (batch_size, 3, 227, 227)
            # labels.shape: (batch_size,)

            images = images.to(device)   # 将图片batch张量移动到目标计算设备（CPU或GPU）
            labels = labels.to(device)   # 将标签batch张量移动到目标计算设备

            optimizer.zero_grad()        # 清除上一次迭代中累积的梯度，PyTorch默认会累加梯度，因此每步必须手动清零 清空 .grad

            #接收BCHW (批次，通道，高，宽)
            outputs = model(images)                       # 前向传播：将batch图片输入AlexNet网络，得到输出logits，shape为(batch_size, 10)

            loss = criterion(outputs, labels)             # 计算交叉熵损失：outputs为未归一化的logits，labels为整数类别的标签
            loss.backward()                               # 反向传播：自动计算损失函数对所有requires_grad=True的参数的梯度,存到参数的.grad里

            optimizer.step()                              # 参数更新：用 param.grad 更新参数

            running_loss += loss.item()                   # loss.item()将单元素张量转为Python浮点数，累加到当前epoch的总损失中
            _, predicted = torch.max(outputs, dim=1)      # torch.max返回(values, indices)，取dim=1即沿类别维度找最大值索引作为预测类别
            total += labels.size(0)                       # labels.size(0)返回当前batch的样本数，累加到总样本计数中
            correct += (predicted == labels).sum().item() # 逐元素比较预测与真实标签，求和得到正确数，转为Python整数后累加

            # 每处理50个batch打印一次中间训练进度
            if (batch_idx + 1) % 50 == 0:                # 当batch序号+1是50的倍数时执行
                print(f"  Epoch [{epoch + 1}/{num_epochs}], "   # 打印当前epoch编号（从1开始计数）
                      f"Batch [{batch_idx + 1}/{len(train_loader)}], "  # 打印当前batch编号和总batch数
                      f"Loss: {loss.item():.4f}")                # 打印当前batch的损失值，保留4位小数

        # 当前epoch训练结束后，计算统计指标
        epoch_loss = running_loss / len(train_loader)     # 平均损失 = epoch总损失 / batch总数
        epoch_acc = 100.0 * correct / total               # 训练准确率(%) = 正确预测数 / 总样本数 × 100
        train_losses.append(epoch_loss)                   # 将当前epoch的平均损失记录到列表中
        train_accs.append(epoch_acc)                      # 将当前epoch的准确率记录到列表中

        # 打印当前epoch的汇总信息
        print(f"Epoch [{epoch + 1}/{num_epochs}] 完成 -> "  # 打印epoch完成标识
              f"平均损失: {epoch_loss:.4f}, "                  # 打印本epoch的平均训练损失
              f"训练准确率: {epoch_acc:.2f}%")                  # 打印本epoch的训练准确率

        scheduler.step()                                  # 调用学习率调度器，判断是否到达衰减步长并更新学习率
        current_lr = optimizer.param_groups[0]['lr']       # 从优化器的参数组中获取当前实际学习率值
        print(f"  当前学习率: {current_lr:.6f}")              # 打印当前学习率，保留6位小数

    print("\n模型训练完成！")                               # 打印全部训练结束的提示信息
    return model, train_losses, train_accs                # 返回训练好的模型以及训练过程中的损失和准确率记录
