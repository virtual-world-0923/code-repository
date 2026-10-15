
def train_model(model, train_dataset, test_dataset,EPOCHS):
    for x, y in test_dataset.take(1):
        print(x.shape, y.shape)
    # model.fit: 执行模型训练，通过前向传播计算损失、反向传播更新权重
    # 参数:
    #   x — 训练输入数据，形状(N, H, W, C)的numpy数组，即(10000, 227, 227, 3)
    #   y — 训练标签数据，形状(N,)的numpy数组，值为0~9的整数
    #   epochs=EPOCHS — 训练总轮数，每轮完整遍历一次训练集
    #   batch_size=BATCH_SIZE — 每次梯度更新使用的样本数
    #     10000个样本 / 64 = 156个批次(最后一个批次可能不足64个)
    #   validation_data=(x_test, y_test) — 验证数据元组
    #     每个epoch结束后，在该数据集上评估模型的损失和指标
    #     该数据不参与梯度更新，仅用于监控模型泛化性能
    #   verbose=1 — 日志显示模式
    #     0=静默无输出, 1=显示进度条和每轮指标, 2=每轮输出一行日志
    # 返回:
    #   history — Keras History对象，包含训练过程中的详细记录
    #     history.history (dict): 键为字符串指标名，值为每轮指标值的列表
    #       'loss': 每轮训练损失
    #       'accuracy': 每轮训练准确率
    #       'val_loss': 每轮验证损失
    #       'val_accuracy': 每轮验证准确率
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,                          # 训练轮数
        validation_data=test_dataset,       # 用测试集作为验证集监控泛化性能
        verbose=1                               # 显示训练进度条
    )
    return model, history
