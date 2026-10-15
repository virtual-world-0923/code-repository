import torch                                           # 导入PyTorch主模块，提供张量计算、自动求导等核心功能

def predicted_model(model, images):
    """
    使用模型进行预测

    参数说明:
        model       (nn.Module)  : 已训练完成的AlexNet模型
        test_loader (DataLoader) : 测试数据加载器，提供批量测试数据
        criterion   (nn.Module)  : 损失函数，用于计算测试损失

    返回值:
        预测值
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()检测CUDA是否可用
    print(f"\测试---当前计算设备: {device}")  # 打印所选的计算设备名称
    model.to(device)
    model.eval()                                          # 将模型切换为评估模式：关闭Dropout层，BatchNorm使用全局统计量

    test_loss = 0.0                                       # 初始化测试累计损失为0
    correct = 0                                           # 初始化正确预测计数为0
    total = 0                                             # 初始化总样本计数为0

    # 禁用梯度计算上下文管理器：在推理阶段不需要计算梯度，可节省内存并加速计算
    with torch.no_grad():                                 # torch.no_grad()上下文管理器内的所有操作不会构建计算图
        images = images.to(device)                   # 将测试标签batch移动到计算设备

        outputs = model(images)                       # 前向传播：将测试图片输入模型得到预测输出logits

        highest_score, class_index = torch.max(outputs, dim=1)      # 获取每个样本的预测类别索引

    return highest_score, class_index                             # 返回平均损失和准确率

