import torch                                           # 导入PyTorch主模块，提供张量计算、自动求导等核心功能

def evaluate_model(model, test_loader, criterion):
    """
    在测试集上评估已训练模型的性能

    参数说明:
        model       (nn.Module)  : 已训练完成的AlexNet模型
        test_loader (DataLoader) : 测试数据加载器，提供批量测试数据
        criterion   (nn.Module)  : 损失函数，用于计算测试损失
        device      (torch.device): 计算设备

    返回值:
        test_loss (float): 测试集上的平均损失值
        test_acc  (float): 测试集上的分类准确率(%)
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
        for images, labels in test_loader:                # 遍历测试数据加载器中的每个batch
            images = images.to(device)                    # 将测试图片batch移动到计算设备
            labels = labels.to(device)                    # 将测试标签batch移动到计算设备

            outputs = model(images)                       # 前向传播：将测试图片输入模型得到预测输出logits
            loss = criterion(outputs, labels)             # 计算当前batch的测试损失

            test_loss += loss.item()                      # 累加当前batch的损失值
            _, predicted = torch.max(outputs, dim=1)      # 获取每个样本的预测类别索引
            total += labels.size(0)                       # 累加当前batch的样本数
            correct += (predicted == labels).sum().item() # 累加当前batch中预测正确的样本数

    # 计算最终的测试集评估指标
    avg_loss = test_loss / len(test_loader)               # 平均测试损失 = 总测试损失 / 测试batch总数
    accuracy = 100.0 * correct / total                    # 测试准确率(%) = 正确数 / 总数 × 100

    # 打印测试集评估结果
    print(f"\n{'=' * 50}")                                # 打印分隔线
    print(f"  测试集评估结果")                              # 打印评估结果标题
    print(f"{'=' * 50}")                                  # 打印分隔线
    print(f"  平均损失  : {avg_loss:.4f}")                 # 打印平均测试损失
    print(f"  准确率    : {accuracy:.2f}% ({correct}/{total})")  # 打印准确率及正确数/总数
    print(f"{'=' * 50}")                                  # 打印分隔线

    return avg_loss, accuracy                             # 返回平均损失和准确率

print("\n" + "=" * 70)                                    # 打印大分隔线
print("           在测试集上评估训练后的AlexNet模型")       # 打印评估标题
print("=" * 70)                                          # 打印分隔线
