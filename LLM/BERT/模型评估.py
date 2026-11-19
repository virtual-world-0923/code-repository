import torch
import torch.nn as nn
import project_config as pc

def evaluate_model(model, loader, device=pc.DEVICE):
    """
    在给定数据加载器上评估模型准确率与平均损失。
    参数:
        model  (nn.Module): 待评估模型
        loader (DataLoader): 数据加载器
        device (torch.device): 计算设备
    返回:
        (float, float): (平均损失, 准确率)
    """
    model.eval()                  # 切换为评估模式（关闭 dropout 等）
    correct = total = 0          # 正确数 / 总数
    loss_sum = 0.0               # 累计损失
    # torch.no_grad()：关闭梯度计算，节省显存并加速
    with torch.no_grad():
        for batch in loader:
            # 将输入张量搬到目标设备
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits = out.logits                     # 模型输出 logits: [B, num_labels]
            labels = batch["label"].to(device)     # 真实标签
            # 计算交叉熵损失
            loss = nn.CrossEntropyLoss()(logits, labels)
            preds = torch.argmax(logits, dim=-1)    # 取最大值的下标作为预测
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
    return loss_sum / total, correct / total
