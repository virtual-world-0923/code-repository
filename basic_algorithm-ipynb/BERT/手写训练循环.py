import os                      # 操作系统接口：用于创建目录、拼接路径
import time                    # 时间：用于统计每轮训练耗时
import torch                   # PyTorch 深度学习框架：张量计算与自动求导
import torch.nn as nn          # PyTorch 神经网络模块：损失函数、层定义
import evaluate                          # HuggingFace evaluate 库：加载评估指标（accuracy 等）
from transformers import (
    AutoTokenizer,                        # 自动选择对应的分词器（tokenizer）
    AutoConfig,                           # 自动加载模型配置文件
    AutoModelForSequenceClassification,   # 自动加载带分类头的序列分类模型
)
import project_config as pc

config = AutoConfig.from_pretrained(pc.MODEL_NAME_BERT, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    pc.MODEL_NAME_BERT, config=config
).to(pc.DEVICE)

def run_mode_A_handcraft(train_loader):
    """
    方式 A：使用原生 PyTorch 手写训练循环微调 bert-base-chinese。
    包含：训练 → 验证 → 测试 → 保存 → 演示预测
    """
    print("\n" + "=" * 70)
    print("【方式 A】原生 PyTorch 手写训练循环  |  bert-base-chinese")
    print("=" * 70)

    # ---- 优化器与损失函数 ----
    # AdamW：带权重衰减的 Adam，BERT 微调首选优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=pc.LR, weight_decay=pc.WEIGHT_DECAY)
    # 交叉熵损失：多分类标准损失
    criterion = nn.CrossEntropyLoss()

    # ---- 训练循环 ----
    print("[训练] 开始 ...")
    for epoch in range(pc.EPOCHS):
        print(f"Epoch {epoch + 1}/{pc.EPOCHS}")
        model.train()  # 切换为训练模式（启用 dropout 等）
        run_loss = 0.0
        run_acc = 0.0
        cnt = 0
        t0 = time.time()  # 计时起点

        for step, batch in enumerate(train_loader):
            # 清零梯度（否则会累加）
            optimizer.zero_grad()
            # 前向传播：输入搬到设备
            out   = model(
                input_ids=batch["input_ids"].to(pc.DEVICE),
                attention_mask=batch["attention_mask"].to(pc.DEVICE),
            )
            logits = out.logits
            labels = batch["label"].to(pc.DEVICE)
            # 计算损失
            loss   = criterion(logits, labels)
            # 反向传播：计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            bs   = labels.size(0)  # 当前 batch 大小
            acc  = (torch.argmax(logits, dim=-1) == labels).float().mean().item()
            run_loss += loss.item() * bs
            run_acc  += acc * bs
            cnt      += bs

            # 每 50 步打印一次中间日志
            if (step + 1) % 50 == 0:
                print(f"  ep{epoch+1} step{step+1} loss={loss.item():.4f} acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"一个epoch花费time={elapsed:.1f}s")
    return model




    # 每个 epoch 结束，在验证集上评估
    v_loss, v_acc = evaluate_model(model, val_loader)
    elapsed = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"train_loss={run_loss/cnt:.4f} train_acc={run_acc/cnt:.4f}  "
          f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}  "
          f"time={elapsed:.1f}s")

# ---- 测试集评估 ----
# t_loss, t_acc = evaluate_model(model, test_loader)
# print(f"\n[测试] loss={t_loss:.4f}  accuracy={t_acc:.4f}")

