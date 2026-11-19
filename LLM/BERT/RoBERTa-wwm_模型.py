import os                      # 操作系统接口：用于创建目录、拼接路径
import time                    # 时间：用于统计每轮训练耗时
import torch                   # PyTorch 深度学习框架：张量计算与自动求导
import torch.nn as nn          # PyTorch 神经网络模块：损失函数、层定义
import project_config as pc
# 从 transformers 库导入所需的类（v5.14.1 均支持）
from transformers import (
    AutoTokenizer,                        # 自动选择对应的分词器（tokenizer）
    AutoConfig,                           # 自动加载模型配置文件
    AutoModelForSequenceClassification,   # 自动加载带分类头的序列分类模型
)


def run_mode_C_roberta(train_loader):
    """
    方式 C：使用哈工大 chinese-roberta-wwm-ext 模型（整词掩码预训练）。
    采用手写训练循环（与方式 A 结构一致，仅模型不同）。
    RoBERTa-wwm 在中文任务上通常比原生 BERT 高 1~3 个点。
    """
    print("\n" + "=" * 70)
    print("【方式 C】RoBERTa-wwm  |  hfl/chinese-roberta-wwm-ext")
    print("=" * 70)

    # ---- 加载分词器与模型 ----
    # 注意：虽然模型名含 roberta，但 Chinese-RoBERTa 仍用 BertTokenizer/BertModel 结构
    tokenizer = AutoTokenizer.from_pretrained(pc.MODEL_NAME_ROBERTA)
    config    = AutoConfig.from_pretrained(pc.MODEL_NAME_ROBERTA, num_labels=2)
    model     = AutoModelForSequenceClassification.from_pretrained(
        pc.MODEL_NAME_ROBERTA, config=config
    ).to(pc.DEVICE)

    # ---- 优化器与损失函数 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=pc.LR, weight_decay=pc.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # ---- 训练循环 ----
    print("[训练] 开始 ...")
    for epoch in range(pc.EPOCHS):
        model.train()
        run_loss = 0.0
        run_acc  = 0.0
        cnt      = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(pc.DEVICE),
                attention_mask=batch["attention_mask"].to(pc.DEVICE),
            )
            logits = out.logits
            labels = batch["label"].to(pc.DEVICE)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs   = labels.size(0)
            acc  = (torch.argmax(logits, dim=-1) == labels).float().mean().item()
            run_loss += loss.item() * bs
            run_acc  += acc * bs
            cnt      += bs

            if (step + 1) % 50 == 0:
                print(f"  ep{epoch+1} step{step+1} loss={loss.item():.4f} acc={acc:.4f}")
    return model


