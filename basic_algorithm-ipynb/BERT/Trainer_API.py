import numpy as np             # 数值计算库：用于设置随机种子
import project_config as pc
# 从 transformers 库导入所需的类（v5.14.1 均支持）
from transformers import (
    AutoTokenizer,                        # 自动选择对应的分词器（tokenizer）
    AutoConfig,                           # 自动加载模型配置文件
    AutoModelForSequenceClassification,   # 自动加载带分类头的序列分类模型
    Trainer,                              # 高层训练器：封装训练循环、评估、保存
    TrainingArguments,                    # 训练参数配置类：学习率、batch、epoch 等
)
import evaluate                          # HuggingFace evaluate 库：加载评估指标（accuracy 等）



def run_mode_B_trainer(train_ds,val_ds):
    """
    方式 B：使用 HuggingFace Trainer API 微调 bert-base-chinese。
    Trainer 内部自动处理：训练循环、梯度累积、混合精度、日志、评估、保存。
    """
    print("\n" + "=" * 70)
    print("【方式 B】HuggingFace Trainer API  |  bert-base-chinese")
    print("=" * 70)

    config    = AutoConfig.from_pretrained(pc.MODEL_NAME_BERT, num_labels=2)
    model     = AutoModelForSequenceClassification.from_pretrained(
        pc.MODEL_NAME_BERT, config=config
    ).to(pc.DEVICE)

    # ---- 加载评估指标 ----
    # evaluate 库加载 accuracy 指标
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        """
        给 Trainer 用的评估函数。
        参数:
            eval_pred: 包含 (logits, labels) 的元组
        返回:
            dict: 指标名 -> 数值
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)  # 转预测类别
        return accuracy_metric.compute(predictions=preds, references=labels)

    # ---- 训练参数配置 ----
    # TrainingArguments：Trainer 的所有超参与开关都在这里
    training_args = TrainingArguments(
        output_dir=pc.SAVE_DIR_BERT,           # 检查点与日志输出目录
        num_train_epochs=pc.EPOCHS,            # 训练轮数
        per_device_train_batch_size=pc.BATCH_SIZE,  # 每设备训练 batch
        per_device_eval_batch_size=pc.BATCH_SIZE,   # 每设备评估 batch
        learning_rate=pc.LR,                  # 初始学习率
        weight_decay=pc.WEIGHT_DECAY,         # 权重衰减
        warmup_steps=pc.WARMUP_STEPS,         # 预热比例
        eval_strategy="epoch",             # v5 参数：每 epoch 结束评估一次
        save_strategy="epoch",             # 每 epoch 保存一次检查点
        load_best_model_at_end=True,       # 训练结束加载验证集最优模型
        metric_for_best_model="accuracy",  # 以 accuracy 为最优标准
        greater_is_better=True,             # accuracy 越大越好
        logging_strategy="steps",          # 按步数记录日志
        logging_steps=50,                  # 每 50 步打印一次
        seed=pc.SEED,                         # 随机种子
        report_to="none",                  # 不向外部平台（wandb 等）上报
    )

    # ---- 构造 Trainer ----
    trainer = Trainer(
        model=model,                  # 要训练的模型
        args=training_args,           # 训练参数
        train_dataset=train_ds,       # 训练集
        eval_dataset=val_ds,         # 验证集
        compute_metrics=compute_metrics,  # 评估指标函数
        # tokenizer 在 v5 中通过 args 自动关联，无需单独传
    )

    # ---- 开始训练 ----
    print("[训练] Trainer 开始 ...")
    trainer.train()

    # ---- 测试集评估 ----
    test_results = trainer.evaluate(val_ds)
    print(f"\n[测试] {test_results}")

    # ---- 保存最终模型与分词器 ----
    # trainer.save_model 只保存模型；分词器需单独保存
    trainer.save_model(pc.SAVE_DIR_BERT)
    print(f"[保存] → {pc.SAVE_DIR_BERT}")
    return model

