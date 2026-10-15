"""
================================================================================
predict.py  —— 独立推理脚本（逐行完整注释版）
================================================================================
功能：
    加载训练好的 BERT / RoBERTa 情感分类模型，对输入的一句话做预测。
    可作为命令行工具运行，也可作为模块被其他 Python 文件 import。

用法（命令行）：
    python predict.py "这家酒店服务态度很好"
    python predict.py "房间又小又脏" --model roberta

默认加载 bert-base-chinese 的训练结果，可用 --model 参数切换：
    --model bert     → 加载方式 A/B 保存的 bert-base-chinese 模型
    --model roberta  → 加载方式 C  保存的 chinese-roberta-wwm-ext 模型
================================================================================
"""

# ==============================================================================
# 1. 导入依赖库
# ==============================================================================
import argparse   # 命令行参数解析库：让用户通过 --xxx 传参
import torch      # PyTorch：张量计算与模型推理
import sys        # 系统库：用于退出程序
import project_config as pc
# 从 transformers 导入自动加载类（v5.14.1 支持）
from transformers import (
    AutoTokenizer,                 # 自动选择对应的分词器
    AutoConfig,                    # 自动加载模型配置
    AutoModelForSequenceClassification,  # 自动加载带分类头的模型
)

# ==============================================================================
# 3. 加载模型函数
# ==============================================================================
def load_model(model_dir):
    """
    从指定目录加载分词器与模型，并切换为评估模式。
    参数:
        model_dir (str): 保存模型的目录路径
    返回:
        (tokenizer, model): 分词器与模型元组
    """
    # 从本地目录加载分词器（读取 tokenizer_config.json / vocab.txt）
    tok = AutoTokenizer.from_pretrained(model_dir)
    # 从本地目录加载模型配置（读取 config.json，含 num_labels 等）
    cfg = AutoConfig.from_pretrained(model_dir)
    # 从本地目录加载模型权重（读取 model.safetensors）
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_dir, config=cfg
    ).to(pc.DEVICE)   # 搬到目标设备（GPU/CPU）
    mdl.eval()     # 切换为评估模式（关闭 dropout、固定 BN）
    return tok, mdl

# ==============================================================================
# 4. 单句预测函数
# ==============================================================================
def predict(text, tokenizer, model):
    """
    对一句话做情感分类预测。
    参数:
        text      (str): 待预测的中文评论
        tokenizer (AutoTokenizer): 分词器
        model     (nn.Module): 已加载的模型
    返回:
        (str, float): ("满意"/"不满意", 置信度 0~1)
    """
    # 使用分词器编码文本：
    #   max_length  - 最大长度，超过截断
    #   padding     - 不足补 pad 到 max_length
    #   truncation  - 启用截断
    #   return_tensors - 返回 PyTorch 张量
    enc = tokenizer(
        text,
        max_length=pc.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(pc.DEVICE)  # 搬到目标设备

    # 不计算梯度（推理阶段不需要反向传播）
    with torch.no_grad():
        # 前向传播，**enc 自动解包为 input_ids 等参数
        logits = model(**enc).logits          # [1, num_labels]
        # softmax 将 logits 转为概率分布
        probs  = torch.softmax(logits, dim=-1)
        # 取概率最大的类别下标
        pred   = torch.argmax(probs, dim=-1).item()
        # 取出该类别的置信度
        conf   = probs[0][pred].item()

    # 将数字标签映射为中文
    return ("满意" if pred == 1 else "不满意"), conf

m1 = pc.SAVE_DIR_BERT
m2 = pc.SAVE_TRAINER_BERT
m3 = pc.SAVE_DIR_ROBERTA
# 加载模型
tok, mdl = load_model(m1)

EXAMPLES = [
    "这家酒店服务态度很好，房间干净整洁，下次还会再来。",
    "房间又小又脏，空调噪音很大，简直没法睡觉，非常失望。",
    "位置不错，离地铁站很近，出行方便。",
    "价格贵得离谱，性价比太低了。",
]
# 预测
for text in EXAMPLES:
    label, conf = predict(text, tok, mdl)
    print(label, conf)

