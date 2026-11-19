# ---- 保存模型 ----
import os


def save_model(model,tokenizer,save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)  # 保存模型权重 + config
    tokenizer.save_pretrained(save_path)  # 保存分词器
    print(f"[保存] → {save_path}")

