import os
# os.environ["HF_HUB_OFFLINE"] = "1" #Hugging Face Hub 禁止联网
# os.environ["TRANSFORMERS_OFFLINE"] = "1" #Transformers 禁止联网
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #设置国内镜像
import project_config as pc
import runpy
from 保存模型 import save_model
from 模型评估 import evaluate_model

# 从 transformers 库导入所需的类（v5.14.1 均支持）
from transformers import (
    AutoTokenizer,                        # 自动选择对应的分词器（tokenizer）
    AutoConfig,                           # 自动加载模型配置文件
    AutoModelForSequenceClassification,   # 自动加载带分类头的序列分类模型
)

result = runpy.run_path("数据处理.py")
func = result["get_data_loader"]
train_loader, val_loader,train_ds,val_ds, tokenizer= func()

def evaluate_save_model(model,save_path):
    v_loss, v_acc = evaluate_model(model, val_loader)
    print(f"\n[验证] loss={v_loss:.4f}  accuracy={v_acc:.4f}")

    save_model(model, tokenizer, save_path)

def dir_bert():
    result = runpy.run_path("手写训练循环.py")
    func = result["run_mode_A_handcraft"]
    model = func(train_loader)

    evaluate_save_model(model,pc.SAVE_DIR_BERT)

def trainer_bert():
    result = runpy.run_path("Trainer_API.py")
    func = result["run_mode_B_trainer"]
    model = func(train_ds, val_ds)

    evaluate_save_model(model,pc.SAVE_TRAINER_BERT)

def dir_roberta():
    result = runpy.run_path("RoBERTa-wwm_模型.py")
    func = result["run_mode_C_roberta"]
    model = func(train_loader)

    evaluate_save_model(model,pc.SAVE_DIR_ROBERTA)

dir_bert()
trainer_bert()
dir_roberta()

# # ---- 加载分词器与模型 ----
# # AutoTokenizer：根据模型名自动选择 BertTokenizer
# tokenizer = AutoTokenizer.from_pretrained(pc.MODEL_NAME_BERT)
# # AutoConfig：加载默认配置并把分类头设为 2 类
# config = AutoConfig.from_pretrained(pc.MODEL_NAME_BERT, num_labels=2)
# # AutoModelForSequenceClassification：加载带分类头的 BERT
# model = AutoModelForSequenceClassification.from_pretrained(
#     pc.MODEL_NAME_BERT, config=config
# ).to(pc.DEVICE)


