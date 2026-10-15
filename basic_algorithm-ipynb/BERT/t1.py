import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import BertTokenizer, BertModel
import torch

# 加载中文BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

text = "如何使用BERT模型"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state: [batch_size, seq_len, 768] 每个token的向量
# pooler_output: [batch_size, 768] 整个句子的向量（经过[CLS]处理）
sentence_embedding = outputs.pooler_output
print(sentence_embedding.shape)  # torch.Size([1, 768])