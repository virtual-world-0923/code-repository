# BERT / RoBERTa 中文评论情感分类

## 环境版本（严格匹配）
- Python 3.13.14
- PyTorch 2.13.0+cu126
- TensorFlow 2.21.0
- Keras 3.15.0
- scikit-learn 1.9.0
- numpy 2.5.1
- pandas 3.0.3
- HuggingFace transformers 5.14.1

## 文件说明
| 文件 | 作用 |
|---|---|
| `bert_sentiment.py` | 主脚本，含三种训练方式（A/B/C），每行列有完整注释 |
| `predict.py` | 独立推理脚本，可命令行运行，也可 import 使用 |

## 三种训练方式（全部保留，通过开关切换）

### 方式 A —— 原生 PyTorch 手写训练循环
- 模型：`bert-base-chinese`
- 特点：完全手写前向/反向/优化器，适合学习原理
- 保存目录：`models\bert_sentiment`

### 方式 B —— HuggingFace Trainer API
- 模型：`bert-base-chinese`
- 特点：用 `Trainer` + `TrainingArguments` 封装训练循环，代码更简洁
- 保存目录：`models\bert_sentiment`（与 A 共用，建议先跑完 A 再跑 B，或改目录避免覆盖）

### 方式 C —— RoBERTa-wwm 模型
- 模型：`hfl/chinese-roberta-wwm-ext`（哈工大中文整词掩码预训练）
- 特点：中文任务通常比原生 BERT 高 1~3 个点
- 保存目录：`models\roberta_wwm_sentiment`

## 切换方式

打开 `bert_sentiment.py`，修改顶部这一行：
```python
RUN_MODE = "A"   # 改成 "B" 或 "C"
```
- `"A"` → 方式 A（手写循环 + bert-base-chinese）
- `"B"` → 方式 B（Trainer API + bert-base-chinese）
- `"C"` → 方式 C（手写循环 + chinese-roberta-wwm-ext）

## 快速开始

### 训练 + 保存
```bash
python bert_sentiment.py
```
训练完成后模型保存在对应目录：
```
models\bert_sentiment\       (方式 A/B)
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── vocab.txt
└── ...

models\roberta_wwm_sentiment\ (方式 C)
├── config.json
├── model.safetensors
└── ...
```

### 加载模型 + 预测一句话

#### 命令行方式
```bash
# 默认用 bert-base-chinese 模型
python predict.py "这家酒店服务态度很好，房间干净整洁"

# 指定用 RoBERTa-wwm 模型
python predict.py "房间又小又脏，非常失望" --model roberta
```

输出示例：
```
模型: bert
评论: 这家酒店服务态度很好，房间干净整洁
情感: 满意  (置信度: 0.9876)
```

#### Python 模块方式
```python
from predict import load_model, predict

# 加载 bert-base-chinese
tok, mdl = load_model(r"C:\Repositories\basic_algorithm\models\bert_sentiment")
label, conf = predict("价格贵得离谱，性价比太低了", tok, mdl)
print(label, conf)   # 不满意 0.97...

# 加载 chinese-roberta-wwm-ext
tok2, mdl2 = load_model(r"C:\Repositories\basic_algorithm\models\roberta_wwm_sentiment")
label2, conf2 = predict("位置不错，出行方便", tok2, mdl2)
print(label2, conf2)  # 满意 0.96...
```

## 关键实现说明（适配 Transformers 5.14.1）

- 使用 `AutoModelForSequenceClassification` + `num_labels=2`，自动挂分类头
- `TrainingArguments` 用 `eval_strategy="epoch"`（v5 参数名，非旧版 `evaluation_strategy`）
- 保存调用 `model.save_pretrained()`，**不传 `safe_serialization`**（v5 已移除该参数）
- 加载使用 `from_pretrained()`，默认 dtype=auto，自动匹配保存时的精度
- 推理时对 logits 做 `softmax` 得到置信度

## 数据格式
`ChnSentiCorp_htl_all.csv`：
```
label,review
1,酒店环境很好，服务也很周到
0,房间又小又脏，空调噪音很大
...
```
按 8:1:1 划分为 train / val / test。

## 三种方式对比
| 对比项 | 方式 A（手写） | 方式 B（Trainer） | 方式 C（RoBERTa-wwm） |
|---|---|---|---|
| 模型 | bert-base-chinese | bert-base-chinese | hfl/chinese-roberta-wwm-ext |
| 训练循环 | 手写 for loop | Trainer 封装 | 手写 for loop |
| 代码量 | 较多 | 较少 | 较多 |
| 适合场景 | 学习原理 | 工程快速迭代 | 追求中文效果 |
| 预期准确率 | ~0.90-0.93 | ~0.90-0.93 | ~0.92-0.95 |
