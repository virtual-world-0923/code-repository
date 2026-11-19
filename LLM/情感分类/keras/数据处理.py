import numpy as np
import database.read_chnsenticorphtlall as rchn
from sklearn.model_selection import train_test_split
from keras import layers
import keras


labels,texts = rchn.read_data()

print(f"有效样本数: {len(texts)}")
print(f"正样本(满意): {int(labels.sum())}  |  负样本(不满意): {int(len(labels) - labels.sum())}")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels          # 保持正负样本比例一致
)
print(f"训练集: {len(X_train)} 条  |  测试集: {len(X_test)} 条")
print("-" * 60)

# 分词 + 编码 (不参与训练，反向更新参数)
vectorize_layer = layers.TextVectorization(
    max_tokens=5000, #词汇表最多保留多少个不同的字符。
    output_mode="int", #整数 ID 序列
    output_sequence_length=200, #"统一裁成 200 长度，短的补0，长的砍尾"
    split="character",  # 逐字切分，适合中文
)

vectorize_layer.adapt(X_train) #从训练数据中"学习"词汇表，扫描数据，建立"词 → ID"的映射表

print(f"词表大小: {vectorize_layer.vocabulary_size()}")
print("-" * 60)

X_train_int = vectorize_layer(X_train)
X_test_int  = vectorize_layer(X_test)
