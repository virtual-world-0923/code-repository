import numpy as np
from torch.nn.utils.rnn import pad_sequence

import database.read_chnsenticorphtlall as rchn
from sklearn.model_selection import train_test_split
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

labels, texts = rchn.read_data()

print(f"有效样本数: {len(texts)}")
print(f"正样本(满意): {int(labels.sum())}  |  负样本(不满意): {int(len(labels) - labels.sum())}")
print("-" * 60)


# 中文分词（使用jieba）
def chinese_tokenize(text):
    return list(jieba.cut(text.strip()))


# 构建词表
fenci = {
    'tokenized_texts': None,  # 句子分成词 [多少句，一句多少词]
    'all_tokens': None,  # 所有词放在一个list
    'vocab': None,  # 词列表（包括<PAD><UNK>） PAD:填充标记  UNK:未知标记
    'word_to_idx': None  # 词：id
}
with open('fenci.pkl', 'rb') as f:
    fenci = pickle.load(f)


def text_to_indexes(text):
    fen = chinese_tokenize(text)
    word_to_idx = fenci["word_to_idx"]
    wi = []
    for ci in fen:
        wi.append(word_to_idx.get(ci, word_to_idx['<UNK>']))
    return wi


X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # 保持正负样本比例一致
)
print(f"训练集: {len(X_train)} 条  |  测试集: {len(X_test)} 条")
print("-" * 60)


# ================================
# 4. 自定义Dataset与DataLoader
# ================================
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_idx = text_to_indexes(self.sequences[idx])
        return torch.tensor(seq_idx, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)


# 自定义批处理时 padding，输入长度长度统一
# 把输入长度不一的文本变成长度一致,按照当前批次最大句长词量来定
def collate_fn(batch):  # 你定义了 collate_fn 返回什么，当从 DataLoader 取值时就给你什么
    sequences, labels = zip(*batch)
    word_to_idx = fenci["word_to_idx"]
    padded = pad_sequence( #自动找出这批最长的元素，返回补齐后的矩阵
        sequences, #必须是 tensor，里面每条元素长度各不相同
        batch_first=True, #输出的形状是 [batch, seq_len] 而不是 [seq_len, batch]
        padding_value=word_to_idx["<PAD>"] #用什么数字来补齐
    )
    labels = torch.tensor(labels, dtype=torch.float)
    # lengths = [len(seq) for seq in sequences]
    # max_len = max(lengths)
    # word_to_idx = fenci["word_to_idx"]
    # if len(sequences) < max_len:
    #     sequences += [word_to_idx["<PAD>"]] * (max_len - len(sequences))
    return padded, labels


train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
