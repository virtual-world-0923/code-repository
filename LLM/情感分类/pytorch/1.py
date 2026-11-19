import os
import numpy as np
import pandas as pd
import jieba
import pickle
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# 设备配置
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 1. 加载数据
# ============================================================
data_path = r'C:\Repositories\basic_algorithm\database\ChnSentiCorp_htl_all.csv'
df = pd.read_csv(data_path)
df = df.dropna(subset=['review', 'label'])
df['review'] = df['review'].astype(str)
df['label']  = df['label'].astype(int)
print(f"数据集大小: {len(df)}")
print(df.head())
print(f"\n标签分布:\n{df['label'].value_counts()}")

# ============================================================
# 2. 数据预处理 — jieba 分词 + 构建词汇表
# ============================================================
print("\n正在进行 jieba 分词...")

def tokenize(text):
    """jieba 中文分词"""
    words = jieba.lcut(text.strip())
    return [w for w in words if w.strip()]

# 分词
all_tokens_list = [tokenize(review) for review in df['review'].tolist()]
labels = df['label'].values

# 按词频构建词汇表
MAX_FEATURES = 20000
MAX_LEN = 200

word_counter = Counter()
for tokens in all_tokens_list:
    word_counter.update(tokens)

# 0=PAD, 1=UNK
word_index = {}
for i, (word, _) in enumerate(word_counter.most_common(MAX_FEATURES - 2)):
    word_index[word] = i + 2

vocab_size = len(word_index) + 2
print(f"词汇表大小: {vocab_size}")

# 文本 → 定长整数序列
def encode_tokens(tokens, wi, ml):
    seq = [wi.get(t, 1) for t in tokens]   # UNK=1
    if len(seq) < ml:
        seq = seq + [0] * (ml - len(seq))  # PAD=0
    else:
        seq = seq[:ml]
    return seq

encoded_sequences = [encode_tokens(t, word_index, MAX_LEN) for t in all_tokens_list]
X = np.array(encoded_sequences, dtype=np.int64)
y = labels.astype(np.float32)
print(f"序列矩阵形状: {X.shape}")

# ============================================================
# 3. 划分训练集和测试集 (8:2)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")

# ============================================================
# 4. PyTorch Dataset 和 DataLoader
# ============================================================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = torch.from_numpy(texts).long()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

BATCH_SIZE = 128

train_dataset = SentimentDataset(X_train, y_train)
test_dataset  = SentimentDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5. 搭建 RNN 网络 (Embedding → LSTM → FC)
# ============================================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=1, dropout=0.3):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)   # 双向 → hidden_dim*2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embeds = self.embedding(x)              # (batch, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embeds)  # lstm_out: (batch, seq_len, hidden*2)

        # 取最后一层正向和反向的隐藏状态拼接
        # h_n shape: (num_layers*2, batch, hidden_dim)
        forward_hidden  = h_n[-2]   # 最后一层正向
        backward_hidden = h_n[-1]   # 最后一层反向
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)  # (batch, hidden*2)

        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)                     # (batch, 1)
        return out

EMBEDDING_DIM = 128
HIDDEN_DIM    = 128

model = SentimentRNN(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

print("\n模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {total_params:,}")

# ============================================================
# 6. 训练模型
# ============================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5

print("\n开始训练...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_texts, batch_labels in train_loader:
        batch_texts  = batch_texts.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_texts).squeeze(1)   # (batch,)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_texts.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == batch_labels).sum().item()
        total   += batch_labels.size(0)

    avg_loss = total_loss / total
    train_acc = correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}  Train Acc: {train_acc:.4f}")

# ============================================================
# 7. 打印训练集和测试集精度
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_texts, batch_labels in loader:
            batch_texts  = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_texts).squeeze(1)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == batch_labels).sum().item()
            total   += batch_labels.size(0)
    return correct / total

train_acc = evaluate(model, train_loader)
test_acc  = evaluate(model, test_loader)

print(f"\n{'=' * 40}")
print(f"训练集精度 (Accuracy): {train_acc:.4f}")
print(f"测试集精度 (Accuracy): {test_acc:.4f}")
print(f"{'=' * 40}")

# ============================================================
# 8. 保存模型 + 词汇表
# ============================================================
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': HIDDEN_DIM,
    'max_len': MAX_LEN,
}, 'sentiment_rnn_model.pth')

with open('word_index.pkl', 'wb') as f:
    pickle.dump({'word_index': word_index, 'max_len': MAX_LEN}, f)

print("\n模型已保存: sentiment_rnn_model.pth")
print("词表已保存: word_index.pkl")

# ============================================================
# 9. 加载模型，预测新评论
# ============================================================
print("\n加载已保存的模型进行预测...")

# 加载 checkpoint
checkpoint = torch.load('sentiment_rnn_model.pth', map_location=device, weights_only=False)

# 重建模型结构
loaded_model = SentimentRNN(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_dim=checkpoint['hidden_dim']
).to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# 加载词汇表
with open('word_index.pkl', 'rb') as f:
    saved = pickle.load(f)
loaded_wi  = saved['word_index']
loaded_len = saved['max_len']


def predict_sentiment(text):
    """预测单条评论"""
    tokens = tokenize(text)
    seq = encode_tokens(tokens, loaded_wi, loaded_len)
    tensor_input = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        output = loaded_model(tensor_input).squeeze(1)
        score = torch.sigmoid(output).item()
    label = "满意" if score >= 0.5 else "不满意"
    return label, score


test_texts = [
    "这家酒店环境非常好，服务态度一流，下次还会再来！",
    "房间很脏，隔音效果差，服务态度恶劣，不推荐。",
    "位置不错，离地铁站很近，出行很方便。",
    "床单有污渍，空调噪音大，严重影响睡眠。"
]

print("\n预测结果:")
print("-" * 60)
for text in test_texts:
    sentiment, score = predict_sentiment(text)
    print(f"评论: {text}")
    print(f"结果: {sentiment}（置信度: {score:.4f}）")
    print("-" * 60)