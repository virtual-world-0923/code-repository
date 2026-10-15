import torch.nn as nn
import torch


# ================================
# 5. RNN模型定义
# ================================
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, pad_i):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_i)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,  # ★ 双向结构对情感分类至关重要
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 双向需×2
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, param_test):
        print(param_test)
        # x: [batch_size, seq_len] [批次，句长]
        #embedded: [batch, seq, embed_dim] [批次，句长，句子里每个词的向量]
        embedded = self.embedding(x)
        '''
            是 PyTorch 给 RNN 准备的"压缩数据格式"，不能直接在上面做运算，但 RNN 认识它，喂进去效率更高。
            让 RNN 跳过 <PAD> 位置、只计算真实 token​ 的工具

        lengths = [len(seq) for seq in x]
        lengths = torch.tensor(lengths, dtype=torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, # [batch, seq_len, embed_dim]
            lengths.cpu(), # 每条句子的真实长度,lengths 必须在 CPU 上
            batch_first=True, #和你的 embedded 维度一致
            enforce_sorted=False) #让 PyTorch 自动帮你按长度降序排。这里的顺序有可能变，所以最好在 collate_fn 里手动排序，labels 跟着排
        '''
        output, (hidden, _) = self.lstm(embedded)
        # 拼接双向LSTM最后时刻的隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return torch.sigmoid(self.fc(hidden)).squeeze()  # squeeze 去掉所有长度为 1 的维度


EMBED_DIM = 128
HIDDEN_DIM = 64


def backbone_network(VOCAB_SIZE, pad_i):
    model = SentimentRNN(
        vocab_size=VOCAB_SIZE,  # 词表大小，决定 Embedding 层有多少行。Embedding 层的行数，每个词一个向量
        embed_dim=EMBED_DIM,  # 每个词嵌入向量的维度。每个词用多少维的向量表示
        hidden_dim=HIDDEN_DIM,  # RNN 隐藏层维度。模型"记忆力"的大小
        num_layers=1,  # RNN 堆叠几层。2 层比 1 层能捕捉更复杂的语义，但训练更慢、更容易过拟合
        pad_i=pad_i # <PAD> 的值
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model
