from keras import layers
import keras

embed = keras.Sequential(
    [
        # 可训练的词表 "整数 ID → 稠密向量"（学语义） (需要训练，反向更新参数)
        # 创建一个形状为 (input_dim, output_dim) 的权重矩阵
        # 每个 token id 对应矩阵中的一行
        layers.Embedding(
            input_dim=5000, #词汇表大小，也就是"最多认识多少个不同的字符/词"。
            output_dim=128, #每个 token 映射成多少维的向量。
            mask_zero=True,  # 把整数 0 当作 padding（填充），告诉后面的层"这个位置是无效的，别管它"。
            name="embedding",
        ),
    ]
)

# RNN 网络
model = keras.Sequential(
    [
        embed,
        # --- 双向 LSTM（RNN）---
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=True), #换成 SimpleRNN、GRU
            name="bi_lstm_1",
        ),
        layers.Bidirectional(
            layers.LSTM(32),
            name="bi_lstm_2",
        ),

        # --- 全连接分类头 ---
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),   # 二分类输出
    ],
    name="SentimentRNN",
)