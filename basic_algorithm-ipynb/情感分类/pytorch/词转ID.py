import jieba
import database.read_chnsenticorphtlall as rchn
from tqdm import tqdm
import pickle
import json
from collections import Counter

labels,texts = rchn.read_data()

# 中文分词（使用jieba）
def chinese_tokenize(text):
    return list(jieba.cut(text.strip()))

# 构建词表
print("\n## 2. 词表构建")
tokenized_texts = [chinese_tokenize(review) for review in tqdm(texts, desc="分词")]
all_tokens = [token for tokens in tokenized_texts for token in tokens]
counter = Counter(all_tokens) #统计频次
#<PAD> 填充标记（Padding），[好评]  →  [好, 评, <PAD>, <PAD>, <PAD>, <PAD>, ...]，告诉模型："<PAD> 这里没有实际语义，别拿它当特征学"，训练时通常配合 attention_mask 或在 loss 里忽略 pad 位置
#<UNK> 未知标记（Unknown），测试/预测时，模型会遇到训练时没见过的词，统一映射成 <UNK>，模型学到的是"这里有个不认识的词"，而不是崩溃
vocab = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common()]# most_common 取出词频最高的多少个
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
print(f"词表大小: {len(vocab)} (含特殊标记)")

preprocessed = {
    'tokenized_texts': tokenized_texts,# 句子分成词 [多少句，一句多少词]
    'all_tokens': all_tokens, # 所有词放在一个list
    'vocab': vocab, #词列表（包括<PAD><UNK>）
    'word_to_idx': word_to_idx #词：id
}

with open('fenci.pkl', 'wb') as f:
    pickle.dump(preprocessed, f)


with open('fenci.pkl', 'rb') as f:
    preprocessed_pick = pickle.load(f)


with open('fenci.json', 'w', encoding='utf-8') as f:
    json.dump(preprocessed, f, ensure_ascii=False)


with open('fenci.json', 'r', encoding='utf-8') as f:
    preprocessed_json = json.load(f)

print(preprocessed_pick)
print(preprocessed_json)

def save_txt():

    with open('tokenized_texts.txt', 'w', encoding='utf-8') as f:
        for tokens in tokenized_texts:
            f.write(' '.join(tokens) + '\n')

    with open('all_tokens.txt', 'w', encoding='utf-8') as f:
        for token in all_tokens:
            f.write(' '.join(token) + '\n')

    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + '\n')

    # 3. word_to_idx：JSON 格式存（它是 dict）
    # import json
    # with open('word_to_idx.json', 'w', encoding='utf-8') as f:
    #     json.dump(word_to_idx, f, ensure_ascii=False)

def load_txt():
    tokenized_texts = []
    all_tokens = []
    vocab = []
    with open('tokenized_texts.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokenized_texts.append(line.strip().split())

    with open('all_tokens.txt', 'r', encoding='utf-8') as f:
        all_tokens = [line.strip() for line in f]

    with open('vocab.txt', 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]

    # # 3. word_to_idx
    # import json
    # with open('word_to_idx.json', 'r', encoding='utf-8') as f:
    #     word_to_idx = json.load(f)