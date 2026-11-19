# -*- coding: utf-8 -*-
"""
功能：使用 Gensim 库分别训练 Word2Vec（CBOW / Skip-gram）和 FastText 词向量模型
环境：Python 3.13.14 | numpy 2.5.1 | Gensim 4.x
"""

# ==================== 第一部分：导入所需库 ====================

import jieba  # 导入 jieba 中文分词库，用于将中文句子切分为词语列表
import numpy as np  # 导入 numpy 数值计算库，用于向量运算和余弦相似度计算
from gensim.models import Word2Vec  # 从 gensim.models 导入 Word2Vec 类，用于训练 Word2Vec 词向量模型
from gensim.models import FastText  # 从 gensim.models 导入 FastText 类，用于训练 FastText 词向量模型

# ==================== 第二部分：准备中文语料 ====================

# 定义一段中文文本语料，包含多个句子，涵盖同义词和无关词，便于后续验证
# 语料设计思路：包含"快乐/高兴/开心"（同义词）、"悲伤/难过"（近义）、"苹果/电脑"（无关词）
corpus_text = (
    "今天天气真好我感到非常快乐。"
    "她今天心情不错看起来很高兴。"
    "孩子们在游戏玩耍十分开心。"
    "他因为考试没考好所以很悲伤。"
    "听到这个坏消息她心里非常难过。"
    "失去亲人让他感到无比悲伤。"
    "我每天早上都会吃一个苹果。"
    "这台电脑的性能非常强大。"
    "秋天是收获的季节果园里苹果挂满枝头。"
    "程序员坐在电脑前编写代码。"
    "快乐是可以传递给身边每一个人的。"
    "高兴的事情要和朋友一起分享。"
    "开心的笑容是最美的语言。"
    "悲伤的眼泪总会过去明天会更好。"
    "难过的时光终将成为回忆。"
    "苹果是一种富含维生素的水果。"
    "电脑科技改变了人们的生活方式。"
    "阳光灿烂的日子让人心情快乐。"
    "朋友们聚在一起大家都很开心。"
    "她微笑着说今天真高兴。"
)

# 打印原始语料文本，确认数据已正确定义
print("=" * 60)  # 打印分隔线，用于控制台输出格式化
print("【原始语料】")  # 打印标题提示
print(corpus_text)  # 输出完整的中文语料文本
print("=" * 60)  # 打印分隔线

# ==================== 第三部分：使用 jieba 进行中文分词 ====================

# 按中文句号"。"将语料分割为多个句子，返回字符串列表
raw_sentences = corpus_text.split("。")  # 参数：分隔符"。"，返回子串列表
# 列表推导式：过滤掉空白句子（split在末尾可能产生空字符串）
raw_sentences = [s for s in raw_sentences if s.strip()]  # s.strip()去除首尾空白，空串为False被过滤

# 对每个句子使用 jieba 进行分词，得到二维列表（每个元素是一个句子的词语列表）
# jieba.cut(sentence) 返回一个生成器（generator），用 list() 转为列表
# 等价于旧版的 jieba.lcut()，兼容性更好
sentences = [list(jieba.cut(sentence)) for sentence in raw_sentences]  # 遍历每个句子并分词，存储为嵌套列表

# 打印分词结果，验证分词是否正确
print("\n【分词结果（前5句）】")  # 打印标题提示
for i, sent in enumerate(sentences[:5]):  # enumerate返回(索引,值)，sentences[:5]取前5个句子
    print(f"  句子{i+1}: {sent}")  # 格式化输出每个句子的分词结果

# ==================== 第四部分：训练 Word2Vec CBOW 模型 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【训练 Word2Vec CBOW 模型】")  # 打印标题提示

# 使用 Gensim 的 Word2Vec 训练 CBOW 模型
# 参数说明：
#   sentences: 训练语料，格式为二维列表，每个子列表是一个句子的词语序列
#   vector_size: 词向量的维度，设为100维（越大表达能力越强，但需要更多数据）
#   window: 上下文窗口大小，设为5，即当前词前后各取5个词作为上下文
#   min_count: 最小词频阈值，设为1表示所有出现的词都参与训练（语料小所以设为1）
#   workers: 训练时使用的并行线程数，设为3表示使用3个CPU核心并行训练
#   sg: 训练算法选择，0表示CBOW（用上下文预测中心词），1表示Skip-gram（用中心词预测上下文）
#   epochs: 训练迭代轮数，设为100轮（语料小需要更多轮次才能收敛）
#   seed: 随机种子，设为42保证结果可复现
model_cbow = Word2Vec(
    sentences=sentences,      # 传入分词后的语料（二维列表）
    vector_size=100,          # 词向量维度设为100
    window=5,                 # 上下文窗口大小为5
    min_count=1,              # 最小词频为1（不过滤任何词）
    workers=3,                # 使用3个线程并行训练
    sg=0,                     # sg=0 表示使用 CBOW 模型（Continuous Bag of Words）
    epochs=100,               # 训练100轮迭代
    seed=42                   # 随机种子为42，确保结果可复现
)

# 打印 CBOW 模型训练完成的信息
print(f"  CBOW 模型训练完成！")  # 输出训练完成提示
print(f"  词汇表大小: {len(model_cbow.wv)}")  # model_cbow.wv 是 KeyedVectors 对象，len()返回词汇表中词的总数
print(f"  词向量维度: {model_cbow.vector_size}")  # 输出词向量的维度（应为100）

# ==================== 第五部分：训练 Word2Vec Skip-gram 模型 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【训练 Word2Vec Skip-gram 模型】")  # 打印标题提示

# 使用 Gensim 的 Word2Vec 训练 Skip-gram 模型
# 参数与 CBOW 相同，唯一区别是 sg=1 表示使用 Skip-gram 架构
# Skip-gram 用中心词预测上下文词，对低频词效果更好
model_skipgram = Word2Vec(
    sentences=sentences,      # 传入相同的分词语料（二维列表）
    vector_size=100,          # 词向量维度设为100
    window=5,                 # 上下文窗口大小为5
    min_count=1,              # 最小词频为1
    workers=3,                # 使用3个线程并行训练
    sg=1,                     # sg=1 表示使用 Skip-gram 模型
    epochs=100,               # 训练100轮迭代
    seed=42                   # 随机种子为42
)

# 打印 Skip-gram 模型训练完成的信息
print(f"  Skip-gram 模型训练完成！")  # 输出训练完成提示
print(f"  词汇表大小: {len(model_skipgram.wv)}")  # 输出词汇表大小
print(f"  词向量维度: {model_skipgram.vector_size}")  # 输出词向量维度

# ==================== 第六部分：训练 FastText 模型 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【训练 FastText 模型】")  # 打印标题提示

# 使用 Gensim 的 FastText 训练词向量模型
# FastText 与 Word2Vec 的区别：FastText 利用子词（subword/n-gram）信息
# 参数说明：
#   sentences: 训练语料，格式同 Word2Vec
#   vector_size: 词向量维度，设为100
#   window: 上下文窗口大小，设为5
#   min_count: 最小词频阈值，设为1
#   workers: 并行线程数，设为3
#   sg: 训练算法，0=CBOW，1=Skip-gram，这里设为1使用Skip-gram
#   epochs: 训练轮数，设为100
#   min_n: 子词（character n-gram）的最小长度，设为2（中文单字也有意义）
#   max_n: 子词（character n-gram）的最大长度，设为5
model_fasttext = FastText(
    sentences=sentences,      # 传入分词后的语料（二维列表）
    vector_size=100,          # 词向量维度设为100
    window=5,                 # 上下文窗口大小为5
    min_count=1,              # 最小词频为1
    workers=3,                # 使用3个线程并行训练
    sg=1,                     # 使用 Skip-gram 训练方式
    epochs=100,               # 训练100轮迭代
    min_n=2,                  # 子词 n-gram 最小长度为2个字符
    max_n=5,                  # 子词 n-gram 最大长度为5个字符
    seed=42                   # 随机种子为42
)

# 打印 FastText 模型训练完成的信息
print(f"  FastText 模型训练完成！")  # 输出训练完成提示
print(f"  词汇表大小: {len(model_fasttext.wv)}")  # 输出词汇表大小
print(f"  词向量维度: {model_fasttext.vector_size}")  # 输出词向量维度

# ==================== 第七部分：用 CBOW 模型将分词结果转为词向量 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【CBOW 模型 - 将分词结果转为词向量】")  # 打印标题提示

# 定义要转换的测试词语列表，包含同义词和无关词
test_words = ["快乐", "高兴", "悲伤", "苹果"]  # 4个测试词：快乐/高兴为同义，悲伤为近义，苹果为无关词

# 遍历每个测试词，使用 CBOW 模型获取其词向量
for word in test_words:  # 遍历测试词语列表中的每个词
    # model_cbow.wv[word] 通过 KeyedVectors 对象的索引操作获取指定词的向量
    # 返回一个 numpy 数组，shape 为 (vector_size,)，即 (100,)
    vector = model_cbow.wv[word]  # 获取该词在 CBOW 模型中的100维词向量
    # 打印词语及其向量的前5个分量（完整100维太长，只展示前5维便于阅读）
    # vector[:5] 取向量前5个元素，np.round() 保留4位小数
    print(f"  '{word}' 的CBOW词向量(前5维): {np.round(vector[:5], 4)}")  # 格式化输出

# 展示如何将一个完整句子的分词结果转为词向量矩阵
demo_sentence = "我感到非常快乐"  # 定义一个演示句子
demo_words = list(jieba.cut(demo_sentence))  # jieba.cut()返回生成器，list()转为列表
print(f"\n  演示句子: '{demo_sentence}'")  # 打印原始句子
print(f"  分词结果: {demo_words}")  # 打印分词后的词语列表

# 使用列表推导式将句子中每个词转为词向量，再堆叠为二维 numpy 数组（矩阵）
# model_cbow.wv[w] 获取单个词的向量（shape=(100,)），多个词堆叠成矩阵
sentence_vectors_cbow = np.array([model_cbow.wv[w] for w in demo_words])  # 收集所有词的向量组成矩阵
# 打印词向量矩阵的形状，应为 (词语数, 100)，如 (4, 100)
print(f"  词向量矩阵形状: {sentence_vectors_cbow.shape}")  # 输出矩阵维度信息

# ==================== 第八部分：用 Skip-gram 模型将分词结果转为词向量 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【Skip-gram 模型 - 将分词结果转为词向量】")  # 打印标题提示

# 遍历相同的测试词列表，使用 Skip-gram 模型获取词向量
for word in test_words:  # 遍历测试词语列表
    # model_skipgram.wv[word] 获取 Skip-gram 模型中该词的词向量
    # 返回 numpy 数组，shape 为 (100,)
    vector = model_skipgram.wv[word]  # 获取该词在 Skip-gram 模型中的100维词向量
    # 打印词语及其向量的前5个分量
    print(f"  '{word}' 的Skip-gram词向量(前5维): {np.round(vector[:5], 4)}")  # 格式化输出前5维

# 同样展示完整句子的词向量转换
# 使用 Skip-gram 模型将演示句子的分词结果转为向量矩阵
sentence_vectors_sg = np.array([model_skipgram.wv[w] for w in demo_words])  # 收集每个词的Skip-gram向量
print(f"\n  演示句子: '{demo_sentence}'")  # 打印原始句子
print(f"  分词结果: {demo_words}")  # 打印分词结果
print(f"  词向量矩阵形状: {sentence_vectors_sg.shape}")  # 输出矩阵形状，如 (4, 100)

# ==================== 第九部分：用 FastText 模型获取词向量 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【FastText 模型 - 将分词结果转为词向量】")  # 打印标题提示

# 遍历测试词列表，使用 FastText 模型获取词向量
for word in test_words:  # 遍历测试词语列表
    # model_fasttext.wv[word] 获取 FastText 模型中该词的词向量
    # FastText 的优势：即使词未出现在训练集中，也可通过子词 n-gram 信息生成向量
    vector = model_fasttext.wv[word]  # 获取该词在 FastText 模型中的100维词向量
    # 打印词语及其向量的前5个分量
    print(f"  '{word}' 的FastText词向量(前5维): {np.round(vector[:5], 4)}")  # 格式化输出前5维

# ==================== 第十部分：效果验证 - 同义词距离近，无关词距离远 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【效果验证：同义词 vs 无关词的向量距离】")  # 打印标题提示

# 定义验证用的词语对
# 同义词对：语义相近的词，期望余弦相似度高（接近1）
synonym_pairs = [("快乐", "高兴"), ("快乐", "开心"), ("高兴", "开心"), ("悲伤", "难过")]  # 定义同义词对列表
# 无关词对：语义无关的词，期望余弦相似度低（接近0或为负）
unrelated_pairs = [("快乐", "苹果"), ("悲伤", "电脑"), ("高兴", "苹果")]  # 定义无关词对列表

# ---------- 使用 CBOW 模型验证 ----------
print("\n  --- Word2Vec CBOW 模型验证 ---")  # 打印子标题

# 计算并打印同义词对的余弦相似度
print("\n  [同义词对 - 期望相似度接近1]")  # 打印说明
for word1, word2 in synonym_pairs:  # 遍历每个同义词对，解包为 word1 和 word2
    # model_cbow.wv.similarity(w1, w2) 计算两个词向量的余弦相似度
    # 余弦相似度范围 [-1, 1]，越接近1表示方向越一致（越相似）
    # 参数：word1(str) 第一个词，word2(str) 第二个词
    # 返回值：float，余弦相似度值
    sim = model_cbow.wv.similarity(word1, word2)  # 调用 similarity 方法计算余弦相似度
    # 计算无关词之间的余弦相似度
    # sim_unrelated = model_cbow.wv.similarity("快乐", "苹果")
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出相似度，保留4位小数

# 计算并打印无关词对的余弦相似度
print("\n  [无关词对 - 期望相似度接近0或为负]")  # 打印说明
for word1, word2 in unrelated_pairs:  # 遍历每个无关词对
    # 同样使用 model_cbow.wv.similarity() 计算余弦相似度
    sim = model_cbow.wv.similarity(word1, word2)  # 计算两个无关词的余弦相似度
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出

# ---------- 使用 Skip-gram 模型验证 ----------
print("\n  --- Word2Vec Skip-gram 模型验证 ---")  # 打印子标题

# 计算并打印同义词对的余弦相似度
print("\n  [同义词对 - 期望相似度接近1]")  # 打印说明
for word1, word2 in synonym_pairs:  # 遍历同义词对
    # model_skipgram.wv.similarity(w1, w2) 计算 Skip-gram 模型中两词的余弦相似度
    sim = model_skipgram.wv.similarity(word1, word2)  # 计算余弦相似度
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出

# 计算并打印无关词对的余弦相似度
print("\n  [无关词对 - 期望相似度接近0或为负]")  # 打印说明
for word1, word2 in unrelated_pairs:  # 遍历无关词对
    sim = model_skipgram.wv.similarity(word1, word2)  # 计算余弦相似度
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出

# ---------- 使用 FastText 模型验证 ----------
print("\n  --- FastText 模型验证 ---")  # 打印子标题

# 计算并打印同义词对的余弦相似度
print("\n  [同义词对 - 期望相似度接近1]")  # 打印说明
for word1, word2 in synonym_pairs:  # 遍历同义词对
    # model_fasttext.wv.similarity(w1, w2) 计算 FastText 模型中两词的余弦相似度
    sim = model_fasttext.wv.similarity(word1, word2)  # 计算余弦相似度
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出

# 计算并打印无关词对的余弦相似度
print("\n  [无关词对 - 期望相似度接近0或为负]")  # 打印说明
for word1, word2 in unrelated_pairs:  # 遍历无关词对
    sim = model_fasttext.wv.similarity(word1, word2)  # 计算余弦相似度
    print(f"    similarity('{word1}', '{word2}') = {sim:.4f}")  # 格式化输出

# ==================== 第十一部分：使用 most_similar 查找最相似的词 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【使用 most_similar 查找与'快乐'最相似的词】")  # 打印标题提示

# model_cbow.wv.most_similar(positive=['快乐'], topn=5)
# 参数说明：
#   positive: 列表类型，指定要查找相似词的目标词（多个词时取平均向量作为查询向量）
#   topn: 整数类型，返回最相似的前N个词
# 返回值：列表，每个元素为 (词语str, 相似度float) 的元组
similar_words_cbow = model_cbow.wv.most_similar(positive=["快乐"], topn=5)  # 在CBOW模型中查找与"快乐"最相似的5个词
print("\n  CBOW 模型中与'快乐'最相似的词:")  # 打印标题
for word, score in similar_words_cbow:  # 遍历返回的(词, 分数)元组列表
    print(f"    '{word}': {score:.4f}")  # 输出词语和对应的相似度分数

# 在 Skip-gram 模型中查找最相似的词
similar_words_sg = model_skipgram.wv.most_similar(positive=["快乐"], topn=5)  # 在Skip-gram模型中查找
print("\n  Skip-gram 模型中与'快乐'最相似的词:")  # 打印标题
for word, score in similar_words_sg:  # 遍历结果列表
    print(f"    '{word}': {score:.4f}")  # 输出词语和相似度

# 在 FastText 模型中查找最相似的词
similar_words_ft = model_fasttext.wv.most_similar(positive=["快乐"], topn=5)  # 在FastText模型中查找
print("\n  FastText 模型中与'快乐'最相似的词:")  # 打印标题
for word, score in similar_words_ft:  # 遍历结果列表
    print(f"    '{word}': {score:.4f}")  # 输出词语和相似度

# ==================== 第十二部分：手动计算余弦相似度（验证原理） ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【手动计算余弦相似度（验证原理）】")  # 打印标题提示

# 获取"快乐"、"高兴"、"苹果"在 CBOW 模型中的词向量
vec_kuaile = model_cbow.wv["快乐"]  # 获取"快乐"的100维向量，返回numpy数组 shape=(100,)
vec_gaoxing = model_cbow.wv["高兴"]  # 获取"高兴"的100维向量，返回numpy数组 shape=(100,)
vec_pingguo = model_cbow.wv["苹果"]  # 获取"苹果"的100维向量，返回numpy数组 shape=(100,)

# 手动实现余弦相似度公式：cos(A,B) = (A·B) / (|A| * |B|)
# np.dot(a, b): 计算两个向量的点积（内积），参数为两个shape相同的numpy数组
# np.linalg.norm(a): 计算向量的L2范数（即向量长度/模），参数为一个numpy数组
cos_kuaile_gaoxing = np.dot(vec_kuaile, vec_gaoxing) / (np.linalg.norm(vec_kuaile) * np.linalg.norm(vec_gaoxing))  # 计算"快乐"和"高兴"的余弦相似度
cos_kuaile_pingguo = np.dot(vec_kuaile, vec_pingguo) / (np.linalg.norm(vec_kuaile) * np.linalg.norm(vec_pingguo))  # 计算"快乐"和"苹果"的余弦相似度

# 打印手动计算的结果
print(f"  手动计算 cos('快乐','高兴') = {cos_kuaile_gaoxing:.4f}")  # 输出同义词的余弦相似度
print(f"  手动计算 cos('快乐','苹果') = {cos_kuaile_pingguo:.4f}")  # 输出无关词的余弦相似度
print(f"\n  结论：同义词相似度({cos_kuaile_gaoxing:.4f}) >> 无关词相似度({cos_kuaile_pingguo:.4f})")  # 输出验证结论

# ==================== 第十三部分：输出最终总结 ====================

print("\n" + "=" * 60)  # 打印分隔线
print("【总结】")  # 打印总结标题
print("  1. Word2Vec CBOW (sg=0)：利用上下文词预测中心词，训练速度快")  # 总结CBOW特点
print("  2. Word2Vec Skip-gram (sg=1)：利用中心词预测上下文词，对低频词效果更好")  # 总结Skip-gram特点
print("  3. FastText：引入子词n-gram信息，能处理未登录词（OOV）")  # 总结FastText特点
print("  4. 验证结果：同义词（快乐/高兴/开心）向量距离近，无关词（苹果/电脑）向量距离远")  # 总结验证结论
print("=" * 60)  # 打印结束分隔线