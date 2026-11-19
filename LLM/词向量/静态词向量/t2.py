# -*- coding: utf-8 -*-
"""
使用 Gensim 库分别训练 Word2Vec 和 FastText 两种词向量模型
包含 CBOW 和 Skip-gram 两种训练方式，并进行效果验证
环境：Python 3.13.14 / Gensim 4.x
"""

# ================================================================
# 第一部分：导入所需的第三方库
# ================================================================

import jieba                                    # 导入 jieba 中文分词库，用于将中文文本切分为词语列表
from gensim.models import Word2Vec              # 从 gensim 库中导入 Word2Vec 词向量模型类
from gensim.models import FastText              # 从 gensim 库中导入 FastText 词向量模型类
import numpy as np                              # 导入 numpy 数值计算库，用于向量距离计算
import warnings                                 # 导入 warnings 模块，用于控制警告信息的显示
warnings.filterwarnings("ignore")               # 过滤掉训练过程中产生的无关警告信息，保持输出整洁


# ================================================================
# 第二部分：准备中文语料
# ================================================================

# 定义原始中文语料文本，涵盖多个主题领域，以提供丰富的共现关系
raw_corpus = [
    "人工智能是计算机科学的一个重要分支，旨在创建能够模拟人类智能的系统",
    "机器学习是人工智能的核心技术之一，通过数据驱动的方式让计算机自动学习规律",
    "深度学习是机器学习的一个子领域，利用多层神经网络来处理复杂的模式识别任务",
    "自然语言处理是人工智能的重要应用方向，致力于让计算机理解和生成人类语言",
    "计算机视觉是人工智能的热门研究方向，使机器能够识别和理解图像中的内容",
    "神经网络是一种模仿生物神经系统的计算模型，由大量相互连接的节点组成",
    "卷积神经网络在图像识别领域取得了巨大成功，能够自动提取图像的特征",
    "循环神经网络擅长处理序列数据，在自然语言处理和语音识别中应用广泛",
    "Transformer 架构革新了自然语言处理领域，成为了当前大语言模型的基础",
    "BERT 是一种预训练语言模型，通过双向编码器来理解文本的深层语义信息",
    "GPT 是一种生成式预训练模型，能够根据上下文生成连贯且有意义的文本",
    "词向量是将词语映射到连续向量空间的技术，能够捕捉词语之间的语义关系",
    "Word2Vec 是经典的词向量模型，包括 CBOW 和 Skip-gram 两种训练方式",
    "FastText 在 Word2Vec 基础上引入了子词信息，能够处理未登录词和形态丰富的语言",
    "CBOW 模型通过上下文词语来预测中心词，适合频繁出现的常用词的训练",
    "Skip-gram 模型通过中心词来预测上下文词语，适合处理低频词和小规模语料",
    "Python 是人工智能领域最流行的编程语言，拥有丰富的机器学习和深度学习库",
    "TensorFlow 和 PyTorch 是两大主流深度学习框架，广泛应用于学术研究和工业部署",
    "数据预处理是机器学习流程中的关键步骤，包括数据清洗、特征工程和数据增强",
    "过拟合是机器学习中常见的问题，可以通过正则化和数据增强来缓解",
    "梯度下降是优化神经网络参数的核心算法，学习率的选择直接影响模型的收敛速度",
    "损失函数衡量模型预测值与真实值之间的差异，常见的有交叉熵损失和均方误差损失",
    "准确率、精确率、召回率和 F1 分数是评估分类模型性能的重要指标",
    "监督学习需要标注好的训练数据，无监督学习则从未标注的数据中发现隐藏模式",
    "强化学习通过与环境的交互来学习最优策略，在游戏和机器人控制中有广泛应用",
    "迁移学习利用已有模型的知识来加速新任务的学习，减少了对大量标注数据的需求",
    "注意力机制使模型能够关注输入中最相关的部分，显著提升了序列建模的效果",
    "生成对抗网络由生成器和判别器组成，通过对抗训练来生成逼真的数据样本",
    "大语言模型通过海量文本数据进行预训练，展现了强大的语言理解和生成能力",
    "提示工程是引导大语言模型产生期望输出的技术，是当前人工智能应用的热点",
    "深度学习模型的训练需要大量的计算资源，GPU 和 TPU 是常用的硬件加速器",
    "批归一化是一种加速深度网络训练的技术，通过标准化每层的输入来稳定训练过程",
    "残差连接解决了深层网络中的梯度消失问题，使得训练非常深的网络成为可能",
    "医学影像分析利用深度学习技术辅助医生进行疾病诊断，提高了诊断的准确性和效率",
    "自动驾驶是人工智能的重要应用场景，涉及感知、决策和控制等多个技术环节",
    "语音识别将人类语音转换为文字，语音合成则将文字转换为自然流畅的语音",
    "推荐系统通过分析用户行为和偏好来预测用户可能感兴趣的内容或商品",
    "知识图谱以图结构组织实体和关系，为智能问答和推理提供结构化的知识基础",
    "联邦学习允许多个参与方在不共享原始数据的情况下协作训练模型，保护了数据隐私",
    "模型压缩技术包括剪枝、量化和知识蒸馏，旨在减小模型体积并提升推理速度",
]

# 定义停用词列表，这些词在中文文本中高频出现但对语义分析贡献较小
stop_words = {
    "的", "了", "在", "是", "和", "与", "对", "中", "为", "等",    # 常见虚词
    "个", "到", "从", "将", "也", "不", "有", "这", "上", "一",    # 常见功能词
    "都", "而", "及", "或", "其", "能", "来", "以", "被",          # 更多虚词
}


# ================================================================
# 第三部分：添加自定义词典，确保专业术语不被切碎
# ================================================================

# 向 jieba 添加自定义词汇，确保复合专业术语作为整体被识别
# jieba.add_word(word, freq=None, tag=None)
# 参数 word: 要添加的词语字符串
# 参数 freq: 词频，值越大越优先被识别为完整词语
# 参数 tag: 词性标注（此处不需要，设为 None）
custom_words = [
    "深度学习", "机器学习", "自然语言处理", "计算机视觉",     # 核心 AI 领域术语
    "神经网络", "卷积神经网络", "循环神经网络",                # 网络结构术语
    "词向量", "生成对抗网络", "注意力机制",                    # 技术术语
    "大语言模型", "迁移学习", "联邦学习",                      # 高级概念术语
    "强化学习", "监督学习", "无监督学习",                      # 学习范式术语
    "梯度下降", "损失函数", "过拟合", "正则化",               # 训练相关术语
    "批归一化", "残差连接", "知识蒸馏",                        # 网络优化术语
    "数据预处理", "特征工程", "数据增强",                      # 数据处理术语
    "模型压缩", "知识图谱", "推荐系统",                        # 应用相关术语
    "语音识别", "语音合成", "自动驾驶",                        # 应用场景术语
    "提示工程", "预训练", "医学影像",                          # 新兴术语
]

for word in custom_words:
    #告诉 jieba：这是一个合法的中文词，以后分词时必须把它当成一个整体，不要拆开。
    jieba.add_word(
        word,  # 要加入词典的词（字符串）
        freq=100000 # 该词在词典中的词频（int）
    )


# ================================================================
# 第四部分：语料预处理 —— 中文分词
# ================================================================

# 定义语料预处理函数：分词 + 去停用词
def preprocess_corpus(corpus, stop_words_set):
    """
    对中文语料进行分词和停用词过滤的预处理函数
    参数:
        corpus (list of str): 原始中文文本列表，每个元素为一个句子
        stop_words_set (set): 停用词集合，用于过滤无意义的高频词
    返回:
        list of list of str: 分词后的二维列表，每个内层列表为一个句子的词语序列
    """
    tokenized_corpus = []                    # 初始化空列表，用于存储所有分词后的句子
    for sentence in corpus:                  # 遍历语料中的每一个句子
        words = jieba.lcut(sentence)         # 使用 jieba 的精确模式对句子进行分词，返回词语列表
        # 过滤：去除停用词、去除长度为 1 的单字词（通常语义较弱）、去除纯空白字符
        filtered = [
            w for w in words                 # 遍历分词结果中的每个词语
            if w not in stop_words_set       # 保留不在停用词表中的词
            and len(w.strip()) > 1           # 且词语去除空白后长度大于 1（过滤单字）
            and w.strip() != ""              # 且不是纯空白字符串
        ]
        if filtered:                         # 如果过滤后该句子仍有词语保留
            tokenized_corpus.append(filtered)  # 将过滤后的词语列表加入最终语料
    return tokenized_corpus                  # 返回预处理完成的分词语料

# 调用预处理函数，将原始语料转为分词后的语料
sentences = preprocess_corpus(raw_corpus, stop_words)  # 传入原始语料和停用词集合，获取分词结果

# 打印分词结果，用于检查预处理效果
print("=" * 70)                              # 打印分隔线，美化输出
print("【语料预处理结果】")                     # 打印标题
print(f"共 {len(sentences)} 个有效句子")       # 打印有效句子数量
print("前 8 个句子的分词结果：")                  # 打印提示信息
for i, sent in enumerate(sentences[:8]):     # 遍历前 8 个分词结果
    print(f"  句子{i+1}: {' / '.join(sent)}")  # 用 " / " 连接词语，清晰展示分词结果
print("=" * 70)                              # 打印分隔线


# ================================================================
# 第五部分：使用 Word2Vec 的 CBOW 模型训练词向量
# ================================================================

print("\n>>> 训练 Word2Vec CBOW 模型...")       # 打印训练开始提示

# 创建 Word2Vec CBOW 模型实例
# sg=0 表示使用 CBOW（Continuous Bag of Words）架构
model_w2v_cbow = Word2Vec(
    sentences=sentences,                     # 传入分词后的语料，格式为 list of list of str
    vector_size=100,                         # 词向量的维度，设为 100 维
    window=5,                                # 上下文窗口大小，取中心词左右各 5 个词作为上下文
    min_count=1,                             # 词语最低出现次数，出现次数低于此值的词将被忽略（语料小设为 1）
    workers=1,                               # 并行训练使用的线程数，设为 1 以保证可复现性
    sg=0,                                    # sg=0 表示使用 CBOW 模型；sg=1 表示 Skip-gram
    epochs=200,                              # 训练的迭代轮数，语料较小所以增加轮数以充分学习
    seed=42                                  # 随机种子，确保每次运行结果一致
)

# 打印训练完成信息及模型基本参数
print(f"  Word2Vec CBOW 词汇量：{len(model_w2v_cbow.wv)}")          # 获取模型学到的词汇总数
print(f"  词向量维度：{model_w2v_cbow.wv.vector_size}")              # 获取词向量的维度


# ================================================================
# 第六部分：使用 Word2Vec 的 Skip-gram 模型训练词向量
# ================================================================

print("\n>>> 训练 Word2Vec Skip-gram 模型...")  # 打印训练开始提示

# 创建 Word2Vec Skip-gram 模型实例
# sg=1 表示使用 Skip-gram 架构
model_w2v_sg = Word2Vec(
    sentences=sentences,                     # 传入分词后的语料
    vector_size=100,                         # 词向量维度，与 CBOW 保持一致便于比较
    window=5,                                # 上下文窗口大小
    min_count=1,                             # 词语最低出现次数
    workers=1,                               # 并行训练线程数
    sg=1,                                    # sg=1 表示使用 Skip-gram 模型
    epochs=200,                              # 训练迭代轮数
    seed=42                                  # 随机种子
)

# 打印训练完成信息
print(f"  Word2Vec Skip-gram 词汇量：{len(model_w2v_sg.wv)}")       # 获取词汇总数
print(f"  词向量维度：{model_w2v_sg.wv.vector_size}")                # 获取向量维度


# ================================================================
# 第七部分：使用 FastText 的 CBOW 模型训练词向量
# ================================================================

print("\n>>> 训练 FastText CBOW 模型...")       # 打印训练开始提示

# 创建 FastText CBOW 模型实例
# FastText 相比 Word2Vec 的关键区别：引入了子词（n-gram）信息
# 使得模型可以为训练集中未出现过的词（OOV）生成词向量
model_ft_cbow = FastText(
    sentences=sentences,                     # 传入分词后的语料
    vector_size=100,                         # 词向量维度
    window=5,                                # 上下文窗口大小
    min_count=1,                             # 词语最低出现次数
    workers=1,                               # 并行训练线程数
    sg=0,                                    # sg=0 表示使用 CBOW 架构
    epochs=200,                              # 训练迭代轮数
    min_n=2,                                 # 子词（character n-gram）最小长度，用于捕捉词内部结构
    max_n=5,                                 # 子词最大长度，与 min_n 共同定义子词范围
    seed=42                                  # 随机种子
)

# 打印训练完成信息
print(f"  FastText CBOW 词汇量：{len(model_ft_cbow.wv)}")           # 获取词汇总数
print(f"  词向量维度：{model_ft_cbow.wv.vector_size}")               # 获取向量维度


# ================================================================
# 第八部分：使用 FastText 的 Skip-gram 模型训练词向量
# ================================================================

print("\n>>> 训练 FastText Skip-gram 模型...")  # 打印训练开始提示

# 创建 FastText Skip-gram 模型实例
model_ft_sg = FastText(
    sentences=sentences,                     # 传入分词后的语料
    vector_size=100,                         # 词向量维度
    window=5,                                # 上下文窗口大小
    min_count=1,                             # 词语最低出现次数
    workers=1,                               # 并行训练线程数
    sg=1,                                    # sg=1 表示使用 Skip-gram 架构
    epochs=200,                              # 训练迭代轮数
    min_n=2,                                 # 子词 n-gram 最小长度
    max_n=5,                                 # 子词 n-gram 最大长度
    seed=42                                  # 随机种子
)

# 打印训练完成信息
print(f"  FastText Skip-gram 词汇量：{len(model_ft_sg.wv)}")        # 获取词汇总数
print(f"  词向量维度：{model_ft_sg.wv.vector_size}")                 # 获取向量维度


# ================================================================
# 第九部分：打印完整词汇表，确认可测试词汇
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【Word2Vec CBOW 词汇表（全部词汇）】")    # 打印标题

# 获取 Word2Vec CBOW 模型的完整词汇表
vocab_words = list(model_w2v_cbow.wv.index_to_key)  # index_to_key 是按词频降序排列的词汇列表
print(f"  共 {len(vocab_words)} 个词")        # 打印词汇总数
# 每行打印 8 个词，便于浏览
for i in range(0, len(vocab_words), 8):       # 以步长 8 遍历词汇列表
    batch = vocab_words[i:i+8]                # 取出当前行的词汇切片
    print(f"  {' | '.join(batch)}")            # 用 " | " 分隔打印


# ================================================================
# 第十部分：词向量提取示例 —— 查看单个词的向量表示
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【词向量提取示例】")                     # 打印标题

# 选择一个确实存在于词汇表中的词作为示例
# 从词汇表中选取前几个高频词之一
example_word = vocab_words[0]                 # 取词汇表中词频最高的词作为示例
print(f"\n  示例词：'{example_word}'")          # 打印示例词

# 定义辅助函数：安全获取词向量
def safe_get_vector(model, word, model_name):
    """
    安全地从模型中获取指定词的词向量
    参数:
        model: 已训练好的 Word2Vec 或 FastText 模型对象
        word (str): 要查询的词语
        model_name (str): 模型名称字符串，用于打印提示信息
    返回:
        numpy.ndarray 或 None: 如果词存在返回词向量数组，否则返回 None
    """
    if word in model.wv:                      # 检查词语是否在模型词汇表中
        vec = model.wv[word]                  # 通过词语作为键索引获取词向量（numpy 数组）
        print(f"  {model_name}：'{word}' 向量（前10维）= {vec[:10]}")  # 打印前 10 维
        return vec                            # 返回词向量
    else:                                     # 如果词不在词汇表中
        print(f"  {model_name}：'{word}' 不在词汇表中")  # 提示词缺失
        return None                           # 返回 None

# 从四个模型中分别获取该示例词的词向量
safe_get_vector(model_w2v_cbow, example_word, "Word2Vec CBOW")       # Word2Vec CBOW
safe_get_vector(model_w2v_sg, example_word, "Word2Vec Skip-gram")    # Word2Vec Skip-gram
safe_get_vector(model_ft_cbow, example_word, "FastText CBOW")        # FastText CBOW
safe_get_vector(model_ft_sg, example_word, "FastText Skip-gram")     # FastText Skip-gram


# ================================================================
# 第十一部分：效果验证 —— 余弦相似度测试
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【效果验证：同义词 vs 无关词的向量距离对比】")  # 打印标题

# 根据实际词汇表构建测试词对
# 先检查哪些候选词在词汇表中，动态筛选可用的测试对

def find_valid_pairs(model, candidate_pairs, label):
    """
    从候选词对列表中筛选出两个词都在词汇表中的有效词对
    参数:
        model: 已训练好的模型对象，用于检查词汇是否存在
        candidate_pairs (list of tuple): 候选词对列表，每个元素为 (词A, 词B)
        label (str): 标签名称，用于打印提示
    返回:
        list of tuple: 筛选后的有效词对列表
    """
    valid = []                                # 初始化空列表，存储有效词对
    for w1, w2 in candidate_pairs:            # 遍历每个候选词对
        if w1 in model.wv and w2 in model.wv:  # 如果两个词都在词汇表中
            valid.append((w1, w2))            # 将该词对加入有效列表
        else:                                 # 如果有词不在词汇表中
            missing = []                      # 初始化缺失词列表
            if w1 not in model.wv:            # 如果词A缺失
                missing.append(w1)            # 记录词A
            if w2 not in model.wv:            # 如果词B缺失
                missing.append(w2)            # 记录词B
            print(f"  跳过 {label} 对 ({w1}, {w2})：缺少 {missing}")  # 打印跳过提示
    return valid                              # 返回有效词对列表


# 定义同义/近义词候选对
synonym_candidates = [
    ("深度学习", "神经网络"),                  # 深度学习基于神经网络，语义相近
    ("机器学习", "人工智能"),                  # 机器学习是人工智能的子领域
    ("自然语言处理", "计算机视觉"),              # 同属 AI 研究方向
    ("训练", "学习"),                          # 在 ML 语境下语义相近
    ("卷积神经网络", "神经网络"),              # 卷积神经网络是神经网络的一种
    ("循环神经网络", "神经网络"),              # 循环神经网络是神经网络的一种
    ("词向量", "自然语言处理"),                # 词向量常用于 NLP
    ("强化学习", "监督学习"),                  # 同属学习范式
    ("生成对抗网络", "神经网络"),              # GAN 基于神经网络
    ("预训练", "训练"),                        # 预训练是训练的一种方式
]

# 定义无关词候选对
unrelated_candidates = [
    ("深度学习", "自动驾驶"),                  # 技术 vs 应用场景
    ("机器学习", "语音合成"),                  # 方法论 vs 具体技术
    ("自然语言处理", "梯度下降"),              # 应用方向 vs 优化算法
    ("计算机视觉", "损失函数"),                # 应用方向 vs 训练组件
    ("神经网络", "推荐系统"),                  # 模型 vs 应用
    ("卷积神经网络", "语音识别"),              # 视觉模型 vs 语音应用
    ("注意力机制", "自动驾驶"),                # 技术机制 vs 应用场景
    ("词向量", "梯度下降"),                    # 表示方法 vs 优化算法
    ("数据预处理", "神经网络"),                # 数据阶段 vs 模型结构
    ("迁移学习", "损失函数"),                  # 学习策略 vs 训练组件
]

# 定义模型字典，统一管理四个训练好的模型
models = {
    "Word2Vec CBOW":       model_w2v_cbow,   # 键为模型名称，值为对应的模型对象
    "Word2Vec Skip-gram":  model_w2v_sg,     # Word2Vec Skip-gram 模型
    "FastText CBOW":       model_ft_cbow,    # FastText CBOW 模型
    "FastText Skip-gram":  model_ft_sg,      # FastText Skip-gram 模型
}

# 遍历每个模型进行相似度测试
for model_name, model in models.items():     # 解包字典的键值对
    print(f"\n{'─' * 70}")                   # 打印子分隔线
    print(f"  模型：{model_name}")             # 打印当前测试的模型名称
    print(f"{'─' * 70}")                     # 打印子分隔线

    # 筛选有效的同义词对和无关词对
    synonym_pairs = find_valid_pairs(model, synonym_candidates, "同义")   # 筛选同义词对
    unrelated_pairs = find_valid_pairs(model, unrelated_candidates, "无关")  # 筛选无关词对

    # --- 测试同义词对的相似度 ---
    print("  [同义/近义词对]")                  # 打印同义词测试标题
    synonym_scores = []                        # 初始化列表，用于存储同义词对的相似度得分

    for word_a, word_b in synonym_pairs:     # 遍历每一对同义词
        # 使用 Gensim 内置的 similarity 方法计算两个词的余弦相似度
        # 余弦相似度范围为 [-1, 1]，值越大表示两个词的向量越接近，语义越相似
        sim_score = model.wv.similarity(word_a, word_b)  # 传入两个词，返回余弦相似度值
        synonym_scores.append(sim_score)      # 将得分加入列表
        print(f"    similarity('{word_a}', '{word_b}') = {sim_score:.4f}")  # 格式化输出相似度

    # --- 测试无关词对的相似度 ---
    print("  [无关词对]")                       # 打印无关词测试标题
    unrelated_scores = []                      # 初始化列表，用于存储无关词对的相似度得分

    for word_a, word_b in unrelated_pairs:   # 遍历每一对无关词
        sim_score = model.wv.similarity(word_a, word_b)  # 计算两个词的余弦相似度
        unrelated_scores.append(sim_score)    # 将得分加入列表
        print(f"    similarity('{word_a}', '{word_b}') = {sim_score:.4f}")  # 格式化输出

    # --- 计算并对比平均相似度 ---
    if synonym_scores and unrelated_scores:  # 确保两种得分列表都非空
        avg_syn = np.mean(synonym_scores)    # 计算同义词对的平均相似度
        avg_unrel = np.mean(unrelated_scores)  # 计算无关词对的平均相似度
        print(f"\n  >>> 同义词平均相似度：{avg_syn:.4f}")       # 打印同义词平均得分
        print(f"  >>> 无关词平均相似度：{avg_unrel:.4f}")        # 打印无关词平均得分
        print(f"  >>> 差值（应为正数）：  {avg_syn - avg_unrel:.4f}")  # 差值越大说明区分能力越强

        # 根据差值判断模型效果
        if avg_syn > avg_unrel:              # 如果同义词相似度高于无关词
            print(f"  ✓ 验证通过：同义词距离 < 无关词距离，模型效果良好")  # 模型有效
        else:                                # 反之
            print(f"  ✗ 验证未通过：可能是语料量不足导致")  # 可能因语料太少


# ================================================================
# 第十二部分：效果验证 —— most_similar 查找最相似词语
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【效果验证：查找与目标词最相似的 Top-5 词语】")  # 打印标题

# 从词汇表中选取不同频率的词作为查询词
# 高频词、中频词各取一些，覆盖面更广
query_words = []                              # 初始化查询词列表
# 遍历词汇表，选取几个有代表性的词
for w in vocab_words:                         # 遍历所有词汇
    if len(w) >= 2 and w not in query_words:  # 确保是多字词且未重复添加
        query_words.append(w)                 # 加入查询列表
    if len(query_words) >= 6:                 # 最多取 6 个查询词
        break                                 # 达到数量后停止

for model_name, model in models.items():     # 遍历每个模型
    print(f"\n  模型：{model_name}")           # 打印模型名称
    for word in query_words:                  # 遍历每个查询词
        # 使用 most_similar 方法查找与目标词语义最接近的 topn 个词
        # 该方法基于余弦相似度在词向量空间中进行近邻搜索
        similar_words = model.wv.most_similar(
            positive=word,                   # positive 参数：正向查询词，即要找相似词的目标词
            topn=5                           # topn 参数：返回前 5 个最相似的词
        )
        # 格式化输出结果
        result_str = ", ".join(              # 用逗号连接所有结果
            [f"{w}({s:.3f})" for w, s in similar_words]  # 每个结果格式为 "词(相似度)"
        )
        print(f"    '{word}' 的近义词：{result_str}")  # 打印查询结果


# ================================================================
# 第十三部分：效果验证 —— FastText 的 OOV（未登录词）能力验证
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【FastText 特有能力：对未登录词（OOV）生成向量】")  # 打印标题

# FastText 的核心优势：通过子词（character n-gram）信息，
# 可以为训练集中从未出现过的词生成词向量
# 而 Word2Vec 无法做到这一点

# 定义多个未登录词进行测试
oov_words = ["人工智能技术", "深度神经网络", "自然语言理解", "智能推荐"]  # 训练集中未出现的组合词

for oov_word in oov_words:                   # 遍历每个未登录词
    print(f"\n  测试未登录词：'{oov_word}'")   # 打印测试词

    for model_name, model in models.items():  # 遍历每个模型
        if oov_word in model.wv:              # 如果该词在模型词汇表中（Word2Vec 不会有）
            vec = model.wv[oov_word]          # 获取词向量
            print(f"    {model_name}：✓ 向量获取成功，前5维 = {vec[:5]}")  # 打印向量片段
        else:                                 # 如果该词不在词汇表中
            try:
                # FastText 的 wv 对象在 get_vector 中会自动通过子词合成向量
                # 即使词不在词汇表中，只要包含已知子词就能生成向量
                vec = model.wv.get_vector(oov_word)  # 尝试通过子词 n-gram 合成向量
                print(f"    {model_name}：✓ 子词合成成功，前5维 = {vec[:5]}")  # FastText 成功
            except KeyError:                  # 如果完全无法生成向量
                print(f"    {model_name}：✗ KeyError —— 无法生成向量")  # Word2Vec 会走到这里


# ================================================================
# 第十四部分：效果验证 —— 向量运算（词类比任务）
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【效果验证：词向量运算（类比任务）】")      # 打印标题

# 使用 most_similar 的 positive/negative 参数进行类比推理
# 词向量运算遵循：positive 中的词向量相加，negative 中的词向量相减

# 定义类比任务列表：(加词1, 加词2, 减词, 任务描述)
analogy_tasks = [
    (["机器学习", "计算机视觉"], ["人工智能"], "机器学习 + 计算机视觉 - 人工智能 ≈ ?"),
    (["神经网络", "自然语言处理"], ["机器学习"], "神经网络 + 自然语言处理 - 机器学习 ≈ ?"),
    (["卷积神经网络", "语音识别"], ["神经网络"], "卷积神经网络 + 语音识别 - 神经网络 ≈ ?"),
]

for model_name, model in models.items():     # 遍历每个模型
    print(f"\n  模型：{model_name}")           # 打印模型名称
    for pos_words, neg_words, desc in analogy_tasks:  # 遍历每个类比任务
        # 检查所有涉及的词是否在词汇表中
        all_words = pos_words + neg_words     # 合并所有需要的词
        all_exist = all(w in model.wv for w in all_words)  # 检查每个词是否都在词汇表中
        if not all_exist:                     # 如果有词缺失
            missing = [w for w in all_words if w not in model.wv]  # 找出缺失的词
            print(f"    {desc} → 跳过（缺少: {missing}）")  # 打印跳过提示
            continue                          # 跳过当前任务

        # 执行类比运算
        # most_similar 的 positive 参数：要相加的词列表
        # most_similar 的 negative 参数：要相减的词列表
        analogy_result = model.wv.most_similar(
            positive=pos_words,               # 正向词列表，其向量会相加
            negative=neg_words,               # 负向词列表，其向量会相减
            topn=5                            # 返回前 5 个最接近的结果
        )
        result_str = ", ".join(              # 用逗号连接所有结果
            [f"{w}({s:.3f})" for w, s in analogy_result]  # 每个结果格式为 "词(相似度)"
        )
        print(f"    {desc}")                  # 打印任务描述
        print(f"      → {result_str}")        # 打印类比结果


# ================================================================
# 第十五部分：保存模型（可选）
# ================================================================

# print("\n" + "=" * 70)                       # 打印分隔线
# print("【保存模型】")                          # 打印标题
#
# # 定义保存路径
# save_dir = "./saved_models"                   # 定义模型保存的目录路径
#
# import os                                     # 导入 os 模块，用于文件和目录操作
# os.makedirs(save_dir, exist_ok=True)          # 创建保存目录，如果目录已存在则不报错
#
# # 保存 Word2Vec CBOW 模型
# model_w2v_cbow.save(f"{save_dir}/w2v_cbow.model")          # 将模型序列化保存到指定路径
# print(f"  已保存：{save_dir}/w2v_cbow.model")               # 打印保存成功提示
#
# # 保存 Word2Vec Skip-gram 模型
# model_w2v_sg.save(f"{save_dir}/w2v_skipgram.model")        # 保存 Word2Vec Skip-gram 模型
# print(f"  已保存：{save_dir}/w2v_skipgram.model")           # 打印保存成功提示
#
# # 保存 FastText CBOW 模型
# model_ft_cbow.save(f"{save_dir}/ft_cbow.model")            # 保存 FastText CBOW 模型
# print(f"  已保存：{save_dir}/ft_cbow.model")                # 打印保存成功提示
#
# # 保存 FastText Skip-gram 模型
# model_ft_sg.save(f"{save_dir}/ft_skipgram.model")          # 保存 FastText Skip-gram 模型
# print(f"  已保存：{save_dir}/ft_skipgram.model")            # 打印保存成功提示


# ================================================================
# 第十六部分：总结
# ================================================================

print("\n" + "=" * 70)                       # 打印分隔线
print("【总结】")                              # 打印标题
print("""
  1. Word2Vec CBOW：通过上下文预测中心词，训练速度快，对高频词效果好
  2. Word2Vec Skip-gram：通过中心词预测上下文，对低频词和小语料表现更好
  3. FastText CBOW：在 CBOW 基础上引入子词信息，能处理未登录词（OOV）
  4. FastText Skip-gram：在 Skip-gram 基础上引入子词信息，OOV 能力 + 低频词优势
  5. 同义词的向量余弦相似度应明显高于无关词，验证了词向量能捕捉语义关系
  6. FastText 独有能力：可为训练集中未出现的词生成向量（通过子词 n-gram 拼接）
""")                                          # 打印总结内容
print("=" * 70)                               # 打印结束分隔线