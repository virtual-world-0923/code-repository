# 导入jieba中文分词库，作用是将中文句子切分为词语序列，是中文词向量训练的必备预处理工具
import jieba
# 从gensim.models模块导入Word2Vec类，作用是构建、训练Word2Vec词向量模型
from gensim.models import Word2Vec
# 从gensim.models模块导入FastText类，作用是构建、训练FastText词向量模型
from gensim.models import FastText


# ==================================================
# 任务1：准备中文语料
# ==================================================
# 定义原始中文语料字符串，作用是为词向量训练提供上下文共现数据
# 语料包含「人工智能/学习」主题的语义相关内容，以及「水果/交通」的无关内容，用于后续相似度验证
corpus_text = """
自然语言处理是人工智能的重要分支，它致力于让计算机理解和生成人类语言。
学习自然语言处理需要掌握分词、词向量、语言模型等核心知识。
词向量可以将词语转化为稠密的低维向量，捕捉词语之间的语义关联。
Word2Vec和fastText是两种经典的词向量训练算法，在NLP任务中应用广泛。
通过深度学习模型，我们可以进一步提升文本分类、情感分析等任务的效果。
研究人工智能需要不断学习新的技术，积累相关的专业技能。
计算机科学领域的知识更新很快，持续学习才能跟上技术发展的步伐。
苹果是一种常见的水果，富含维生素，对人体健康有益。
汽车是现代社会重要的交通工具，方便了人们的日常出行。
"""

# 对原始语料按换行符分割，通过列表推导式清洗得到干净的句子列表
# split('\n')：按换行符切割字符串，返回子字符串列表
# strip()：去除字符串首尾的空白字符（空格、换行、制表符）
# if sent.strip()：过滤掉空行和仅包含空白的无效行
sentences_raw = [sent.strip() for sent in corpus_text.split('\n') if sent.strip()]

# 初始化空列表，作用是存储分词后的二维语料（每个元素是一个句子的分词词语列表）
# 该格式是Gensim词向量模型的标准输入格式
sentences_cut = []

# 遍历原始句子列表，对每一个句子执行分词处理
# sent：循环变量，代表当前遍历到的单句原始文本
for sent in sentences_raw:
    # 调用jieba.lcut方法对句子进行精确模式分词，返回分词后的词语列表
    # 参数sent：待分词的中文句子字符串
    # 参数cut_all=False：分词模式，False为精确模式（默认值），分词结果精准，最适合词向量训练
    words = jieba.lcut(sent, cut_all=False)
    # 将当前句子的分词结果追加到分词语料列表中，构建二维语料结构
    sentences_cut.append(words)


# ==================================================
# 任务2：训练两种词向量模型（Word2Vec + FastText）
# 分别实现 CBOW 和 Skip-gram 两种训练模式
# ==================================================

# ----------------------
# 2.1 训练 Word2Vec CBOW 模型
# ----------------------
# 实例化Word2Vec类并传入语料，直接完成模型初始化与训练
# 参数sentences：训练语料，二维列表格式，每个元素为分词后的句子词语列表
# 参数vector_size：词向量的维度大小，设置为100；维度越高可承载更丰富语义，但需更大语料支撑
# 参数window：上下文窗口大小，代表中心词与预测上下文词的最大距离，设置为5
# 参数min_count：最小词频阈值，语料中出现次数低于该值的词语会被忽略；小语料设为1以保留所有词
# 参数workers：训练使用的CPU线程数，设置为4以提升训练速度
# 参数sg：训练模式开关，0代表CBOW（连续词袋）模型，用上下文词语预测中心词
# 参数epochs：训练迭代轮数，小语料设置为20轮，提升词向量收敛效果
w2v_cbow_model = Word2Vec(
    sentences=sentences_cut,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=100
)

# ----------------------
# 2.2 训练 Word2Vec Skip-gram 模型
# ----------------------
# 实例化Word2Vec类，使用Skip-gram模式训练
# 核心参数与CBOW模型一致，仅sg参数设为1，启用Skip-gram模式（用中心词预测上下文词语）
w2v_sg_model = Word2Vec(
    sentences=sentences_cut,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100
)

# ----------------------
# 2.3 训练 FastText CBOW 模型
# ----------------------
# 实例化FastText类并训练CBOW模式模型；FastText基于字符n-gram，可处理未登录词
# 参数sentences：训练语料，格式同Word2Vec
# 参数vector_size：词向量维度，100维
# 参数window：上下文窗口大小，5
# 参数min_count：最小词频，1
# 参数workers：线程数，4
# 参数sg：训练模式，0代表CBOW
# 参数epochs：迭代轮数，20
# 参数min_n：字符级n-gram的最小长度，设为2，捕捉短字符组合特征
# 参数max_n：字符级n-gram的最大长度，设为5，捕捉长字符组合特征
ft_cbow_model = FastText(
    sentences=sentences_cut,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=0,
    epochs=100,
    min_n=2,
    max_n=5
)

# ----------------------
# 2.4 训练 FastText Skip-gram 模型
# ----------------------
# 实例化FastText类，使用Skip-gram模式训练
# 参数与FastText CBOW模型一致，仅sg参数设为1，启用Skip-gram模式
ft_sg_model = FastText(
    sentences=sentences_cut,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100,
    min_n=2,
    max_n=5
)


# ==================================================
# 任务3：用 CBOW 模型将分词结果转为词向量
# ==================================================
# 定义目标词语，用于演示词向量提取效果
target_word = "学习"

# 从Word2Vec CBOW模型中提取目标词的词向量
# wv是模型的KeyedVectors属性，存储所有训练完成的词向量
# 传入词语字符串，返回对应numpy数组格式的词向量
w2v_cbow_vector = w2v_cbow_model.wv[target_word]
# 打印分隔标识与词向量形状，验证维度是否符合预设
print("=" * 50)
print("【Word2Vec CBOW 词向量】")
print(f"词语「{target_word}」的向量形状：", w2v_cbow_vector.shape)
print(f"词语「{target_word}」的向量数值：\n", w2v_cbow_vector)

# 从FastText CBOW模型中提取目标词的词向量
ft_cbow_vector = ft_cbow_model.wv[target_word]
# 打印FastText CBOW模型的词向量信息
print("\n【FastText CBOW 词向量】")
print(f"词语「{target_word}」的向量形状：", ft_cbow_vector.shape)


# ==================================================
# 任务4：用 Skip-gram 模型将分词结果转为词向量
# ==================================================
# 从Word2Vec Skip-gram模型中提取目标词的词向量
w2v_sg_vector = w2v_sg_model.wv[target_word]
# 打印Word2Vec Skip-gram模型的词向量信息
print("\n" + "=" * 50)
print("【Word2Vec Skip-gram 词向量】")
print(f"词语「{target_word}」的向量形状：", w2v_sg_vector.shape)

# 从FastText Skip-gram模型中提取目标词的词向量
ft_sg_vector = ft_sg_model.wv[target_word]
# 打印FastText Skip-gram模型的词向量信息
print("\n【FastText Skip-gram 词向量】")
print(f"词语「{target_word}」的向量形状：", ft_sg_vector.shape)


# ==================================================
# 任务5：效果验证 - 语义距离对比
# 验证：同义词向量距离近，无关词向量距离远
# ==================================================
# 定义语义相关词：与「学习」在语料中共现上下文多，语义关联强
related_word = "知识"
# 定义语义无关词：与「学习」所属领域完全不同，上下文共现极少
unrelated_word = "苹果"

# 定义通用相似度验证函数，封装计算逻辑与结果打印，提升代码复用性
# 参数model：训练完成的词向量模型实例（支持Word2Vec/FastText）
# 参数model_name：模型名称字符串，用于结果打印时的模型标识
def verify_similarity(model, model_name):
    # 调用模型wv的similarity方法，计算两个词语的余弦相似度
    # 参数1：第一个目标词语
    # 参数2：第二个目标词语
    # 返回值：余弦相似度，取值范围[-1, 1]，数值越大代表语义越接近、向量距离越近
    sim_related = model.wv.similarity(target_word, related_word)
    # 计算目标词与无关词的余弦相似度
    sim_unrelated = model.wv.similarity(target_word, unrelated_word)

    # 打印模型名称与分隔线
    print("\n" + "-" * 40)
    print(f"模型：{model_name}")
    # 打印相关词相似度，保留4位小数
    print(f"「{target_word}」 ↔ 「{related_word}」（语义相关）相似度：{sim_related:.4f}")
    # 打印无关词相似度
    print(f"「{target_word}」 ↔ 「{unrelated_word}」（语义无关）相似度：{sim_unrelated:.4f}")

    # 判断并输出验证结论
    if sim_related > sim_unrelated:
        print("✅ 验证通过：语义相关词的向量距离更近，符合词向量特性")
    else:
        print("⚠️  验证未通过：语料规模过小，模型未充分学习语义特征")

# 依次对4个模型执行相似度验证
print("\n" + "=" * 50)
print("【词向量语义效果验证】")
verify_similarity(w2v_cbow_model, "Word2Vec CBOW")
verify_similarity(w2v_sg_model, "Word2Vec Skip-gram")
verify_similarity(ft_cbow_model, "FastText CBOW")
verify_similarity(ft_sg_model, "FastText Skip-gram")

# 打印补充说明，解释演示语料的局限性
print("\n" + "=" * 50)
print("【说明】")
print("本次演示使用极小语料，仅用于展示代码流程与API用法。")
print("工业级词向量需要百万级以上大规模语料训练，才能获得精准的语义效果。")