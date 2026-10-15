# 导入 pandas 库，主要用于数据的读取、处理和操作
import pandas as pd

# 从 sklearn 库中导入模型选择模块，用于将数据集拆分为训练集和测试集
from sklearn.model_selection import train_test_split

# 从 sklearn 库中导入文本特征提取模块，用于将文本转换为词频矩阵（CountVectorizer）或 TF-IDF 矩阵（TfidfVectorizer）
from sklearn.feature_extraction.text import CountVectorizer

# 从 sklearn 库中导入多项式朴素贝叶斯分类器，非常适合处理文本分类这种离散特征的数据
from sklearn.naive_bayes import MultinomialNB

# 从 sklearn 库中导入准确率评估指标，用于计算模型预测正确的比例
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 定义数据集的本地文件路径，指向 SMSSpamCollection 文件
file_path = r"D:\Repositories\basic_algorithm\database\SMSSpamCollection"

# 使用 pandas 的 read_csv 方法读取数据集
# 参数1: file_path，即上面定义的文件路径字符串
# 参数2: sep='\t'，指定列与列之间的分隔符为制表符（Tab），因为该数据集是 TSV 格式
# 参数3: header=None，指定文件中没有表头行，列名需要手动指定
# 参数4: names=['label', 'text']，手动为这两列数据指定列名，第一列是标签，第二列是短信文本
df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])

# 将 'label' 列中的字符串标签转换为数值型标签，以便模型能够处理
# 参数: df['label'] == 'spam'，生成一个布尔型 Series，如果文本是垃圾邮件则为 True，否则为 False
# astype(int) 方法将布尔值转换为整数，True 变为 1，False 变为 0
df['label'] = (df['label'] == 'spam').astype(int)

# le = LabelEncoder()
# # 在全部标签拟合编码器
# df['label'] = le.fit_transform(df['label'])
# # 查看映射关系
# print(dict(zip(le.classes_, le.transform(le.classes_))))
# # {'ham': 0, 'spam': 1}

# 从数据集中提取特征（X）和目标变量（y）
# df['text'] 提取所有的短信文本内容，作为模型的输入特征
X = df['text']

# df['label'] 提取所有的标签（0 或 1），作为模型需要预测的目标变量
y = df['label']

# 使用 train_test_split 将数据集拆分为训练集和测试集
# 参数1: X，需要拆分的特征数据
# 参数2: y，需要拆分的目标标签数据，与 X 保持索引对齐
# 参数3: test_size=0.2，指定测试集占总数据的比例为 20%，即 80% 用于训练，20% 用于测试
# 参数4: random_state=42，设置随机种子，保证每次运行代码时拆分的结果完全一致，便于复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 CountVectorizer 对象，用于将文本数据转换为词频矩阵（Bag of Words 模型）
# 参数: 此处未传入额外参数，使用默认配置，即统计每个单词在每条短信中出现的次数
vectorizer = CountVectorizer()

# 使用训练集文本拟合向量化器，并将训练集文本转换为词频矩阵
# 参数: X_train，训练集的短信文本 Series
# 返回值: X_train_vec，一个稀疏矩阵，每一行代表一条短信，每一列代表一个单词，值为该单词在该短信中出现的次数
X_train_vec = vectorizer.fit_transform(X_train)

# 使用已经拟合好的向量化器，将测试集文本转换为词频矩阵
# 参数: X_test，测试集的短信文本 Series
# 注意: 这里只调用 transform 而不调用 fit_transform，是为了确保测试集使用与训练集完全相同的词汇表
X_test_vec = vectorizer.transform(X_test)

# 初始化多项式朴素贝叶斯分类器
# 参数: 此处未传入额外参数，使用默认的平滑参数 alpha=1.0（即拉普拉斯平滑），防止出现概率为 0 的情况
model = MultinomialNB()

# 使用训练集的词频矩阵和对应的标签训练朴素贝叶斯模型
# 参数1: X_train_vec，训练集转换后的词频矩阵（稀疏矩阵）
# 参数2: y_train，训练集对应的数值型标签（0 或 1）
model.fit(X_train_vec, y_train)

# 使用训练好的模型对训练集进行预测，用于计算训练集精度
# 参数: X_train_vec，训练集的词频矩阵
# 返回值: y_train_pred，模型对训练集每条短信的预测标签数组
y_train_pred = model.predict(X_train_vec)

# 使用训练好的模型对测试集进行预测，用于计算测试集精度（评估模型的泛化能力）
# 参数: X_test_vec，测试集的词频矩阵
# 返回值: y_test_pred，模型对测试集每条短信的预测标签数组
y_test_pred = model.predict(X_test_vec)

# 计算模型在训练集上的准确率
# 参数1: y_train，训练集的真实标签
# 参数2: y_train_pred，模型对训练集的预测标签
# 返回值: train_accuracy，一个 0 到 1 之间的浮点数，表示训练集预测正确的比例
train_accuracy = accuracy_score(y_train, y_train_pred)

# 计算模型在测试集上的准确率
# 参数1: y_test，测试集的真实标签
# 参数2: y_test_pred，模型对测试集的预测标签
# 返回值: test_accuracy，一个 0 到 1 之间的浮点数，表示测试集预测正确的比例
test_accuracy = accuracy_score(y_test, y_test_pred)

# 在控制台打印训练集的精度结果
# 参数: f"..."，使用 Python 的 f-string 格式化字符串，:.4f 表示保留 4 位小数
print(f"训练集精度 (Training Accuracy): {train_accuracy:.4f}")

# 在控制台打印测试集的精度结果
print(f"测试集精度 (Testing Accuracy): {test_accuracy:.4f}")

demo_text = ["Hello friend, I've won a prize. Could you come with me to collect it on the weekend?"]
demo_vec = vectorizer.transform(demo_text)
demo_result = model.predict(demo_vec)
demo_proba = model.predict_proba(demo_vec)
print(f"测试短信：{demo_text[0]}")
print(f"预测结果：{demo_result[0]}")
print(f"各类别概率：ham={demo_proba[0][0]:.4f}, spam={demo_proba[0][1]:.4f}")