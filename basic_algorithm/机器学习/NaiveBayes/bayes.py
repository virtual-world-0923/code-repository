# ====================== 导入所需库 ======================
# pandas：用于读取txt数据集，结构化处理表格数据
import pandas as pd
# train_test_split：划分训练集、测试集
from sklearn.model_selection import train_test_split
# CountVectorizer：文本词袋向量化，把短信文本转为单词计数特征
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# MultinomialNB：多项式朴素贝叶斯，文本分类标准模型
from sklearn.naive_bayes import MultinomialNB
# accuracy_score：计算分类准确率
from sklearn.metrics import accuracy_score

# ====================== 1. 读取SMS垃圾邮件数据集 ======================
# 文件路径，对应你本地存放位置
file_path = r"D:\Repositories\basic_algorithm\database\SMSSpamCollection"
# sep='\t'：原始数据集使用制表符\t分隔标签和文本
# names=["label", "message"]：手动设置两列列名，第一列为标签，第二列为短信内容
df = pd.read_csv(
    file_path,
    sep='\t',
    names=["label", "message"]
)

# 查看数据集基础信息（可选，调试用）
# print(df.head())
# label列：ham=正常短信，spam=垃圾短信

# ====================== 2. 分离特征与标签 ======================
# X：特征，短信文本内容
X = df["message"]
# y：标签，ham / spam
y = df["label"]

# ====================== 3. 划分训练集、测试集 ======================
# 参数说明：
# X,y：特征与标签
# test_size=0.2：20%数据作为测试集，80%作为训练集
# random_state=42：随机种子，固定后每次运行划分结果一致，方便复现
# stratify=y：分层抽样，保证训练集、测试集中spam/ham比例和原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ====================== 4. 文本向量化：词袋模型 ======================
# 实例化词袋转换器
# stop_words='english'：自动移除英文停用词（the,a,is等无意义词汇）
# vectorizer = CountVectorizer(stop_words='english')
vectorizer = TfidfVectorizer(stop_words='english')

# fit_transform：两步合一
# fit：基于训练文本统计所有词汇，构建词汇词典
# transform：将训练文本转换成词频数字矩阵
# ⚠️重要：只能用训练集fit，测试集只允许transform，防止数据泄露
X_train_vec = vectorizer.fit_transform(X_train)

# transform：使用训练集学到的词典转换测试文本，不新增词汇
X_test_vec = vectorizer.transform(X_test)

# ====================== 5. 创建并训练多项式朴素贝叶斯模型 ======================
# MultinomialNB参数：
# alpha=1.0 拉普拉斯平滑，解决训练集中某些单词未出现导致概率为0的问题
model = MultinomialNB(alpha=1.0)

# fit()：使用向量化后的训练特征、训练标签训练朴素贝叶斯
model.fit(X_train_vec, y_train)

# ====================== 6. 预测 & 计算准确率 ======================
# 对训练集进行预测
y_train_pred = model.predict(X_train_vec)
# 计算训练集准确率
train_acc = accuracy_score(y_train, y_train_pred)

# 对测试集进行预测
y_test_pred = model.predict(X_test_vec)
# 计算测试集准确率
test_acc = accuracy_score(y_test, y_test_pred)

# ====================== 7. 打印结果 ======================
print("="*50)
print(f"训练集准确率：{train_acc:.4f}")
print(f"测试集准确率：{test_acc:.4f}")
print("="*50)

# 【拓展演示】单条短信预测示例
# demo_text = ["Win free prize now, claim your reward!"]
demo_text = ["Hello friend, I've won a prize. Could you come with me to collect it on the weekend?"]
demo_vec = vectorizer.transform(demo_text)
demo_result = model.predict(demo_vec)
demo_proba = model.predict_proba(demo_vec)
print(f"测试短信：{demo_text[0]}")
print(f"预测结果：{demo_result[0]}")
print(f"各类别概率：ham={demo_proba[0][0]:.4f}, spam={demo_proba[0][1]:.4f}")