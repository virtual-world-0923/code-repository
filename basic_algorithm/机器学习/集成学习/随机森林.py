# 导入pandas，用于读取本地iris.data文本文件
import pandas as pd
# 导入numpy，用于构造测试样本、数组运算
import numpy as np
# 导入数据集划分函数，拆分训练集、测试集
from sklearn.model_selection import train_test_split
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 导入内置鸢尾花数据集，作为本地文件缺失时的兜底方案
from sklearn.datasets import load_iris
# 导入tree模块，用于导出文本形式决策树结构
from sklearn import tree

# --------------------------
# 1 加载数据集
# --------------------------
# 定义本地iris.data绝对路径，r原始字符串防止反斜杠转义
local_iris_path = r"/database/iris.data"
# iris.data无表头，手动设置列名：4特征 + 1标签
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

try:
    # pd.read_csv：读取csv格式文本数据
    # filepath_or_buffer：读取文件路径
    # names：自定义列名，文件不含表头时必填
    # sep：字段分隔符，iris原始文件使用逗号分隔
    df = pd.read_csv(
        filepath_or_buffer=local_iris_path,
        names=column_names,
        sep=","
    )
    print("✅ 成功加载本地iris.data文件")
except FileNotFoundError:
    # 捕获文件不存在异常，切换sklearn内置数据集
    print("⚠️ 未找到本地iris.data")
    sys.exit()

# 提取特征矩阵X：所有行，前4列特征
X = df.iloc[:, 0:4]
# 提取标签向量y：所有行，最后一列分类标签
y = df.iloc[:, 4]

# --------------------------
# 2 划分训练集与测试集
# --------------------------
# train_test_split：随机切分数据集
# X：特征数据
# y：标签数据
# test_size=0.2：测试集占全部样本20%，训练集80%
# random_state=42：固定随机种子，保证分割结果可复现
# stratify=y：分层抽样，保证训练、测试集各类别比例和原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------
# 3 初始化随机森林并训练
# --------------------------
# RandomForestClassifier：随机森林分类器
# n_estimators=10：森林内决策树总数量
# random_state=42：固定随机种子，训练结果可复现
# max_depth=None：不限制单棵树最大深度，允许自由生长
# bootstrap=True：开启自助采样，随机森林标准配置
# n_jobs=-1：使用本机全部CPU核心并行训练
rf = RandomForestClassifier(
    n_estimators=10,
    random_state=42,
    max_depth=3,
    bootstrap=True,
    n_jobs=-1
)

# fit：使用训练数据拟合模型（执行训练）
# X_train：训练集特征
# y_train：训练集标签
rf.fit(X_train, y_train)

# --------------------------
# 4 输出训练集、测试集精度
# --------------------------
# score：计算数据集上分类准确率
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print("=" * 80)
print(f"训练集准确率：{train_acc:.4f}")
print(f"测试集准确率：{test_acc:.4f}")
print("=" * 80)

import 打印树结构 as ptree
ptree.print_tree_info(rf,X_train, y_train)
# --------------------------
# 输入测试样本，打印决策路径
# --------------------------
# 构造测试样本 [sepal_length,sepal_width,petal_length,petal_width]
test_sample_a = np.array([[5.4, 3.4, 1.7, 0.2]])
test_sample = pd.DataFrame(test_sample_a, columns=column_names[0:4])
print(f"\n输入测试数据：{test_sample}")

# 随机森林整体预测结果
pred_res = rf.predict(test_sample)
print(f"\n随机森林最终预测类别：{pred_res[0]}")
print("=" * 80)

ptree.tree_decision_path(rf.estimators_[0], test_sample,column_names)

# a = len(rf.estimators_samples_[0])
# b = len(np.unique(rf.estimators_samples_[0]))
# print(a)
# print(b)

# print(rf.estimators_) #树对象列表
# print('-'*100)
# print(rf.estimator)
# print('-'*100)
# print(rf.estimator_)
# print('-'*100)
# print(rf.estimators_samples_) #每棵树使用的样本
# print('-'*100)
# print(rf.estimator_params) #每棵树的参数
# print('-'*100)
# print(rf.n_estimators) #树总量
# print('-'*100)