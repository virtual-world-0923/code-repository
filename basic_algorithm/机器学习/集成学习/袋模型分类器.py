# 导入pandas，用于读取本地iris.data文本文件
import pandas as pd
# 导入numpy，用于构造测试样本、数组运算
import numpy as np
# 导入数据集划分函数，拆分训练集、测试集
from sklearn.model_selection import train_test_split
# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# 导入内置鸢尾花数据集，作为本地文件缺失时的兜底方案
from sklearn.datasets import load_iris
# 导入tree模块，用于导出文本形式决策树结构
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

local_iris_path = r"/database/iris.data"

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

try:
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

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                        n_estimators=10, random_state=42)

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
