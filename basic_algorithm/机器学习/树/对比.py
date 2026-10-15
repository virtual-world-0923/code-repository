import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False
# 加载乳腺癌数据：30 个特征，二分类，规模适中
X, y = load_breast_cancer(return_X_y=True)

# 划分训练/测试，固定随机种子保证可复现
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------- 对比 1：单棵决策树（不剪枝，故意让它过拟合） ----------
dt = DecisionTreeClassifier(random_state=42).fit(X_tr, y_tr)
print(f"单决策树   训练={accuracy_score(y_tr, dt.predict(X_tr)):.3f}  "
      f"测试={accuracy_score(y_te, dt.predict(X_te)):.3f}")   # 预期：训练1.0，测试偏低

# ---------- 对比 2：Bagging（只装袋、不随机选特征） ----------
# 注意：新版用 estimator=，旧版(<=1.1)叫 base_estimator=
bag = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),
                        n_estimators=200, random_state=42).fit(X_tr, y_tr)
print(f"Bagging    训练={accuracy_score(y_tr, bag.predict(X_tr)):.3f}  "
      f"测试={accuracy_score(y_te, bag.predict(X_te)):.3f}")

# ---------- 对比 3：随机森林（装袋 + 随机选特征，开 OOB） ----------
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",   # sqrt 是分类默认，30特征→每棵树看约5个
                            oob_score=True, random_state=42).fit(X_tr, y_tr)
print(f"随机森林   训练={accuracy_score(y_tr, rf.predict(X_tr)):.3f}  "
      f"测试={accuracy_score(y_te, rf.predict(X_te)):.3f}  "
      f"OOB={rf.oob_score_:.3f}")   # OOB 应接近测试准确率

# 观察重点：从 单树→Bagging→RF，训练精度逐步下降、测试精度逐步上升 = 方差被压下来了

# ---------- 特征重要性：哪几个特征最顶用 ----------
# imp = rf.feature_importances_
# names = load_breast_cancer().feature_names
# for n, v in sorted(zip(names, imp), key=lambda t: -t[1])[:5]:   # 打印前 5 重要特征
#     print(f"  {n:25s} {v:.3f}")


# ---------- n_estimators 收敛曲线（用 OOB，RF 特色） ----------
oob_scores = []
for n in range(10, 301, 10):                                   # 树的数量从 10 扫到 300
    m = RandomForestClassifier(n_estimators=n, oob_score=True,
                               random_state=42).fit(X_tr, y_tr)
    oob_scores.append(m.oob_score_)                            # 记录该规模下的 OOB 准确率

plt.plot(range(10, 301, 10), oob_scores, marker="o")
plt.xlabel("n_estimators（树的数量）"); plt.ylabel("OOB 准确率")
plt.title("随机森林：树越多越稳，但收益递减"); plt.grid(True); plt.show()
# 你会看到曲线快速上升后趋于平稳——这就是"集成收敛"的直观证据