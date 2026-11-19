import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

df_check = pd.read_csv('二特征_二分类.csv')
print(df_check.head())

# 1. 生成100条清晰分离的二分类数据
X, y = make_blobs(n_samples=100, n_features=2, centers=2,
                  cluster_std=0.8, center_box=(-10.0, 10.0),
                  random_state=42)

# 2. 将特征 X 和标签 y 合并，确保最后一列是类别
# 注意：y 需要是二维数组才能拼接，所以使用 y.reshape(-1, 1)
data = np.column_stack((X, y.reshape(-1, 1)))

# 3. 保存为 CSV 文件
file_path = '二特征_二分类.csv'

# 方式一：使用 numpy 保存（简单快捷）
# np.savetxt(file_path, data, delimiter=',', fmt='%.4f',
#            header='feature_1,feature_2,class', comments='')

# 方式二：使用 pandas 保存（列名对齐更优雅）
# df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'class'])
# df.to_csv(file_path, index=False)

print(f"数据已成功保存到 {file_path}，数据形状为: {data.shape}")