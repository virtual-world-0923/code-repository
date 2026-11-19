import numpy as np
import pandas as pd
import os
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# os.chdir(r'D:\Repositories\basic_algorithm\data')
# mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 1. 设置最大显示列数为 None（即不限制，显示所有列）
pd.set_option('display.max_columns', None)

# 2. 设置显示宽度（例如设置为 1000 字符，防止换行截断）
pd.set_option('display.width', 1000)

# 3. (可选) 如果某列内容太长被截断，可以设置最大列宽
pd.set_option('display.max_colwidth', None)

# 设置浮点数显示格式为普通小数，保留2位小数（可根据需要修改 .2f）
pd.set_option('display.float_format', '{:.2f}'.format)

# 使用 lambda 函数定义格式：{:.2f} 表示保留2位小数的浮点数
np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})


def data_transition(df):
    if df is None:
        return None
    feature_columns = df.iloc[:,:-1].to_numpy() #最后一列之前的列当作特征
    label_column = df.iloc[:,-1].to_numpy() #最后一列当作类别
    if label_column.dtype == object or isinstance(label_column[0], str):
        le = LabelEncoder()
        label_column = le.fit_transform(label_column) #把类别转成编码格式 0，1，2，3.....
    return feature_columns, label_column.reshape(-1, 1) #转成二维数组

def getPath(file_name: str):
    # 1. 获取当前 .py 文件所在的文件夹绝对路径
    script_dir = Path(__file__).parent

    # 2. 使用 / 运算符优雅地拼接路径
    file_path = script_dir / file_name
    return file_path

class ReadIris:
    """
        读取鸢尾花（Iris）数据集，150 x 5
        sepal_length：花萼长度
        sepal_width：花萼宽度
        petal_length：花瓣长度
        petal_width：花瓣宽度
        class：
            -- Iris Setosa：山鸢尾
            -- Iris Versicolour：变色鸢尾
            -- Iris Virginica：维尔吉鸢尾
    """
    # 1. 将列名定义为类常量，方便复用
    COLUMN_NAMES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    def __init__(self, file_path='iris.data'):
        """
        初始化方法：负责读取数据并存储为实例属性
        """
        file_path = getPath(file_path)
        try:
            # 使用 self.df 将数据绑定到实例上
            self.df = pd.read_csv(file_path, header=None, names=self.COLUMN_NAMES)
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}，请检查路径。")
            self.df = None

    def get_data(self, feature_number=None, categories_number=3, sample_size=150):
        """
        获取鸢尾花数据集的子集

        参数:
            feature_number: list, 需要获取的特征名称列表，默认为None（获取全部4个特征）
            categories_number: int, 需要获取的分类数量，默认为3
            sample_size: int, 需要获取的样本数量，默认为150
        返回:
            X: np.ndarray, shape=(sample_size, len(feature_number))
            y: np.ndarray, shape=(sample_size, 1)
        """
        # 1. 处理分类数量（只保留前 categories_number 个类别）
        # 获取数据集中所有的类别，并截取前几个（例如 [0, 1]）
        unique_classes = sorted(self.df['class'].unique())[:categories_number]
        # 使用 pandas 的 isin 进行过滤
        df_filtered = self.df[self.df['class'].isin(unique_classes)]

        # 2. 处理特征选择
        if feature_number is None:
            # 默认获取除 'class' 外的全部4个特征
            feature_cols = self.COLUMN_NAMES[:-1]
        else:
            feature_cols = feature_number

        # 3. 处理样本数量（截取前 sample_size 条）
        df_result = df_filtered.head(sample_size)

        # 4. 提取 X 和 y，并转换为 numpy 数组
        X = df_result[feature_cols].values  # 自动变为 (m, n) 的二维数组
        y = df_result['class'].values    # 如果参数写成的是双中括号 [['class']]，保证形状为 (n, 1)

        y = LabelEncoder().fit_transform(y).reshape(-1, 1)
        return X, y


class CaliforniaHousing:
    """
    原始加州房价数据集 20640

    Attributes:
        longitude：经度
        latitude：纬度
        housing_median_age：房屋中位年龄
        total_rooms：总房间数
        total_bedrooms：总卧室数
        population：人口数
        households：家庭户数
        median_income：收入中位数(万美元)
        median_house_value：房价中位数(美元)
        ocean_proximity：邻近海洋程度
            <1H OCEAN（距海洋不到1小时车程）
            INLAND（内陆）
            NEAR OCEAN（靠近海洋）
            NEAR BAY（靠近海湾）
            ISLAND（岛屿，样本极少）
    """

    def __init__(self):
        """
        初始化方法：负责读取数据并存储为实例属性
        """
        file_path = getPath("housing.csv")
        try:
            self.df = pd.read_csv(file_path, header=0)
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}，请检查路径。")
            self.df = None

    def get_full_data(self):
        """
        返回全部数据
        :return:
            返回：numpy数组
        """
        full_data = self.df.to_numpy()
        return full_data

    def get_assign_field(self, field:[]):
        """
        返回全部数据
        :return:
            返回：numpy数组
        """
        field_data = self.df[field].to_numpy()
        return field_data

class Sample_Custom:
    def __init__(self):
        file_path = getPath("二特征_二分类.csv")
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def get_binary_classification(self):

        """
        从读取的CSV中分离出特征X和标签y
        规则：最后一列作为y，其余所有列作为X
                """
        # 1. 获取特征 X（提取除最后一列之外的所有列）
        # df.iloc[:, :-1] 表示：取所有行，取从第1列到倒数第2列
        X = self.df.iloc[:, :-1].values  # 转换为 numpy 数组，shape: (n_samples, n_features)
        y = self.df.iloc[:, -1].values  # 转换为 numpy 一维数组，shape: (n_samples,)

        return X, y


if __name__ == "__main__":
    sc = Sample_Custom()
    x, y = sc.get_binary_classification()
    print(x)
    print(y)

    # iris_loader = ReadIris()
    # COLUMN_NAMES = ['petal_length', 'petal_width']
    # data_binary = iris_loader.get_data(COLUMN_NAMES,2,100)
    # print(data_binary)


    # californ = CaliforniaHousing()
    # field = ['median_income','median_house_value']
    # data = californ.get_assign_field(field)
    # print(data)

    # if data_binary is not None:
    #     print("二分类数据预览：")
    #     print(data_binary.head())
    #     print(f"当前类别分布:\n{data_binary['class'].value_counts()}")
