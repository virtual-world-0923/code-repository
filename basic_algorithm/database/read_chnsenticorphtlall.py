import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold


def read_data():
    current_dir = Path(__file__).resolve().parent

    DATA_PATH = current_dir / "ChnSentiCorp_htl_all.csv"
    df = pd.read_csv(DATA_PATH)

    # 清洗空值和异常值
    df = df.dropna(subset=['review']).reset_index(drop=True)
    df = df[df['review'].str.strip() != ''].reset_index(drop=True)

    texts  = df["review"].to_numpy()
    labels = df["label"].to_numpy()
    return labels,texts

def qiefen1():
    labels,texts = read_data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # 保持正负样本比例一致
    )

def qiefen2():
    labels,texts = read_data()
    df = pd.DataFrame()
    df['text'] = texts
    df['label'] = labels
    # 把数据集分成 5 份（5-fold），每次用 1 份当验证集，其余 4 份当训练集，并保证每一份里正负样本比例和整体一致。
    # 这就是 5 折分层交叉验证。
    # 每一折里，类别比例 ≈ 整体类别比例
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    generator = skf.split(texts, labels)

    for fold, (train_idx, val_idx) in enumerate(generator):
        train_df = texts[train_idx]
        val_df = texts[val_idx]
        train_lab = labels[train_idx]
        val_lab = labels[val_idx]
        print()

def qiefen3():
    labels,texts = read_data()
    df = pd.DataFrame()
    df['text'] = texts
    df['label'] = labels
    # sample：不放回随机抽样,frac控制抽取比例，1就是100%，0.5就是50%
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.iloc[:5000]
    val_df = df.iloc[5000:]


