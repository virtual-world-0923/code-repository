import os

import pandas as pd
from datasets import Dataset as HFDataset
from datasets import ClassLabel,Value
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import project_config as pc

def load_data(path: str) -> pd.DataFrame:
    """
    读取 CSV 并做基础清洗
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[["label", "review"]].dropna()
    df["label"] = df["label"].astype(int)

    first_half = df.head(50)
    second_half = df.tail(50)
    return pd.concat([first_half, second_half], ignore_index=True)


def tokenize(batch, tokenizer, max_len=pc.MAX_LEN):
    """
    Hugging Face Dataset map 用的 tokenizer 函数
    """
    return tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None,   # map 内部会处理
    )


def get_data_loader():
    tokenizer = AutoTokenizer.from_pretrained(pc.MODEL_NAME_BERT)
    # 1. 读 CSV
    df = load_data(pc.CSV_PATH)
    print(f"[数据] 总量: {len(df)}")
    print(df["label"].value_counts())

    # features = {
    #     "text": Value("string"),
    #     "label": ClassLabel(names=["不满意", "满意"], num_classes=2)
    # }
    # df["label"] = df["label"].astype(int)
    # dataset = HFDataset.from_pandas(df, features=features)

    # 2. pandas → HFDataset
    ds = HFDataset.from_pandas(df)

    # 3. 声明 label 为 ClassLabel
    # 把 Dataset 中名为 "label" 的这一列，从普通的数值列（Value(dtype='int64')），显式声明为“分类标签列”（ClassLabel），并规定它只有两个类别：0=不满意，1=满意。
    ds = ds.cast_column(
        "label",
        ClassLabel(names=["不满意", "满意"], num_classes=2)
    )

    # 4. 分层切分
    split = ds.train_test_split( # 按照 label 列分割，label列必须为ClassLabel类型
        test_size=0.2,
        seed=pc.SEED,
        stratify_by_column="label"
    )

    train_ds = split["train"]
    val_ds = split["test"]

    print(f"[划分] train={len(train_ds)}  val={len(val_ds)}")

    # 5. ✅ 批量 tokenizer
    # train_ds.map函数只会调用 f(batch),传一个batch参数进去，但自己写的tokenize函数是两个参数，所以需要lambda定义匿名函数传两个参数进去
    train_ds = train_ds.map(
        lambda batch: tokenize(batch, tokenizer), #把这批数据送入tokenize函数处理
        batched=True, # 每次传一批数据
        # num_proc=os.cpu_count(), #多CPU进程
        desc="Tokenizing train" #度条设置一个描述文字
    )
    val_ds = val_ds.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        # num_proc=os.cpu_count(),
        desc="Tokenizing val"
    )

    # 6. ✅ 告诉 HF：__getitem__ 返回 torch tensor
    train_ds = train_ds.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    val_ds = val_ds.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 7. ✅ 直接进 PyTorch DataLoader（不需要任何自定义 Dataset）
    train_loader = DataLoader(
        train_ds,
        batch_size=pc.BATCH_SIZE,
        # num_workers=2, #让 CPU 多线程并行准备数据。启动 4 个子进程，并行地把数据从 Dataset 读出来、tokenizer、拼成 batch，然后交给 GPU
        # pin_memory=True, # 让 CPU → GPU 的数据拷贝更快
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=pc.BATCH_SIZE,
        # num_workers=2,
        # pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader,train_ds,val_ds,tokenizer