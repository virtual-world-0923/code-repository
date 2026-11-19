import os
import csv  # 新增：引入 csv 模块用于规范写入
import struct
import numpy as np
from PIL import Image

# ================= 配置区域 =================
# 输出根目录
OUTPUT_DIR = "MNIST_Sampled_Data"

# 输入文件名 (请确保这些文件在当前目录下)
FILES = {
    "train_img": "train-images-idx3-ubyte",
    "train_lbl": "train-labels-idx1-ubyte",
    "test_img": "t10k-images-idx3-ubyte",
    "test_lbl": "t10k-labels-idx1-ubyte"
}

# 抽样数量设置
SAMPLE_COUNTS = {
    "train": 1000,
    "test": 100
}
# ===========================================

def load_idx(path, is_image=True):
    """解析 IDX 格式二进制文件"""
    print(f"正在读取: {path}")
    with open(path, 'rb') as f:
        if is_image:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        else:
            magic, num = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def process_dataset(img_file, lbl_file, sample_count, subset_name):
    """处理并保存指定子集的数据"""
    # 1. 加载全部数据
    images = load_idx(img_file, is_image=True)
    labels = load_idx(lbl_file, is_image=False)

    total = len(images)
    print(f"[{subset_name}] 原始数据量: {total} 张")

    # 2. 随机生成索引 (不重复)
    k = min(sample_count, total)
    indices = np.random.choice(total, k, replace=False)

    # 3. 创建输出文件夹
    img_dir = os.path.join(OUTPUT_DIR, subset_name, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 4. 保存图片和记录标签
    label_records = []
    print(f"[{subset_name}] 正在保存 {k} 张随机图片到: {img_dir} ...")

    for idx in indices:
        img_array = images[idx]
        label_val = int(labels[idx])

        # 保存图片: 格式严格为 "原始索引.png"
        filename = f"{idx}.png"
        img_path = os.path.join(img_dir, filename)

        # 使用 PIL 保存灰度图
        Image.fromarray(img_array, mode='L').save(img_path)

        # 记录标签信息
        label_records.append([idx, label_val])

    # 5. 保存标签清单为 CSV 格式
    csv_file = os.path.join(OUTPUT_DIR, subset_name, "labels_summary.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头：第一列 Original_Index，第二列 Label
        writer.writerow(["Original_Index", "Label"])
        # 写入数据
        writer.writerows(label_records)

    print(f"[{subset_name}] 完成! 标签摘要已保存至: {csv_file}\n")

if __name__ == "__main__":
    # 检查文件是否存在
    missing_files = [f for f in FILES.values() if not os.path.exists(f)]
    if missing_files:
        print(f"错误: 找不到以下文件: {missing_files}")
        print("请确保脚本与数据文件在同一目录下。")
    else:
        # 执行转换
        process_dataset(FILES["train_img"], FILES["train_lbl"], SAMPLE_COUNTS["train"], "train_sampled")
        process_dataset(FILES["test_img"], FILES["test_lbl"], SAMPLE_COUNTS["test"], "test_sampled")

        print("=" * 40)
        print("所有任务已完成！")
        print(f"结果保存在: {os.path.abspath(OUTPUT_DIR)}")