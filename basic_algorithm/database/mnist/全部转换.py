import os
import struct
import numpy as np
from PIL import Image
import csv

# ================= 配置区域 =================
# 设置你要保存结果的根目录
OUTPUT_DIR = "MNIST_Converted_Data"

# 确保输入文件名与你的文件一致（如果是在当前目录下）
FILE_CONFIG = {
    "train_img": "train-images-idx3-ubyte",
    "train_lbl": "train-labels-idx1-ubyte",
    "test_img": "t10k-images-idx3-ubyte",
    "test_lbl": "t10k-labels-idx1-ubyte"
}


# ===========================================

def load_idx(path, is_image=True):
    """解析 IDX 格式的二进制文件"""
    print(f"正在读取: {path} ...")
    with open(path, 'rb') as f:
        if is_image:
            # 图像格式头: 魔数(4), 数量(4), 行(4), 列(4)
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        else:
            # 标签格式头: 魔数(4), 数量(4)
            magic, num = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def convert_images(img_path, lbl_path, save_folder, prefix):
    """将图像转为 PNG，标签转为 CSV"""
    os.makedirs(save_folder, exist_ok=True)

    images = load_idx(img_path, is_image=True)
    labels = load_idx(lbl_path, is_image=False)

    # 1. 导出标签为 CSV (Excel 可读)
    csv_path = os.path.join(save_folder, f"{prefix}_labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Label'])  # 表头
        for i, label in enumerate(labels):
            writer.writerow([i, label])
    print(f"[完成] 标签已保存至: {csv_path}")

    # 2. 导出图像为 PNG
    # 注意：60000张图片体积较大，这里演示批量保存
    img_save_dir = os.path.join(save_folder, f"{prefix}_images")
    os.makedirs(img_save_dir, exist_ok=True)

    count = 0
    for i, img_data in enumerate(images):
        # MNIST 是灰度图，模式为 'L'
        img = Image.fromarray(img_data, mode='L')
        # 为了不让文件夹里文件太多导致卡顿，可以按标签分类保存，或者直接平铺
        # 这里直接平铺保存，文件名格式: 0_00001.png (标签_索引.png)
        filename = f"{labels[i]}_{i:05d}.png"
        img.save(os.path.join(img_save_dir, filename))
        count += 1

    print(f"[完成] {count} 张 {prefix} 图片已保存至: {img_save_dir}")


if __name__ == "__main__":
    print("开始转换 MNIST 数据集...")

    # 创建总输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 处理训练集
        convert_images(
            FILE_CONFIG["train_img"],
            FILE_CONFIG["train_lbl"],
            os.path.join(OUTPUT_DIR, "Training_Set"),
            "train"
        )

        # 处理测试集
        convert_images(
            FILE_CONFIG["test_img"],
            FILE_CONFIG["test_lbl"],
            os.path.join(OUTPUT_DIR, "Test_Set"),
            "test"
        )

        print("\n所有任务已完成！请查看文件夹:", os.path.abspath(OUTPUT_DIR))

    except FileNotFoundError as e:
        print(f"\n错误: 找不到文件 - {e}")
        print("请确保脚本与数据文件在同一目录下，或修改 FILE_CONFIG 中的路径。")