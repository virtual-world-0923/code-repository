import os
import struct
import csv
import math
import numpy as np
from PIL import Image

# ================= 配置区域 =================
OUTPUT_DIR = "MNIST_Windows_View"  # 输出文件夹名称
SPRITE_SIZE = 28  # 单张图片尺寸 (28x28)
GRID_WIDTH = 100  # 概览图每行显示的图片数量 (可根据屏幕宽度调整)


# ===========================================

def load_idx(path, is_image=True):
    """解析 IDX 格式二进制文件"""
    print(f"正在读取: {os.path.basename(path)} ...")
    with open(path, 'rb') as f:
        if is_image:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        else:
            magic, num = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def create_sprite_sheet(images, output_path):
    """将多张图片拼接成一张大网格图"""
    n = len(images)
    cols = GRID_WIDTH
    rows = math.ceil(n / cols)

    # 创建空白画布
    sprite = Image.new('L', (cols * SPRITE_SIZE, rows * SPRITE_SIZE), color=255)

    for i, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        x = (i % cols) * SPRITE_SIZE
        y = (i // cols) * SPRITE_SIZE
        sprite.paste(img, (x, y))

    sprite.save(output_path)
    print(f"已生成概览图: {output_path} (尺寸: {sprite.size})")


def save_to_csv(labels, output_path, prefix=""):
    """保存标签为 CSV 格式"""
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Label', 'Description'])
        for i, label in enumerate(labels):
            writer.writerow([f"{prefix}_{i}", label, f"Digit {label}"])
    print(f"已生成标签表: {output_path}")


def main():
    # 检查文件是否存在
    files = [
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    ]
    if not all(os.path.exists(f) for f in files):
        print("错误：请在当前目录下放齐 4 个 MNIST 原始文件！")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 处理训练集
    train_imgs = load_idx("train-images-idx3-ubyte")
    train_lbls = load_idx("train-labels-idx1-ubyte", is_image=False)

    # 生成训练集概览图 (取前 10000 张以免文件过大，或去掉 [:10000] 生成全量)
    # 注意：全量 60000 张生成的图片会非常大，建议先预览
    create_sprite_sheet(train_imgs[:10000], os.path.join(OUTPUT_DIR, "Train_Sample_Overview.png"))
    save_to_csv(train_lbls, os.path.join(OUTPUT_DIR, "Train_Labels.csv"), prefix="train")

    # 2. 处理测试集
    test_imgs = load_idx("t10k-images-idx3-ubyte")
    test_lbls = load_idx("t10k-labels-idx1-ubyte", is_image=False)

    create_sprite_sheet(test_imgs, os.path.join(OUTPUT_DIR, "Test_Full_Overview.png"))
    save_to_csv(test_lbls, os.path.join(OUTPUT_DIR, "Test_Labels.csv"), prefix="test")

    print("\n完成！请打开 'MNIST_Windows_View' 文件夹查看结果。")


if __name__ == "__main__":
    main()