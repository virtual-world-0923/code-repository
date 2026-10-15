import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 加载模型结构（不加载权重）,从头训练
# model = YOLO("yolov8s.yaml")

# 加载预训练权重做微调
model = YOLO("yolov8s.pt")

# 开始训练
# data: 数据集 yaml 路径
# epochs: 跑多少轮完整数据
# imgsz: 输入尺寸 640 是 YOLOv8 默认
# batch: 一次几张图，-1 表示按显存自动
# device: 0 用第 0 号 GPU；"cpu" 用 CPU
results = model.train(
    data=r"C:\Repositories\basic_algorithm\深度学习\YOLO\coco8\coco8.yaml",
    epochs=10,        # COCO8 很小，50 足够
    imgsz=416,        # 950M 显存小，降分辨率
    batch=4,          # 先试 4，不行再降到 2 或 1
    device=0,
    workers=0,        # Windows 建议 0
    cache=False       # 小数据不开 cache
)

# 训练完最佳权重在 runs/detect/train/weights/best.pt