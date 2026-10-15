from ultralytics import YOLO

# 检测（默认 COCO 80 类）
m1 = YOLO("yolov8s.pt")
m2 = YOLO("yolo11s.pt")
# 分类
m3 = YOLO("yolov8s-cls.pt")

results = m2("../test_image.jpg")
results[0].show()

