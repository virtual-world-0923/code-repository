# =========================
# 1. 解决 OpenMP 冲突（必须放在最前面）
# =========================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================
# 2. 导入 YOLO 与 OpenCV
# =========================
from ultralytics import YOLO
import cv2

# =========================
# 3. 加载训练好的模型
# 如果是 COCO 预训练：
# model = YOLO("yolov8s.pt")
#
# 如果是你自己训练的：
# model = YOLO("runs/detect/train/weights/best.pt")
# =========================
model = YOLO("yolov8s.pt")

# =========================
# 4. 打开笔记本摄像头
# 参数说明：
# 0 = 默认摄像头
# 如果是外接 USB 摄像头，可能需要改成 1 / 2
# =========================
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已打开，按 q 键退出")

# =========================
# 5. 实时读取帧并进行推理
# =========================
while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    # =========================
    # 6. YOLO 推理（单帧）
    # source=frame：直接传入 numpy 图像
    # conf=0.3：置信度阈值（过滤低置信度框）
    # imgsz=416：输入尺寸（950M 建议小一点）
    # device=0：使用 GPU
    # =========================
    results = model.predict(
        source=frame,
        conf=0.3,
        imgsz=416,
        device=0,
        verbose=False   # 不打印推理日志
    )
    # results = model.track(
    #     source=frame,
    #     conf=0.3,
    #     imgsz=416,
    #     device=0,
    #     persist=True,
    #     verbose=False
    # )

    # =========================
    # 7. 获取带检测框的图像
    # results[0].plot() 会自动：
    # - 画检测框
    # - 写类别名
    # - 写置信度
    # =========================
    annotated_frame = results[0].plot()

    # =========================
    # 8. 显示画面
    # 窗口名："YOLOv8 Real-time Detection"
    # =========================
    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # =========================
    # 9. 按键退出逻辑
    # 等待 1ms，如果按下 q 键则退出循环
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 退出摄像头检测")
        break

# =========================
# 10. 释放资源
# =========================
cap.release()          # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口