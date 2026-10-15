# =========================================================
# 1. 解决 Anaconda 环境中 OpenMP 运行时冲突问题
# 必须放在所有 import 之前
# =========================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================
# 2. 导入所需库
# ultralytics.YOLO ：YOLOv8 模型接口
# cv2              ：OpenCV，用于摄像头与画面显示
# =========================================================
from ultralytics import YOLO
import cv2

# =========================================================
# 3. 加载模型
# yolov8s.pt ：COCO 预训练模型，可识别 80 类
# 若使用自己训练的权重，替换为 best.pt 路径即可
# =========================================================
model = YOLO("yolov8s.pt")

# =========================================================
# 4. 打开笔记本默认摄像头
# 参数说明：
# 0 ：默认摄像头（内置）
# 1 / 2 ：外接 USB 摄像头
# =========================================================
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头已打开，按 q 退出，按 s 截图")

# =========================================================
# 5. 人数统计相关变量
# current_ids ：当前帧中所有人的 track_id 集合（自动去重）
# max_count   ：本次运行中出现过的最大人数
# frame_id    ：帧计数器
# =========================================================
current_ids = set()
max_count = 0
frame_id = 0

# =========================================================
# 6. 实时检测 + 跟踪 + 人数统计主循环
# =========================================================
while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取摄像头画面")
        break

    frame_id += 1

    # =====================================================
    # 7. YOLOv8 跟踪推理（关键步骤）
    # source=frame   ：直接传入当前帧（numpy 数组）
    # conf=0.35      ：置信度阈值，过滤误检
    # imgsz=416      ：输入尺寸，950M 显存小，建议 ≤ 416
    # device=0       ：使用 GPU（cuda:0）
    # classes=[0]    ：只检测“人”（COCO 中 person = 0）
    # persist=True   ：跨帧保持 ID 不变（人数统计必须）
    # verbose=False  ：不打印推理日志
    # =====================================================
    results = model.track(
        source=frame,
        conf=0.35,
        imgsz=416,
        device=0,
        classes=[0],          # 只检测人
        persist=True,         # 关键：ID 跨帧连续
        verbose=False
    )

    # =====================================================
    # 8. 人数统计核心逻辑
    # track_id ：同一个人在不同帧中 ID 保持不变
    # set()   ：自动去重，同一个人不会被重复计数
    # =====================================================
    current_ids.clear()  # 每帧先清空

    # results[0].boxes.id 只有在 track() 时才有
    if results[0].boxes.id is not None:
        # 将所有 track_id 加入集合
        for tid in results[0].boxes.id.int().cpu().numpy():
            current_ids.add(int(tid))

    # 当前画面中的人数
    person_count = len(current_ids)

    # 更新历史峰值
    if person_count > max_count:
        max_count = person_count

    # =====================================================
    # 9. 在画面上绘制检测框与标签
    # plot() 会自动：
    # - 画检测框
    # - 写类别名（Person）
    # - 写 track_id
    # - 写置信度
    # =====================================================
    annotated_frame = results[0].plot()

    # =====================================================
    # 10. 叠加人数统计信息文字
    # =====================================================

    # 当前人数（左上角，大字体）
    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),  # 黄色
        3
    )

    # 峰值人数（左下角，小字体）
    cv2.putText(
        annotated_frame,
        f"Max Count: {max_count}",
        (20, annotated_frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),  # 橙色
        2
    )

    # 帧编号（右上角）
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_id}",
        (annotated_frame.shape[1] - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # 白色
        2
    )

    # =====================================================
    # 11. 显示画面
    # =====================================================
    cv2.imshow("YOLOv8 People Counting", annotated_frame)

    # =====================================================
    # 12. 键盘控制逻辑
    # q ：退出程序
    # s ：截图保存
    # =====================================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 退出人数统计")
        break
    elif key == ord('s'):
        # 截图文件名包含当前人数与时间戳
        filename = f"screenshot_{person_count}_{frame_id}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"📸 截图已保存：{filename}")

# =========================================================
# 13. 释放资源（非常重要）
# =========================================================
cap.release()              # 释放摄像头
cv2.destroyAllWindows()    # 关闭所有 OpenCV 窗口

# =========================================================
# 14. 打印本次运行统计摘要
# =========================================================
print("=" * 50)
print("📊 本次会话统计")
print(f"   总帧数   : {frame_id}")
print(f"   峰值人数 : {max_count}")
print("=" * 50)