import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# 加载自己训出来的最佳权重
model = YOLO("yolo11s.pt")

# 在验证集上算 mAP@0.5 / mAP@0.5:0.95
# metrics = model.val()
# print(metrics.box.map)  # mAP@0.5:0.95
# print(metrics.box.map50)  # mAP@0.5

# 对新图片推理，保存带框图到 runs/detect/predict/
'''
"test_imgs/"		 文件夹，批量推理所有图片
"test_imgs/001.jpg"	 单张图片
"test.mp4"			 视频文件
"0"					 摄像头（笔记本自带）
"rtsp://..."		 网络流
"https://xxx.jpg"	 网络图片
np.ndarray	OpenCV 	 读出来的数组
'''
# results_pre = model.predict(
#     source=r"../test_video.mp4", # 去哪个文件夹找图片/视频
#     batch=20,          # 一次送 8 张图片进模型 （视频只能自己攒N帧一起送入）
#     conf=0.3,  # 只保留置信度 ≥ 0.3 的检测框
#     save=True, # 把画好框的结果保存成新文件
#     imgsz = 416,
#     device = 0
# )

#等价
# results = model("../test_image.jpg")

#实时跟踪。有track_id，跨帧连续，统计人数
# results_tra = model.track(
#     source="demo.mp4",
#     persist=True, #	跨多次调用保持 ID 一致性
#     save=True) #保存带跟踪框的视频

results = model.track(
    source="../test_video.mp4",
    save=True,
    conf=0.3,
    imgsz=416,
    device=0,
    persist=True
)

'''
model.predict(
    source="../test_image.jpg",
    save=True,          # ✅ 保存结果图
    save_txt=True,      # ✅ 保存检测结果为 YOLO 格式 txt
    save_conf=True,     # ✅ txt 中带上置信度
    save_crop=True,     # ✅ 裁剪目标小图
    save_frames=True,   # ✅ 额外把每一帧存成图片(如果预测的是视频)
    stream=True,        # ✅ 长视频必加。逐帧生成器，不一次性加载，不然内存会爆
    project="runs/detect",     # 根目录
    name="my_video_result",    # 子文件夹名（不会自动加数字）
    exist_ok=True              # 允许覆盖，不再生成 predict2
)


# results 是一个 Python list。列表中的每一个元素，对应一张输入图片（或一帧视频）
re = results[0]
re.probs #分类概率（如果是 cls 模型）
re.names #类别名映射
re.orig_img #原始图像（numpy）
re.path #图像路径
boxes = re.boxes  #检测框
boxes.xyxy        # [x1, y1, x2, y2]
boxes.xywh        # [x, y, w, h]
boxes.xyxyn       # 归一化 xyxy
boxes.xywhn       # 归一化 xywh
boxes.conf        # 置信度
boxes.cls         # 类别 id
boxes.id          # track_id（仅 track 时有）


# plot() 把检测结果（框、类别、置信度）画回到原始图像上，返回一个带标注的 BGR numpy 数组
re.plot(
    conf=True,      # 是否显示置信度
    labels=True,  # 是否显示标签
    line_width=2,   # 框线粗细
    font_size=0.8  # 字体大小
)
re.show()  #弹窗显示

re.save("result.jpg") # 保存结果图

re.verbose() # 打印结果摘要
'''