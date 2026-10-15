import ultralytics, shutil, os
src = os.path.join(os.path.dirname(ultralytics.__file__),
                   "cfg/datasets/coco8.yaml")
dst = "/深度学习/YOLO/coco8/coco8.yaml"
shutil.copy(src, dst)
print("已复制，改 path 为你的绝对路径即可")