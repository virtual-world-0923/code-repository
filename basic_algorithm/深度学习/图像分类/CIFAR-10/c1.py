# from tensorflow.keras.datasets import cifar10
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

import torchvision.models as models
from torchvision.models import detection, segmentation, video, optical_flow

print(models.list_models())                        # 分类
print(models.list_models(module=detection))        # 检测
print(models.list_models(module=segmentation))     # 分割
print(models.list_models(module=video))            # 视频
print(models.list_models(module=optical_flow))     # 光流