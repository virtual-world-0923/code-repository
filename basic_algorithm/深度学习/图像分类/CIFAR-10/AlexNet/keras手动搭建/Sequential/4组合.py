import keras                  # 导入独立Keras 3.x库，用于构建网络模型、编译训练和保存加载
from keras import layers      # 从Keras中导入layers模块，提供Conv2D、Dense、MaxPooling2D等网络层定义
import runpy

BATCH_SIZE = 64
EPOCHS = 1

result = runpy.run_path("2数据处理网络.py")
preprocess = result["preprocess"]

result = runpy.run_path("3主网络.py")
backbone = result["backbone"]

#方式一
def combination_one():
    #建图，定义数据处理流程，定义数据处理流水线，没有真实数据，只建模，在fit，fit里validation_data参数，evaluate，predict，数据走模型的时候按照这个流程图处理数据
    inputs = keras.Input(shape=(3, 32, 32))
    data_processed = preprocess(inputs)
    model_returned = backbone(data_processed)
    #告诉 Keras："这个模型的入口是 inputs，出口是 outputs，请把中间经过的所有层连成一张计算图。"
    combined_model = keras.Model(inputs=inputs, outputs=model_returned, name='alexnet_full')
    return combined_model

#方式二
def combination_two():
    combined_model = keras.Sequential([preprocess, backbone]) #把两个sequential组合到一起，顺序执行，自动判断入口和出口
    return combined_model
