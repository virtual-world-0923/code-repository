import tensorflow as tf
from keras import layers, models
import keras
import numpy as np

preprocess = keras.Sequential([
    layers.Input(shape=(3, 32, 32)),                     # BCHW
    layers.Permute((2, 3, 1), name='BCHW_to_BHWC'),      # → BHWC
    layers.Resizing(227, 227),                            # → 227×227
    layers.Rescaling(1.0 / 255.0),                        # → 0~1
    layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],   # variance = std²
    ),
    #或 使用lambda,加载保存的模型时，Keras默认禁止反序列化 Lambda 层，设置参数 keras.models.load_model(model_save_path, safe_mode=False)
    # layers.Lambda(
    #     lambda x: (x - tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)) /
    #               tf.constant([0.229, 0.224, 0.225], dtype=tf.float32),
    #     name='normalize_imagenet'
    # ),
], name='preprocess')
