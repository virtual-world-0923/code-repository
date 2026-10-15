import numpy as np       # 导入numpy库，用于数组形状变换、数值运算和argmax等操作
import keras             # 导入Keras库，用于加载保存的模型(.keras格式)并进行预测推理


def predicted_model(model, img_batch):

    # ==============================================================================
    # 第五部分：使用模型进行预测
    # ==============================================================================

    # model.predict: 使用训练好的模型对输入数据进行前向推理
    # 参数:
    #   x (numpy array 或 tf.Tensor) - 输入数据，形状(N, H, W, C)
    #     这里N=1表示只预测一张图片
    #   verbose (int) - 日志显示模式
    #     0=静默无输出(单张图片预测速度极快，无需进度条)
    #     1=显示进度条(大批量预测时使用)
    #   batch_size (int, 可选) - 预测时的批次大小(默认32)
    #     这里只有1张图片，无需指定
    # 返回:
    #   numpy.ndarray - 预测结果，形状(N, num_classes)=(1, 10)
    #     每行是一个概率分布(softmax输出)，10个值分别对应10个类别的预测概率
    #     所有概率之和等于1.0
    predictions = model.predict(img_batch, verbose=0)

    return predictions