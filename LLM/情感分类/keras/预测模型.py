import keras
import tensorflow as tf


def predicted_model(MODEL_PATH,vectorize_layer):

    # ================================================================
    # 9. 加载模型 → 预测一句评论
    # ================================================================
    loaded_model = keras.models.load_model(MODEL_PATH)

    test_sentence = "这家酒店环境优雅，服务态度非常好，房间干净舒适，下次还会再来！"

    test_sentence_vectorize = vectorize_layer([test_sentence])
    # 1. 看看 vectorize 之后的形状
    print(test_sentence_vectorize.shape)  # 应该显示 (1, MAX_LEN)

    # 2. 看看模型期望的输入形状
    print(loaded_model.input_shape)  # 应该显示 (None, MAX_LEN)

    pred_prob = loaded_model.predict(test_sentence_vectorize)[0][0]

    # pred_prob = loaded_model.predict([test_sentence], verbose=0)[0][0]
    sentiment = "满意 ✅" if pred_prob >= 0.5 else "不满意 ❌"

    print(f"评论: {test_sentence}")
    print((f"预测: {sentiment}   (正向置信度: {pred_prob:.4f})"))