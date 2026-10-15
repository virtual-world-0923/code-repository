import keras
import os

def train_model(combined_model, x_train, y_train):

    combined_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    combined_model.build(input_shape=(None,128)) #手动触发构建，输入形状为 (batch_size,) 的字符串
    combined_model.summary()
    print("-" * 60)

    history = combined_model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )
    return combined_model, history
