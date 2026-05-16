import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(frames=8, img_size=64):
    inputs = layers.Input(shape=(frames, img_size, img_size, 3))

    # CNN part - extracts features from each frame
    cnn_base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    cnn_base.trainable = False

    # Apply CNN to each frame
    cnn_out = layers.TimeDistributed(cnn_base)(inputs)

    # LSTM part - learns patterns across frames
    x = layers.LSTM(64, return_sequences=False)(cnn_out)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, x)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()