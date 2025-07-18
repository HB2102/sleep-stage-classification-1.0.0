import tensorflow as tf
from tensorflow.keras import layers, models

def build_feature_extractor(input_shape=(60, 76, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Block 1
    x = layers.Conv2D(16, (3, 3), padding='same')(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 5
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.Activation('gelu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Final pooling and dense
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)

    model = models.Model(inputs, x, name='feature_extractor')
    return model
