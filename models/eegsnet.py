import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, BatchNormalization,
    MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,
    Dense, TimeDistributed, Bidirectional, LSTM, Dropout
)


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

def build_feature_extractor(input_shape=(60, 76, 3)):
    inputs = Input(shape=input_shape)
    x = inputs

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('gelu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64)(x)

    return models.Model(inputs, x, name='feature_extractor')


def build_eegsnet(sequence_length=10, input_shape=(60, 76, 3), num_classes=5):
    cnn_model = build_feature_extractor(input_shape)

    inputs = Input(shape=(sequence_length, *input_shape))  

    x = TimeDistributed(cnn_model)(inputs)  

    aux_output = TimeDistributed(Dense(num_classes, activation='softmax'), name='aux_output')(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)  
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)  
    x = Dropout(0.5)(x)

    outputs = TimeDistributed(Dense(num_classes, activation='softmax'), name='main_output')(x)

    model = models.Model(inputs=inputs, outputs=[outputs, aux_output], name='EEGSNet')
    return model
