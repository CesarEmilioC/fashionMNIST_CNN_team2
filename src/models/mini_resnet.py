from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


def _residual_block(x, filters, stride=1, l2_reg=1e-4):
    """Two conv layers with a skip connection. Adjusts dimensions via 1x1 conv when needed."""
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_mini_resnet(l2_reg=1e-4, dropout_rate=0.4):
    """
    Model 3 — Mini-ResNet adapted for 28x28 grayscale images (Functional API).

    Architecture:
      - Data augmentation (active only during training)
      - Initial conv (64 filters)
      - Stage 1: 2 residual blocks @ 64 filters, 28x28
      - Stage 2: 2 residual blocks @ 128 filters, 14x14 (stride-2 downsampling)
      - Stage 3: 2 residual blocks @ 256 filters, 7x7  (stride-2 downsampling)
      - GlobalAveragePooling -> Dense(10, softmax)

    Skip connections mitigate vanishing gradients and act as an implicit regularizer.
    GlobalAveragePooling replaces Flatten+Dense to cut parameters and reduce overfitting.
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')

    inputs = layers.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = _residual_block(x, 64,  stride=1, l2_reg=l2_reg)
    x = _residual_block(x, 64,  stride=1, l2_reg=l2_reg)
    x = layers.Dropout(dropout_rate / 2)(x)

    x = _residual_block(x, 128, stride=2, l2_reg=l2_reg)
    x = _residual_block(x, 128, stride=1, l2_reg=l2_reg)
    x = layers.Dropout(dropout_rate / 2)(x)

    x = _residual_block(x, 256, stride=2, l2_reg=l2_reg)
    x = _residual_block(x, 256, stride=1, l2_reg=l2_reg)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='Mini_ResNet')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
