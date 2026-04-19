from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


def build_improved_cnn(l2_reg=1e-4, dropout_conv=0.25, dropout_dense=0.5):
    """
    Model 2 — Regularized CNN.
    Adds BatchNorm, Dropout, data augmentation, L2 weight decay, Adam optimizer,
    and LR scheduling over the baseline. Three conv blocks with increasing filters
    (32 -> 64 -> 128). Augmentation is the first layer, active only during training.
    """
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Data augmentation (active only during training)
        data_augmentation,

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_conv),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_conv),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_conv),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_dense),
        layers.Dense(10, activation='softmax')
    ], name='Improved_CNN')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
