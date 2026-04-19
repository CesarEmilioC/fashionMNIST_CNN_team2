from tensorflow import keras
from tensorflow.keras import layers, models


def build_baseline_cnn():
    """
    Model 1 — Bare-bones CNN with no regularization.
    Serves as a reference baseline: 2 conv blocks + dense head, SGD optimizer.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name='Baseline_CNN')

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
