import json
import os
from tensorflow.keras import callbacks, optimizers


def get_improved_callbacks():
    """ReduceLROnPlateau + EarlyStopping for the Improved CNN."""
    return [
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
        ),
    ]


def get_resnet_callbacks(patience=10):
    """EarlyStopping for Mini-ResNet variants."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
        ),
    ]


def get_cosine_decay(initial_lr, epochs, n_train, batch_size, alpha=1e-6):
    """Cosine decay schedule for learning rate."""
    decay_steps = epochs * (n_train // batch_size)
    return optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=alpha,
    )


def save_history(history, path):
    """Persist Keras History object as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"History saved -> {path}")


def load_history_dict(path):
    """Load a previously saved history JSON."""
    with open(path, 'r') as f:
        return json.load(f)
