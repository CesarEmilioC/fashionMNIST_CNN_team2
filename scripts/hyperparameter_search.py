"""
Manual grid search over Mini-ResNet hyperparameters.

Can be imported as a module (run_search) or executed standalone:
    python scripts/hyperparameter_search.py [--data-dir PATH] [--epochs 20]
"""

import argparse
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks


def _configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        target = gpus[-1]
        tf.config.set_visible_devices(target, "GPU")
        tf.config.experimental.set_memory_growth(target, True)

from src.models.mini_resnet import build_mini_resnet

HP_CONFIGS = [
    {"lr": 1e-2, "batch_size": 64,  "dropout": 0.4},
    {"lr": 1e-3, "batch_size": 32,  "dropout": 0.3},
    {"lr": 1e-3, "batch_size": 64,  "dropout": 0.4},
    {"lr": 1e-3, "batch_size": 64,  "dropout": 0.5},
    {"lr": 1e-3, "batch_size": 128, "dropout": 0.4},
    {"lr": 1e-4, "batch_size": 64,  "dropout": 0.4},
    {"lr": 1e-3, "batch_size": 64,  "dropout": 0.3},
    {"lr": 1e-4, "batch_size": 32,  "dropout": 0.5},
]


def run_search(X_train, y_train, X_val, y_val,
               configs=None, screening_epochs=20, seed=42):
    """
    Evaluate each configuration for `screening_epochs` epochs.
    Returns (best_config_dict, results_DataFrame) sorted by val accuracy.
    """
    if configs is None:
        configs = HP_CONFIGS

    _configure_gpu()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    results = []
    total = len(configs)
    print(f"Grid search: {total} configs × {screening_epochs} epochs")
    print("=" * 70)

    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{total}] LR={cfg['lr']}, BS={cfg['batch_size']}, Dropout={cfg['dropout']}")

        model = build_mini_resnet(dropout_rate=cfg["dropout"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train, y_train,
            epochs=screening_epochs,
            batch_size=cfg["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                               restore_best_weights=True)],
            verbose=0,
        )
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"   → Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        results.append({**cfg, "val_accuracy": val_acc, "val_loss": val_loss})
        tf.keras.backend.clear_session()

    df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    print("\nResults (sorted by val accuracy):")
    print(df.to_string())

    best = df.iloc[0][["lr", "batch_size", "dropout"]].to_dict()
    best["batch_size"] = int(best["batch_size"])
    return best, df


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for Mini-ResNet")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.data.loader import (
        load_fashion_mnist_from_csv,
        load_fashion_mnist_from_kaggle,
        preprocess_data,
    )

    if args.data_dir:
        train_df, test_df = load_fashion_mnist_from_csv(args.data_dir)
    else:
        train_df, test_df = load_fashion_mnist_from_kaggle()

    X_train, X_val, _, y_train, y_val, _ = preprocess_data(train_df, test_df, seed=args.seed)

    best, df = run_search(X_train, y_train, X_val, y_val,
                          screening_epochs=args.epochs, seed=args.seed)
    print(f"\nBest configuration: {best}")

    out = os.path.join(ROOT, "results", "metrics", "hyperparameter_search.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=True)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
