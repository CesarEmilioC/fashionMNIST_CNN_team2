"""
Full training pipeline — runs all four models sequentially and saves:
  - Model weights      -> models/saved/
  - Training histories -> results/metrics/
  - Figures            -> results/figures/

Usage (local):
    python scripts/train_all.py --data-dir /path/to/csv/dir
    python scripts/train_all.py  # downloads from Kaggle automatically
"""

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive — saves figures without opening windows
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf

from src.data.loader import (
    CLASS_NAMES,
    load_fashion_mnist_from_csv,
    load_fashion_mnist_from_kaggle,
    preprocess_data,
)
from src.models.baseline_cnn import build_baseline_cnn
from src.models.improved_cnn import build_improved_cnn
from src.models.mini_resnet import build_mini_resnet
from src.training.trainer import (
    get_cosine_decay,
    get_improved_callbacks,
    get_resnet_callbacks,
    save_history,
)
from src.utils.evaluation import build_results_table, evaluate_model, get_confusion_matrix
from src.utils.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_accuracy,
    plot_results_bars,
    plot_sample_images,
    plot_sample_predictions,
    plot_training_history,
)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Reproducibility ENABLED — SEED = {seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fashion MNIST full training pipeline")
    parser.add_argument("--data-dir", default=None,
                        help="Path to directory with fashion-mnist CSVs. "
                             "If omitted, downloads from Kaggle via kagglehub.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU available: {gpus}")
    if not gpus:
        print("WARNING: No GPU detected. Training will be slow on CPU.")

    FIGURES_DIR = os.path.join(ROOT, "results", "figures")
    METRICS_DIR = os.path.join(ROOT, "results", "metrics")
    MODELS_DIR  = os.path.join(ROOT, "models", "saved")
    for d in (FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.data_dir:
        train_df, test_df = load_fashion_mnist_from_csv(args.data_dir)
    else:
        train_df, test_df = load_fashion_mnist_from_kaggle()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        train_df, test_df, seed=args.seed
    )

    plot_class_distribution(y_train, y_test, CLASS_NAMES,
                            save_path=os.path.join(FIGURES_DIR, "eda_class_distribution.png"))
    plot_sample_images(X_train, y_train, CLASS_NAMES,
                       save_path=os.path.join(FIGURES_DIR, "eda_sample_images.png"))

    # ── Model 1: Baseline CNN ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Model 1 — Baseline CNN")
    print("=" * 60)
    baseline_model = build_baseline_cnn()
    baseline_model.summary()

    t0 = time.time()
    baseline_history = baseline_model.fit(
        X_train, y_train, epochs=25, batch_size=64,
        validation_data=(X_val, y_val), verbose=1
    )
    baseline_time = time.time() - t0
    print(f"Training time: {baseline_time:.1f}s")

    plot_training_history(baseline_history, title="Model 1: Baseline CNN",
                          save_path=os.path.join(FIGURES_DIR, "history_baseline_cnn.png"))
    baseline_acc, baseline_loss, baseline_f1, baseline_preds, _ = evaluate_model(
        baseline_model, X_test, y_test, "Model 1: Baseline CNN", CLASS_NAMES
    )
    cm_b = get_confusion_matrix(y_test, baseline_preds)
    plot_confusion_matrix(cm_b, CLASS_NAMES, "Model 1: Baseline CNN",
                          save_path=os.path.join(FIGURES_DIR, "cm_baseline_cnn.png"))
    baseline_model.save(os.path.join(MODELS_DIR, "baseline_cnn.h5"))
    save_history(baseline_history, os.path.join(METRICS_DIR, "history_baseline_cnn.json"))

    # ── Model 2: Improved CNN ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Model 2 — Improved CNN")
    print("=" * 60)
    improved_model = build_improved_cnn()
    improved_model.summary()

    t0 = time.time()
    improved_history = improved_model.fit(
        X_train, y_train, epochs=50, batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=get_improved_callbacks(), verbose=1
    )
    improved_time = time.time() - t0
    print(f"Training time: {improved_time:.1f}s")

    plot_training_history(improved_history, title="Model 2: Improved CNN",
                          save_path=os.path.join(FIGURES_DIR, "history_improved_cnn.png"))
    improved_acc, improved_loss, improved_f1, improved_preds, _ = evaluate_model(
        improved_model, X_test, y_test, "Model 2: Improved CNN", CLASS_NAMES
    )
    cm_i = get_confusion_matrix(y_test, improved_preds)
    plot_confusion_matrix(cm_i, CLASS_NAMES, "Model 2: Improved CNN",
                          save_path=os.path.join(FIGURES_DIR, "cm_improved_cnn.png"))
    improved_model.save(os.path.join(MODELS_DIR, "improved_cnn.h5"))
    save_history(improved_history, os.path.join(METRICS_DIR, "history_improved_cnn.json"))

    # ── Model 3: Mini-ResNet ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Model 3 — Mini-ResNet")
    print("=" * 60)
    EPOCHS_RESNET = 40
    BATCH_RESNET  = 128

    resnet_model = build_mini_resnet()
    cosine_lr = get_cosine_decay(1e-3, EPOCHS_RESNET, len(X_train), BATCH_RESNET)
    resnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    resnet_model.summary()

    t0 = time.time()
    resnet_history = resnet_model.fit(
        X_train, y_train, epochs=EPOCHS_RESNET, batch_size=BATCH_RESNET,
        validation_data=(X_val, y_val),
        callbacks=get_resnet_callbacks(patience=10), verbose=1
    )
    resnet_time = time.time() - t0
    print(f"Training time: {resnet_time:.1f}s ({resnet_time / 60:.1f} min)")

    plot_training_history(resnet_history, title="Model 3: Mini-ResNet",
                          save_path=os.path.join(FIGURES_DIR, "history_mini_resnet.png"))
    resnet_acc, resnet_loss, resnet_f1, resnet_preds, _ = evaluate_model(
        resnet_model, X_test, y_test, "Model 3: Mini-ResNet", CLASS_NAMES
    )
    cm_r = get_confusion_matrix(y_test, resnet_preds)
    plot_confusion_matrix(cm_r, CLASS_NAMES, "Model 3: Mini-ResNet",
                          save_path=os.path.join(FIGURES_DIR, "cm_mini_resnet.png"))
    resnet_model.save(os.path.join(MODELS_DIR, "mini_resnet.h5"))
    save_history(resnet_history, os.path.join(METRICS_DIR, "history_mini_resnet.json"))

    # ── Hyperparameter search + Final tuned model ────────────────────────────
    print("\n" + "=" * 60)
    print("Running hyperparameter search on Mini-ResNet")
    print("=" * 60)
    from scripts.hyperparameter_search import run_search
    best_config, hp_df = run_search(
        X_train, y_train, X_val, y_val, screening_epochs=20, seed=args.seed
    )
    hp_df.to_csv(os.path.join(METRICS_DIR, "hyperparameter_search.csv"), index=False)
    print(f"Best config: {best_config}")

    EPOCHS_FINAL = 60
    final_model = build_mini_resnet(dropout_rate=best_config["dropout"])
    final_cosine_lr = get_cosine_decay(
        best_config["lr"], EPOCHS_FINAL, len(X_train), best_config["batch_size"]
    )
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=final_cosine_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0 = time.time()
    final_history = final_model.fit(
        X_train, y_train,
        epochs=EPOCHS_FINAL,
        batch_size=best_config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=get_resnet_callbacks(patience=10),
        verbose=1,
    )
    final_time = time.time() - t0
    print(f"Training time: {final_time:.1f}s ({final_time / 60:.1f} min)")

    plot_training_history(final_history, title="Final Model: Tuned Mini-ResNet",
                          save_path=os.path.join(FIGURES_DIR, "history_tuned_resnet.png"))
    final_acc, final_loss, final_f1, final_preds, final_probs = evaluate_model(
        final_model, X_test, y_test, "Final Model: Tuned Mini-ResNet", CLASS_NAMES
    )
    cm_f = get_confusion_matrix(y_test, final_preds)
    plot_confusion_matrix(cm_f, CLASS_NAMES, "Final Model: Tuned Mini-ResNet",
                          save_path=os.path.join(FIGURES_DIR, "cm_tuned_resnet.png"))
    plot_per_class_accuracy(cm_f, CLASS_NAMES, "Tuned Mini-ResNet",
                            save_path=os.path.join(FIGURES_DIR, "per_class_accuracy.png"))
    final_model.save(os.path.join(MODELS_DIR, "tuned_mini_resnet.h5"))
    save_history(final_history, os.path.join(METRICS_DIR, "history_tuned_resnet.json"))
    print(f"Best model saved -> {os.path.join(MODELS_DIR, 'tuned_mini_resnet.h5')}")

    # ── Comparison plots ─────────────────────────────────────────────────────
    plot_model_comparison(
        [baseline_history, improved_history, resnet_history, final_history],
        ["Baseline CNN", "Improved CNN", "Mini-ResNet", "Tuned Mini-ResNet"],
        save_path=os.path.join(FIGURES_DIR, "comparison_all_models.png"),
    )
    plot_sample_predictions(
        X_test, y_test, final_preds, np.max(final_probs, axis=1),
        CLASS_NAMES, "Tuned Mini-ResNet",
        save_path=os.path.join(FIGURES_DIR, "sample_predictions.png"),
    )

    model_entries = [
        {"model_name": "Baseline CNN",      "model_obj": baseline_model, "test_acc": baseline_acc, "test_loss": baseline_loss, "f1": baseline_f1, "train_time_s": baseline_time},
        {"model_name": "Improved CNN",      "model_obj": improved_model, "test_acc": improved_acc, "test_loss": improved_loss, "f1": improved_f1, "train_time_s": improved_time},
        {"model_name": "Mini-ResNet",       "model_obj": resnet_model,   "test_acc": resnet_acc,   "test_loss": resnet_loss,   "f1": resnet_f1,   "train_time_s": resnet_time},
        {"model_name": "Tuned Mini-ResNet", "model_obj": final_model,    "test_acc": final_acc,    "test_loss": final_loss,    "f1": final_f1,    "train_time_s": final_time},
    ]
    results_df = build_results_table(model_entries)
    results_df.to_csv(os.path.join(METRICS_DIR, "results_summary.csv"), index=False)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    plot_results_bars(results_df, colors,
                      save_path=os.path.join(FIGURES_DIR, "results_bar_comparison.png"))

    print("\nAll models trained. Results saved to results/ and models/saved/")


if __name__ == "__main__":
    main()
