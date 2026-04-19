import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history, title='Training History', save_path=None):
    """Accuracy and loss curves for training and validation splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title(f'{title} — Accuracy', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title(f'{title} — Loss', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_class_distribution(y_train, y_test, class_names, save_path=None):
    """Bar chart of label counts for training and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, data, title in zip(axes, [y_train, y_test], ['Training Set', 'Test Set']):
        unique, counts = np.unique(data, return_counts=True)
        bars = ax.bar([class_names[i] for i in unique], counts, color=colors)
        ax.set_title(f'Class Distribution — {title}', fontsize=13)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 50,
                    str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_sample_images(X, y, class_names, save_path=None):
    """One sample image per class."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        idx = np.where(y == i)[0][0]
        ax.imshow(X[idx].squeeze(), cmap='gray')
        ax.set_title(class_names[i], fontsize=11)
        ax.axis('off')
    plt.suptitle('Sample Images — One per Class', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_average_class_images(X, y, class_names, save_path=None):
    """Mean image per class displayed as a heat map."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for i, ax in enumerate(axes.flat):
        mean_img = X[y == i].mean(axis=0).squeeze()
        ax.imshow(mean_img, cmap='hot')
        ax.set_title(class_names[i], fontsize=10)
        ax.axis('off')
    plt.suptitle('Average Image per Class (Heat Map)', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_model_comparison(histories, model_names, save_path=None):
    """Overlaid val accuracy and val loss curves for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    line_styles = ['-', '--', '-.', ':']

    for hist, name, ls, color in zip(histories, model_names, line_styles, colors):
        ax1.plot(hist.history['val_accuracy'], label=name, linestyle=ls, linewidth=2, color=color)
        ax2.plot(hist.history['val_loss'], label=name, linestyle=ls, linewidth=2, color=color)

    ax1.set_title('Validation Accuracy Comparison', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Loss Comparison', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name='Model', save_path=None):
    """Annotated confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} — Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_per_class_accuracy(cm, class_names, model_name='Model', save_path=None):
    """Horizontal bar chart of per-class accuracy."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(class_names, per_class_acc, color=plt.cm.RdYlGn(per_class_acc))
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Per-Class Accuracy — {model_name}', fontsize=14)
    ax.set_xlim(0.8, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, acc in zip(bars, per_class_acc):
        ax.text(acc + 0.003, bar.get_y() + bar.get_height() / 2.,
                f'{acc:.3f}', va='center', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()
    return per_class_acc


def plot_sample_predictions(X_test, y_test, y_pred_labels, y_pred_confidence,
                            class_names, model_name='Model', save_path=None):
    """Top-5 most confident correct and incorrect predictions."""
    correct_mask = y_pred_labels == y_test
    incorrect_mask = ~correct_mask

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    correct_conf = y_pred_confidence.copy()
    correct_conf[incorrect_mask] = 0
    top_correct = np.argsort(correct_conf)[-5:][::-1]

    for i, idx in enumerate(top_correct):
        axes[0, i].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(
            f'{class_names[y_test[idx]]}\nConf: {y_pred_confidence[idx]:.3f}',
            fontsize=9, color='green')
        axes[0, i].axis('off')

    wrong_conf = y_pred_confidence.copy()
    wrong_conf[correct_mask] = 0
    top_wrong = np.argsort(wrong_conf)[-5:][::-1]

    for i, idx in enumerate(top_wrong):
        axes[1, i].imshow(X_test[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(
            f'True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_labels[idx]]}\nConf: {y_pred_confidence[idx]:.3f}',
            fontsize=8, color='red')
        axes[1, i].axis('off')

    axes[0, 0].annotate('Most Confident\nCorrect', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color='green', va='center', ha='right')
    axes[1, 0].annotate('Most Confident\nIncorrect', xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color='red', va='center', ha='right')

    plt.suptitle(f'Sample Predictions — {model_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()


def plot_results_bars(results_df, colors, save_path=None):
    """Bar charts comparing accuracy, F1, and parameter count across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_names = results_df['Model']
    x = np.arange(len(model_names))

    for ax, metric, ylim, label in zip(
        axes,
        ['Test Accuracy', 'Weighted F1', None],
        [(0.85, 1.0), (0.85, 1.0), None],
        ['Test Accuracy', 'Weighted F1 Score', 'Model Size (K params)']
    ):
        if metric:
            vals = results_df[metric]
            bars = ax.bar(x, vals, color=colors, alpha=0.85)
            ax.set_ylim(*ylim)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                        f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            vals = results_df['Parameters'] / 1000
            bars = ax.bar(x, vals, color=colors, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 5,
                        f'{val:.0f}K', ha='center', va='bottom', fontweight='bold')

        ax.set_title(label, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if not save_path:
        plt.show()
    plt.close()
