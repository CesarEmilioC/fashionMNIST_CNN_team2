import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_model(model, X_test, y_test, model_name, class_names):
    """
    Compute test accuracy, loss, weighted F1, and print the full classification report.
    Returns (test_acc, test_loss, f1, y_pred_classes, y_pred_probs).
    """
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    print(f"\n{'=' * 50}")
    print(f"{model_name} — Test Results")
    print(f"{'=' * 50}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Weighted F1   : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    return test_acc, test_loss, f1, y_pred_classes, y_pred_probs


def get_confusion_matrix(y_test, y_pred_classes):
    return confusion_matrix(y_test, y_pred_classes)


def build_results_table(model_entries):
    """
    Build a comparison DataFrame from a list of dicts with keys:
    model_name, model_obj, test_acc, test_loss, f1, train_time_s
    """
    rows = []
    for entry in model_entries:
        rows.append({
            'Model': entry['model_name'],
            'Parameters': entry['model_obj'].count_params(),
            'Test Accuracy': round(entry['test_acc'], 4),
            'Test Loss': round(entry['test_loss'], 4),
            'Weighted F1': round(entry['f1'], 4),
            'Train Time (s)': round(entry['train_time_s'], 1),
        })
    df = pd.DataFrame(rows)
    print("\nModel Comparison Summary")
    print("=" * 90)
    print(df.to_string(index=False))
    return df
