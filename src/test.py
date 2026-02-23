"""
Test evaluation script for the KWS CNN model.

Loads the best saved model and evaluates it on both clean and noisy
test sets.  Produces a classification report and confusion matrix.
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PROCESSED_DIR, MODELS_DIR, IDX_TO_LABEL
from pipeline import load_prepared_data


def evaluate_model(model_path=None):
    """
    Evaluate the trained model on clean and noisy test sets.

    Args:
        model_path: Path to the saved .keras model.
                    Defaults to MODELS_DIR/best_model.keras.

    Returns:
        Dict with test metrics.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model.keras")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # ------------------------------------------------------------------
    # 2. Load test data
    # ------------------------------------------------------------------
    data = load_prepared_data()
    X_test_clean = data["X_test_clean"][..., np.newaxis]
    X_test_noisy = data["X_test_noisy"][..., np.newaxis]
    y_test = data["y_test"]
    label_map = data["label_map"]  # {idx: label_str}

    class_names = [label_map[i] for i in range(len(label_map))]

    print(f"\nTest clean shape : {X_test_clean.shape}")
    print(f"Test noisy shape : {X_test_noisy.shape}")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION ON CLEAN TEST SET")
    print("=" * 60)
    clean_loss, clean_acc = model.evaluate(X_test_clean, y_test, verbose=0)
    y_pred_clean = np.argmax(model.predict(X_test_clean, verbose=0), axis=1)
    print(f"Loss: {clean_loss:.4f}  Accuracy: {clean_acc:.4f}\n")
    print(classification_report(y_test, y_pred_clean, target_names=class_names))

    print("\n" + "=" * 60)
    print("EVALUATION ON NOISY TEST SET")
    print("=" * 60)
    noisy_loss, noisy_acc = model.evaluate(X_test_noisy, y_test, verbose=0)
    y_pred_noisy = np.argmax(model.predict(X_test_noisy, verbose=0), axis=1)
    print(f"Loss: {noisy_loss:.4f}  Accuracy: {noisy_acc:.4f}\n")
    print(classification_report(y_test, y_pred_noisy, target_names=class_names))

    # ------------------------------------------------------------------
    # 4. Confusion matrices
    # ------------------------------------------------------------------
    _save_confusion_matrices(y_test, y_pred_clean, y_pred_noisy, class_names)

    # ------------------------------------------------------------------
    # 5. Save summary to JSON
    # ------------------------------------------------------------------
    results = {
        "clean_test_loss": float(clean_loss),
        "clean_test_accuracy": float(clean_acc),
        "noisy_test_loss": float(noisy_loss),
        "noisy_test_accuracy": float(noisy_acc),
        "robustness_gap": float(clean_acc - noisy_acc),
    }
    results_path = os.path.join(MODELS_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def _save_confusion_matrices(y_true, y_pred_clean, y_pred_noisy, class_names):
    """Plot and save confusion matrices for clean and noisy predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, y_pred, title in [
        (ax1, y_pred_clean, "Clean Test Set"),
        (ax2, y_pred_noisy, "Noisy Test Set"),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {plot_path}")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KWS CNN Model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .keras model (default: best_model.keras)")
    args = parser.parse_args()

    evaluate_model(model_path=args.model)
