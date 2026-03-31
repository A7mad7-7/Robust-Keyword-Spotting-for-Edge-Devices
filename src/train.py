"""
Training script for the KWS CNN model.

Loads preprocessed data, builds the CNN, trains with dual validation,
and saves the best model based on noisy validation accuracy.
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PROCESSED_DIR, MODELS_DIR, NUM_CLASSES,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    FIGURES_DIR, TARGET_KEYWORDS, UNKNOWN_LABEL, SILENCE_LABEL
)
from pipeline import prepare_data, save_prepared_data, load_prepared_data
from model import build_kws_model
from callbacks import DualValidationCallback


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, test_mode=False):
    """
    Full training routine.

    Args:
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.
        lr:         Initial learning rate for Adam.
        test_mode:  If True, use a small data subset (smoke-test).

    Returns:
        Keras History object.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    processed_flag = os.path.join(PROCESSED_DIR, "X_train.npy")

    if os.path.exists(processed_flag) and not test_mode:
        print("Loading previously processed data...")
        data = load_prepared_data()
    else:
        print("Running data pipeline...")
        data = prepare_data(test_mode=test_mode)
        if not test_mode:
            save_prepared_data(data)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val_clean = data["X_val_clean"]
    X_val_noisy = data["X_val_noisy"]
    y_val = data["y_val"]

    num_classes = data["num_classes"]
    feature_shape = data["feature_shape"]  # (n_mels, n_frames)

    # ------------------------------------------------------------------
    # 2. Reshape for Conv2D: (samples, n_mels, n_frames) -> (..., 1)
    # ------------------------------------------------------------------
    X_train = X_train[..., np.newaxis]
    X_val_clean = X_val_clean[..., np.newaxis]
    X_val_noisy = X_val_noisy[..., np.newaxis]

    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    print(f"\nTraining data shape : {X_train.shape}")
    print(f"Val clean shape     : {X_val_clean.shape}")
    print(f"Val noisy shape     : {X_val_noisy.shape}")
    print(f"Input shape (model) : {input_shape}")
    print(f"Num classes         : {num_classes}")

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    model = build_kws_model(input_shape, num_classes)
    model.summary()

    # ------------------------------------------------------------------
    # 4. Compile
    # ------------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")

    dual_val_cb = DualValidationCallback(
        val_clean=(X_val_clean, y_val),
        val_noisy=(X_val_noisy, y_val),
        save_path=best_model_path,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_noisy_loss",
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=1e-6,
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_noisy_accuracy",
        mode="max",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=False,  # we already save best via callback
        verbose=1,
    )

    callbacks = [dual_val_cb, reduce_lr, early_stop]

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        # We pass val_noisy as validation_data so Keras' built-in
        # val metrics are also available; but the ground-truth dual
        # metrics live in the DualValidationCallback.
        validation_data=(X_val_noisy, y_val),
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best noisy val accuracy : {dual_val_cb.best_noisy_acc:.4f}")
    print(f"Best model saved to     : {best_model_path}")
    print(f"Total epochs run        : {len(dual_val_cb.noisy_acc_history)}")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING BEST MODEL FOR EVALUATION")
    print("=" * 60)
    try:
        model.load_weights(best_model_path)
        evaluate_model(model, X_val_clean, X_val_noisy, y_val)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    return history


def evaluate_model(model, X_clean, X_noisy, y_val, save_plots=True):
    """
    Evaluates the model on clean and noisy validation sets.
    Prints robustness gap and plots confusion matrix.
    
    Args:
        model: Trained Keras model.
        X_clean: Clean validation features.
        X_noisy: Noisy validation features.
        y_val: Validation labels.
        save_plots: Whether to save plots to disk.
    """
    # Create output directory
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Predict
    print("\n" + "="*60)
    print("POST-TRAINING DUAL EVALUATION")
    print("="*60)
    
    # Clean evaluation
    print("Evaluating on Clean Data...")
    y_pred_clean = np.argmax(model.predict(X_clean, verbose=0), axis=1)
    acc_clean = accuracy_score(y_val, y_pred_clean)
    
    # Noisy evaluation
    print("Evaluating on Noisy Data...")
    y_pred_noisy = np.argmax(model.predict(X_noisy, verbose=0), axis=1)
    acc_noisy = accuracy_score(y_val, y_pred_noisy)
    
    # Robustness Gap
    gap = acc_clean - acc_noisy
    print("-" * 30)
    print(f"Clean Validation Accuracy : {acc_clean:.4f}")
    print(f"Noisy Validation Accuracy : {acc_noisy:.4f}")
    print(f"Robustness Gap            : {gap:.4f}")
    print("-" * 30)
    
    if save_plots:
        plot_confusion_matrix(y_val, y_pred_noisy, acc_noisy, suffix="noisy")
        plot_accuracy_comparison(y_val, y_pred_clean, y_pred_noisy)


def plot_confusion_matrix(y_true, y_pred, accuracy, suffix=""):
    """Plots and saves a confusion matrix."""
    class_names = TARGET_KEYWORDS + [UNKNOWN_LABEL, SILENCE_LABEL]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    title = f'Confusion Matrix ({suffix.capitalize()} Val) - Acc: {accuracy:.2f}'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix_{suffix}.png' if suffix else 'confusion_matrix.png'
    cm_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")


def plot_accuracy_comparison(y_true, y_pred_clean, y_pred_noisy):
    """Plots a side-by-side bar chart of accuracy per class."""
    class_names = TARGET_KEYWORDS + [UNKNOWN_LABEL, SILENCE_LABEL]
    clean_accs = []
    noisy_accs = []
    
    for i in range(len(class_names)):
        indices = np.where(y_true == i)[0]
        if len(indices) == 0:
            clean_accs.append(0)
            noisy_accs.append(0)
            continue
            
        clean_accs.append(np.mean(y_pred_clean[indices] == y_true[indices]))
        noisy_accs.append(np.mean(y_pred_noisy[indices] == y_true[indices]))
        
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, clean_accs, width, label='Clean')
    plt.bar(x + width/2, noisy_accs, width, label='Noisy')
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class: Clean vs Noisy')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    comp_path = os.path.join(FIGURES_DIR, 'accuracy_comparison.png')
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved accuracy comparison to: {comp_path}")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KWS CNN Model")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use small data subset for a quick smoke-test")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_mode=args.test_mode,
    )
