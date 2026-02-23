"""
Custom Keras callbacks for the KWS training pipeline.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt

from config import MODELS_DIR


class DualValidationCallback(tf.keras.callbacks.Callback):
    """
    Evaluate the model on both a clean and a noisy validation set at the
    end of every epoch. Saves the best model based on **noisy** accuracy
    (because robustness is the primary goal).

    Also records per-epoch metrics so they can be consumed by
    ReduceLROnPlateau via the logs dict.

    Usage:
        cb = DualValidationCallback(
            val_clean=(X_val_clean, y_val),
            val_noisy=(X_val_noisy, y_val),
            save_path="data/models/best_model.keras",
        )
        model.fit(..., callbacks=[cb])
    """

    def __init__(self, val_clean, val_noisy, save_path):
        """
        Args:
            val_clean: Tuple (X_val_clean, y_val).
            val_noisy: Tuple (X_val_noisy, y_val).
            save_path: File path to save the best model.
        """
        super().__init__()
        self.X_val_clean, self.y_val_clean = val_clean
        self.X_val_noisy, self.y_val_noisy = val_noisy
        self.save_path = save_path

        # History lists
        self.clean_acc_history = []
        self.noisy_acc_history = []
        self.clean_loss_history = []
        self.noisy_loss_history = []

        self.best_noisy_acc = -np.inf

    # ------------------------------------------------------------------ #
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # --- Evaluate on clean val set ---
        clean_loss, clean_acc = self.model.evaluate(
            self.X_val_clean, self.y_val_clean, verbose=0
        )

        # --- Evaluate on noisy val set ---
        noisy_loss, noisy_acc = self.model.evaluate(
            self.X_val_noisy, self.y_val_noisy, verbose=0
        )

        # Record
        self.clean_acc_history.append(clean_acc)
        self.noisy_acc_history.append(noisy_acc)
        self.clean_loss_history.append(clean_loss)
        self.noisy_loss_history.append(noisy_loss)

        # Inject into logs so ReduceLROnPlateau / EarlyStopping can use them
        logs["val_clean_accuracy"] = clean_acc
        logs["val_clean_loss"] = clean_loss
        logs["val_noisy_accuracy"] = noisy_acc
        logs["val_noisy_loss"] = noisy_loss

        print(
            f"  ── dual_val │ clean_acc: {clean_acc:.4f}  "
            f"noisy_acc: {noisy_acc:.4f}  "
            f"gap: {clean_acc - noisy_acc:+.4f}"
        )

        # Save best model (based on noisy accuracy)
        if noisy_acc > self.best_noisy_acc:
            self.best_noisy_acc = noisy_acc
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.save(self.save_path)
            print(f"  ── ✓ Best noisy accuracy improved → saved to {self.save_path}")

    # ------------------------------------------------------------------ #
    def on_train_end(self, logs=None):
        """Plot clean vs noisy accuracy over epochs."""
        epochs = range(1, len(self.clean_acc_history) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Accuracy ---
        ax1.plot(epochs, self.clean_acc_history, "b-o", markersize=4, label="Clean Val Acc")
        ax1.plot(epochs, self.noisy_acc_history, "r-s", markersize=4, label="Noisy Val Acc")
        ax1.fill_between(
            epochs,
            self.noisy_acc_history,
            self.clean_acc_history,
            alpha=0.15,
            color="grey",
            label="Robustness Gap",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Clean vs Noisy Validation Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Loss ---
        ax2.plot(epochs, self.clean_loss_history, "b-o", markersize=4, label="Clean Val Loss")
        ax2.plot(epochs, self.noisy_loss_history, "r-s", markersize=4, label="Noisy Val Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Clean vs Noisy Validation Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(os.path.dirname(self.save_path), "dual_validation_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\nDual-validation plot saved to {plot_path}")
