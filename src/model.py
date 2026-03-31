"""
Lightweight CNN model for Keyword Spotting (KWS).
Designed for embedded / edge deployment.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from config import DROPOUT_RATE_1, DROPOUT_RATE_2


def build_kws_model(input_shape, num_classes):
    """
    Build a lightweight 3-block CNN for keyword spotting.

    Architecture:
        Conv Block 1: Conv2D(32, 3x3) -> BN -> ReLU -> MaxPool(2,2)
        Conv Block 2: Conv2D(64, 3x3) -> BN -> ReLU -> MaxPool(2,2)
        Conv Block 3: Conv2D(64, 3x3) -> BN -> ReLU -> MaxPool(2,2)
        Classifier:   Flatten -> Dropout(0.5) -> Dense(64, relu)
                       -> Dropout(0.25) -> Dense(32, relu)
                       -> Dense(num_classes, linear)

    Args:
        input_shape: Tuple (n_mels, n_frames, 1), e.g. (40, 97, 1).
        num_classes: Number of output classes.

    Returns:
        A tf.keras.Model instance (uncompiled).
    """
    inputs = layers.Input(shape=input_shape, name="mel_input")

    # --- Conv Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # --- Conv Block 2 ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # --- Conv Block 3 ---
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # --- Classifier Head ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(DROPOUT_RATE_1, name="dropout1")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(DROPOUT_RATE_2, name="dropout2")(x)
    x = layers.Dense(32, activation="relu", name="dense2")(x)
    outputs = layers.Dense(num_classes, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="kws_cnn")
    return model


if __name__ == "__main__":
    from config import NUM_CLASSES, N_MELS, N_FFT, HOP_LENGTH, AUDIO_LENGTH

    n_frames = 1 + AUDIO_LENGTH // HOP_LENGTH
    input_shape = (N_MELS, n_frames, 1)

    model = build_kws_model(input_shape, NUM_CLASSES)
    model.summary()

    # --- Parameter & Size Breakdown ---
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )

    # Estimate size: float32 = 4 bytes per param
    size_fp32_kb = (total_params * 4) / 1024
    size_fp32_mb = size_fp32_kb / 1024
    size_int8_kb = total_params / 1024  # 1 byte per param after INT8 quantization

    print(f"\n{'='*50}")
    print(f"MODEL STATISTICS")
    print(f"{'='*50}")
    print(f"Input shape          : {input_shape}")
    print(f"Output shape         : {model.output_shape}")
    print(f"Num classes          : {NUM_CLASSES}")
    print(f"{'─'*50}")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Non-trainable params : {non_trainable_params:,}")
    print(f"{'─'*50}")
    print(f"Size (FP32)          : {size_fp32_kb:.1f} KB  ({size_fp32_mb:.2f} MB)")
    print(f"Size (INT8 est.)     : {size_int8_kb:.1f} KB")
    print(f"{'='*50}")
