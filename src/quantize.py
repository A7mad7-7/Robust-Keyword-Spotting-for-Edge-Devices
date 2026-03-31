"""
Post-Training Quantization (PTQ) script for the KWS model.

Converts a trained .keras model to INT8 TFLite format using
Post-Training Quantization with a representative dataset.
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODELS_DIR, N_MELS, N_FFT, HOP_LENGTH, AUDIO_LENGTH, NUM_CLASSES
)


def representative_dataset_gen(input_shape, n_samples=100):
    """
    Generator that yields representative samples for INT8 calibration.

    Uses random data of the correct shape. For best accuracy, replace
    with real validation samples.

    Args:
        input_shape: Model input shape (n_mels, n_frames, 1)
        n_samples: Number of calibration samples

    Yields:
        List containing a single sample as np.float32 array
    """
    np.random.seed(42)
    for _ in range(n_samples):
        # Generate random sample with realistic mel-spectrogram range
        # Log-mel spectrograms typically range from ~-80 to 0 dB (unnormalized)
        # After z-score normalization: roughly [-3, 3]
        sample = np.random.randn(1, *input_shape).astype(np.float32)
        yield [sample]


def quantize_model(model_path=None, output_path=None, n_calibration_samples=100):
    """
    Convert a trained Keras model to INT8 TFLite format.

    Args:
        model_path: Path to the .keras model file.
                    Defaults to MODELS_DIR/best_model.keras.
        output_path: Path to save the .tflite model.
                     Defaults to MODELS_DIR/best_model_int8.tflite.
        n_calibration_samples: Number of samples for calibration.

    Returns:
        Tuple of (original_size_kb, quantized_size_kb)
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model.keras")

    if output_path is None:
        output_path = os.path.join(MODELS_DIR, "best_model_int8.tflite")

    # ------------------------------------------------------------------
    # 1. Load the trained model
    # ------------------------------------------------------------------
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    input_shape = model.input_shape[1:]  # Remove batch dimension
    print(f"\nModel input shape: {input_shape}")

    # ------------------------------------------------------------------
    # 2. Convert to TFLite (FP32 baseline)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Converting to FP32 TFLite (baseline)")
    print("="*60)

    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()

    fp32_path = output_path.replace("_int8.tflite", "_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)

    fp32_size_kb = len(tflite_fp32) / 1024
    print(f"FP32 TFLite model saved to: {fp32_path}")
    print(f"FP32 model size: {fp32_size_kb:.1f} KB ({fp32_size_kb/1024:.2f} MB)")

    # ------------------------------------------------------------------
    # 3. Convert to TFLite with INT8 PTQ
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: Converting to INT8 TFLite (PTQ)")
    print("="*60)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset for calibration
    converter.representative_dataset = lambda: representative_dataset_gen(
        input_shape, n_samples=n_calibration_samples
    )

    # Force full INT8 quantization (input/output remain float for compatibility)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(f"Running INT8 calibration with {n_calibration_samples} samples...")
    tflite_int8 = converter.convert()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_int8)

    int8_size_kb = len(tflite_int8) / 1024

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE")
    print("="*60)
    print(f"INT8 TFLite model saved to : {output_path}")
    print(f"{'─'*50}")
    print(f"Original Keras model       : {os.path.getsize(model_path)/1024:.1f} KB")
    print(f"FP32 TFLite model          : {fp32_size_kb:.1f} KB")
    print(f"INT8 TFLite model          : {int8_size_kb:.1f} KB")
    print(f"{'─'*50}")
    print(f"Compression ratio (vs FP32): {fp32_size_kb/int8_size_kb:.1f}x")
    print(f"Size reduction             : {(1 - int8_size_kb/fp32_size_kb)*100:.1f}%")
    print(f"{'='*60}")

    return fp32_size_kb, int8_size_kb


def verify_tflite_model(tflite_path, input_shape):
    """
    Quick verification that the TFLite model can run inference.

    Args:
        tflite_path: Path to the .tflite model
        input_shape: Model input shape (n_mels, n_frames, 1)
    """
    print(f"\nVerifying TFLite model: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    # Run inference with a dummy sample
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.int8:
        # For INT8 models, we need to quantize the input
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        dummy = np.random.randn(1, *input_shape).astype(np.float32)
        dummy_quantized = (dummy / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], dummy_quantized)
    else:
        dummy = np.random.randn(1, *input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy)

    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"  Output shape: {output.shape}")
    print(f"  ✓ Inference successful!")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize KWS model to INT8 TFLite")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .keras model (default: best_model.keras)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for output .tflite (default: best_model_int8.tflite)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of calibration samples (default: 100)")
    args = parser.parse_args()

    fp32_kb, int8_kb = quantize_model(
        model_path=args.model,
        output_path=args.output,
        n_calibration_samples=args.n_samples,
    )

    # Verify the INT8 model
    n_frames = 1 + AUDIO_LENGTH // HOP_LENGTH
    input_shape = (N_MELS, n_frames, 1)

    output_path = args.output or os.path.join(MODELS_DIR, "best_model_int8.tflite")
    verify_tflite_model(output_path, input_shape)
