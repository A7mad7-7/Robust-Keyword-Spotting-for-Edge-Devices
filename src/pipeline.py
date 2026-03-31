"""
Main pipeline script for preparing the Keyword Spotting dataset.
Orchestrates data loading, preprocessing, augmentation, and feature extraction.
"""
import os
import sys
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DATA_DIR, PROCESSED_DIR, SAMPLE_RATE, AUDIO_LENGTH,
    NOISE_INJECTION_RATIO, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES
)
from data_loader import SpeechCommandsDataLoader
from audio_processor import AudioProcessor, create_noisy_dataset, load_all_noise_files
from augmentation import balance_classes


def load_waveforms(file_paths, labels, processor, desc="Loading audio"):
    """
    Load and preprocess audio files to waveforms.
    Skips files that fail to load/process and keeps labels in sync.
    
    Args:
        file_paths: List of audio file paths
        labels: List/Array of corresponding labels
        processor: AudioProcessor instance
        desc: Description for progress bar
        
    Returns:
        Tuple of (waveforms, valid_labels)
    """
    waveforms = []
    valid_labels = []
    
    if labels is None:
        for path in tqdm(file_paths, desc=desc):
            try:
                waveform = processor.load_audio(path)
                waveform = processor.preprocess(waveform)
                waveforms.append(waveform)
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return np.array(waveforms), None

    for path, label in tqdm(zip(file_paths, labels), total=len(file_paths), desc=desc):
        try:
            waveform = processor.load_audio(path)
            waveform = processor.preprocess(waveform)
            waveforms.append(waveform)
            valid_labels.append(label)
        except Exception as e:
            print(f"\nError processing file {path}: {e}")
            continue
            
    if len(waveforms) == 0:
        print(f"Warning: No valid waveforms loaded for {desc}")
        return np.array([]), np.array([])
        
    return np.array(waveforms), np.array(valid_labels)


def prepare_noisy_split(waveforms, noise_waveforms, injection_ratio=NOISE_INJECTION_RATIO):
    """
    Create noisy versions of a data split.
    
    Returns both clean and noisy versions with matching indices.
    """
    print(f"  Creating noisy version ({injection_ratio*100:.0f}% with noise)...")
    noisy_waveforms, noise_mask = create_noisy_dataset(
        waveforms, noise_waveforms, injection_ratio=injection_ratio
    )
    print(f"  {noise_mask.sum()} samples have noise added")
    return waveforms, noisy_waveforms


def prepare_data(data_dir=RAW_DATA_DIR, save_dir=PROCESSED_DIR, test_mode=False):
    """
    Full data preparation pipeline.
    
    Steps:
    1. Load and split data
    2. Preprocess all audio files
    3. Add noise to 50% of each split
    4. Augment and balance training set
    5. Extract features
    6. Normalize using training statistics
    """
    print("="*60)
    print("KEYWORD SPOTTING DATA PIPELINE")
    print("="*60)
    
    # Initialize processor
    processor = AudioProcessor()
    print(f"\nAudio Processor initialized:")
    print(f"  Sample rate: {processor.sample_rate} Hz")
    print(f"  Audio length: {processor.audio_length} samples")
    print(f"  Feature shape: {processor.feature_shape}")
    
    # Step 1: Load and split data
    print("\n" + "-"*40)
    print("STEP 1: Loading and Splitting Data")
    print("-"*40)
    
    loader = SpeechCommandsDataLoader(data_dir=data_dir)
    splits = loader.prepare_dataset()
    
    noise_files = splits['background_noise_files']
    noise_waveforms = load_all_noise_files(noise_files)
    
    # In test mode, use only a small subset
    if test_mode:
        print("\n[TEST MODE] Using small subset of data")
        for key in ['train_files', 'val_files', 'test_files']:
            splits[key] = splits[key][:100]
        for key in ['train_labels', 'val_labels', 'test_labels']:
            splits[key] = splits[key][:100]
    
    # Step 2: Load and preprocess audio files
    print("\n" + "-"*40)
    print("STEP 2: Loading and Preprocessing Audio")
    print("-"*40)
    
    print("\nLoading training set...")
    train_waveforms, train_labels = load_waveforms(splits['train_files'], splits['train_labels'], processor, "Loading train")
    
    print("\nLoading validation set...")
    val_waveforms, val_labels = load_waveforms(splits['val_files'], splits['val_labels'], processor, "Loading val")
    
    print("\nLoading test set...")
    test_waveforms, test_labels = load_waveforms(splits['test_files'], splits['test_labels'], processor, "Loading test")
    
    # Step 3: Create noisy versions for val/test
    print("\n" + "-"*40)
    print("STEP 3: Creating Noisy Datasets")
    print("-"*40)
    
    print("\nValidation set:")
    val_clean, val_noisy = prepare_noisy_split(val_waveforms, noise_waveforms)
    
    print("\nTest set:")
    test_clean, test_noisy = prepare_noisy_split(test_waveforms, noise_waveforms)
    
    # Training set: add noise to 50%
    print("\nTraining set:")
    _, train_waveforms_noisy = prepare_noisy_split(train_waveforms, noise_waveforms)
    train_waveforms = train_waveforms_noisy  # Training uses noisy version
    
    # Step 4: Augment and balance training set
    print("\n" + "-"*40)
    print("STEP 4: Augmenting and Balancing Training Set")
    print("-"*40)
    
    train_waveforms, train_labels = balance_classes(train_waveforms, train_labels)
    
    # Step 5: Extract features
    print("\n" + "-"*40)
    print("STEP 5: Extracting Features")
    print("-"*40)
    
    print("\nExtracting training features...")
    X_train = processor.process_batch(train_waveforms, normalize=False)
    
    print("\nExtracting validation features (clean)...")
    X_val_clean = processor.process_batch(val_clean, normalize=False)
    
    print("\nExtracting validation features (noisy)...")
    X_val_noisy = processor.process_batch(val_noisy, normalize=False)
    
    print("\nExtracting test features (clean)...")
    X_test_clean = processor.process_batch(test_clean, normalize=False)
    
    print("\nExtracting test features (noisy)...")
    X_test_noisy = processor.process_batch(test_noisy, normalize=False)
    
    # Step 6: Fit scaler and normalize
    print("\n" + "-"*40)
    print("STEP 6: Normalizing Features")
    print("-"*40)
    
    print("\nFitting scaler on training data...")
    processor.fit_scaler(X_train)
    
    print("Normalizing all datasets...")
    X_train = processor.normalize(X_train)
    X_val_clean = processor.normalize(X_val_clean)
    X_val_noisy = processor.normalize(X_val_noisy)
    X_test_clean = processor.normalize(X_test_clean)
    X_test_noisy = processor.normalize(X_test_noisy)
    
    # Verify normalization
    print(f"\nNormalization verification (training set):")
    print(f"  Mean: {X_train.mean():.4f} (should be ≈0)")
    print(f"  Std:  {X_train.std():.4f} (should be ≈1)")
    
    # Save processor configuration
    os.makedirs(save_dir, exist_ok=True)
    processor_path = os.path.join(save_dir, 'audio_processor_config.json')
    processor.save(processor_path)
    
    # Prepare output dictionary
    result = {
        'X_train': X_train,
        'y_train': train_labels,
        'X_val_clean': X_val_clean,
        'X_val_noisy': X_val_noisy,
        'y_val': val_labels,
        'X_test_clean': X_test_clean,
        'X_test_noisy': X_test_noisy,
        'y_test': test_labels,
        'label_map': IDX_TO_LABEL,
        'num_classes': NUM_CLASSES,
        'feature_shape': processor.feature_shape
    }
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nDataset shapes:")
    print(f"  X_train:      {X_train.shape}")
    print(f"  y_train:      {train_labels.shape}")
    print(f"  X_val_clean:  {X_val_clean.shape}")
    print(f"  X_val_noisy:  {X_val_noisy.shape}")
    print(f"  y_val:        {val_labels.shape}")
    print(f"  X_test_clean: {X_test_clean.shape}")
    print(f"  X_test_noisy: {X_test_noisy.shape}")
    print(f"  y_test:       {test_labels.shape}")
    print(f"\nNumber of classes: {NUM_CLASSES}")
    print(f"Label mapping: {IDX_TO_LABEL}")
    
    return result


def save_prepared_data(result, save_dir=PROCESSED_DIR):
    """
    Save prepared data arrays to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving prepared data to {save_dir}...")
    
    # Save arrays
    np.save(os.path.join(save_dir, 'X_train.npy'), result['X_train'])
    np.save(os.path.join(save_dir, 'y_train.npy'), result['y_train'])
    np.save(os.path.join(save_dir, 'X_val_clean.npy'), result['X_val_clean'])
    np.save(os.path.join(save_dir, 'X_val_noisy.npy'), result['X_val_noisy'])
    np.save(os.path.join(save_dir, 'y_val.npy'), result['y_val'])
    np.save(os.path.join(save_dir, 'X_test_clean.npy'), result['X_test_clean'])
    np.save(os.path.join(save_dir, 'X_test_noisy.npy'), result['X_test_noisy'])
    np.save(os.path.join(save_dir, 'y_test.npy'), result['y_test'])
    
    # Save metadata
    import json
    metadata = {
        'num_classes': result['num_classes'],
        'feature_shape': list(result['feature_shape']),
        'label_map': {str(k): v for k, v in result['label_map'].items()}
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data saved successfully!")


def load_prepared_data(load_dir=PROCESSED_DIR):
    """
    Load previously prepared data from disk.
    """
    import json
    
    print(f"Loading prepared data from {load_dir}...")
    
    result = {
        'X_train': np.load(os.path.join(load_dir, 'X_train.npy')),
        'y_train': np.load(os.path.join(load_dir, 'y_train.npy')),
        'X_val_clean': np.load(os.path.join(load_dir, 'X_val_clean.npy')),
        'X_val_noisy': np.load(os.path.join(load_dir, 'X_val_noisy.npy')),
        'y_val': np.load(os.path.join(load_dir, 'y_val.npy')),
        'X_test_clean': np.load(os.path.join(load_dir, 'X_test_clean.npy')),
        'X_test_noisy': np.load(os.path.join(load_dir, 'X_test_noisy.npy')),
        'y_test': np.load(os.path.join(load_dir, 'y_test.npy'))
    }
    
    # Load metadata
    with open(os.path.join(load_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    result['num_classes'] = metadata['num_classes']
    result['feature_shape'] = tuple(metadata['feature_shape'])
    result['label_map'] = {int(k): v for k, v in metadata['label_map'].items()}
    
    print(f"Loaded dataset shapes:")
    print(f"  X_train: {result['X_train'].shape}")
    print(f"  X_val:   {result['X_val_clean'].shape}")
    print(f"  X_test:  {result['X_test_clean'].shape}")
    
    return result


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Robust Keyword Spotting — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                     # Run full pipeline (data → train → test → quantize)
  python pipeline.py --test-mode         # Quick smoke-test with small subset
  python pipeline.py --skip-data         # Skip data prep, use saved .npy files
  python pipeline.py --skip-train        # Skip training, use existing model
  python pipeline.py --skip-quantize     # Skip INT8 quantization
        """
    )
    parser.add_argument('--test-mode', action='store_true',
                        help="Run in test mode with small subset")
    parser.add_argument('--skip-data', action='store_true',
                        help="Skip data preparation (use saved .npy files)")
    parser.add_argument('--skip-train', action='store_true',
                        help="Skip training (use existing model)")
    parser.add_argument('--skip-quantize', action='store_true',
                        help="Skip INT8 quantization")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument('--batch-size', type=int, default=None,
                        help="Override batch size")
    parser.add_argument('--lr', type=float, default=None,
                        help="Override learning rate")
    args = parser.parse_args()

    from config import MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE

    start_time = time.time()

    print("=" * 70)
    print("  ROBUST KEYWORD SPOTTING FOR EDGE DEVICES — FULL PIPELINE")
    print("=" * 70)

    # ==================================================================
    # STEP 1: Data Preparation
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STEP 1 / 4 — DATA PREPARATION")
    print("▓" * 70)

    processed_flag = os.path.join(PROCESSED_DIR, "X_train.npy")

    if args.skip_data and os.path.exists(processed_flag):
        print("  ⏭ Skipping data preparation. Using saved .npy files.")
    else:
        result = prepare_data(test_mode=args.test_mode)
        save_prepared_data(result)

    # ==================================================================
    # STEP 2: Training
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STEP 2 / 4 — MODEL TRAINING")
    print("▓" * 70)

    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")

    if args.skip_train and os.path.exists(best_model_path):
        print(f"  ⏭ Skipping training. Using existing model: {best_model_path}")
    else:
        from train import train

        train_kwargs = {}
        if args.epochs is not None:
            train_kwargs['epochs'] = args.epochs
        if args.batch_size is not None:
            train_kwargs['batch_size'] = args.batch_size
        if args.lr is not None:
            train_kwargs['lr'] = args.lr
        train_kwargs['test_mode'] = args.test_mode

        train(**train_kwargs)

    # ==================================================================
    # STEP 3: Test Evaluation
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STEP 3 / 4 — MODEL EVALUATION ON TEST SET")
    print("▓" * 70)

    if os.path.exists(best_model_path):
        from test import evaluate_model as test_evaluate
        test_evaluate(model_path=best_model_path)
    else:
        print("  ✗ No trained model found. Skipping evaluation.")

    # ==================================================================
    # STEP 4: INT8 Quantization
    # ==================================================================
    print("\n" + "▓" * 70)
    print("  STEP 4 / 4 — INT8 QUANTIZATION")
    print("▓" * 70)

    if args.skip_quantize:
        print("  ⏭ Skipping quantization.")
    elif os.path.exists(best_model_path):
        from quantize import quantize_model, verify_tflite_model
        from config import N_MELS, N_FFT, HOP_LENGTH, AUDIO_LENGTH

        quantize_model(model_path=best_model_path)

        # Verify the INT8 model
        n_frames = 1 + AUDIO_LENGTH // HOP_LENGTH
        input_shape = (N_MELS, n_frames, 1)
        int8_path = os.path.join(MODELS_DIR, "best_model_int8.tflite")
        if os.path.exists(int8_path):
            verify_tflite_model(int8_path, input_shape)
    else:
        print("  ✗ No trained model found. Skipping quantization.")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {minutes}m {seconds}s")

    # List output files
    print("\n  Output files:")
    from config import FIGURES_DIR
    for subdir in [MODELS_DIR, FIGURES_DIR, PROCESSED_DIR]:
        if os.path.isdir(subdir):
            for f in sorted(os.listdir(subdir)):
                fpath = os.path.join(subdir, f)
                if os.path.isfile(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    print(f"    {f:40s}  {size_kb:>8.1f} KB")

    print("=" * 70)

