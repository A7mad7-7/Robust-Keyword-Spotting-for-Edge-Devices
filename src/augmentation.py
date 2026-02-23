"""
Augmentation functions for the training set.
Includes time shift, pitch shifting, time stretching, and class balancing.
"""
import numpy as np
import librosa
from collections import Counter

from config import SAMPLE_RATE, AUDIO_LENGTH, RANDOM_SEED


np.random.seed(RANDOM_SEED)


def time_shift(waveform, max_shift_ratio=0.1):
    """
    Shift audio in time domain (circular shift).
    
    Args:
        waveform: Input audio waveform
        max_shift_ratio: Maximum shift as ratio of audio length
        
    Returns:
        Time-shifted waveform
    """
    max_shift = int(len(waveform) * max_shift_ratio)
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(waveform, shift)


def pitch_shift(waveform, sr=SAMPLE_RATE, n_steps_range=(-2, 2)):
    """
    Change pitch of audio using librosa.
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
        n_steps_range: Range of semitones to shift (min, max)
        
    Returns:
        Pitch-shifted waveform
    """
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    
    # Ensure same length
    if len(shifted) > len(waveform):
        shifted = shifted[:len(waveform)]
    elif len(shifted) < len(waveform):
        shifted = np.pad(shifted, (0, len(waveform) - len(shifted)), mode='constant')
    
    return shifted


def time_stretch(waveform, rate_range=(0.8, 1.2)):
    """
    Stretch/compress audio in time using librosa.
    
    Args:
        waveform: Input audio waveform
        rate_range: Range of stretch rates (min, max). >1 = faster, <1 = slower
        
    Returns:
        Time-stretched waveform (padded/truncated to original length)
    """
    rate = np.random.uniform(rate_range[0], rate_range[1])
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    
    # Restore to original length
    target_length = len(waveform)
    if len(stretched) > target_length:
        # Truncate (center crop)
        start = (len(stretched) - target_length) // 2
        stretched = stretched[start:start + target_length]
    elif len(stretched) < target_length:
        # Pad (center padding)
        pad_left = (target_length - len(stretched)) // 2
        pad_right = target_length - len(stretched) - pad_left
        stretched = np.pad(stretched, (pad_left, pad_right), mode='constant')
    
    return stretched


def add_gaussian_noise(waveform, noise_std_range=(0.001, 0.005)):
    """
    Add Gaussian noise to waveform.
    
    Args:
        waveform: Input audio waveform
        noise_std_range: Range of noise standard deviation
        
    Returns:
        Noisy waveform
    """
    noise_std = np.random.uniform(noise_std_range[0], noise_std_range[1])
    noise = np.random.normal(0, noise_std, len(waveform))
    return waveform + noise


def random_volume(waveform, volume_range=(0.8, 1.2)):
    """
    Randomly scale volume.
    
    Args:
        waveform: Input audio waveform
        volume_range: Range of volume scaling factor
        
    Returns:
        Volume-scaled waveform
    """
    volume = np.random.uniform(volume_range[0], volume_range[1])
    return waveform * volume


def augment_sample(waveform, sr=SAMPLE_RATE):
    """
    Apply random augmentations to a single sample.
    
    Randomly applies a combination of:
    - Time shift
    - Pitch shift
    - Time stretch
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
        
    Returns:
        Augmented waveform
    """
    augmented = waveform.copy()
    
    # Apply augmentations with some probability
    if np.random.random() < 0.5:
        augmented = time_shift(augmented)
    
    if np.random.random() < 0.5:
        augmented = pitch_shift(augmented, sr=sr)
    
    if np.random.random() < 0.5:
        augmented = time_stretch(augmented)
    
    return augmented


def balance_classes(waveforms, labels, target_count=None):
    """
    Balance classes by oversampling minority classes with augmentation.
    
    Args:
        waveforms: Array of audio waveforms
        labels: Array of labels (integers)
        target_count: Target count per class (default: max class count)
        
    Returns:
        Tuple of (balanced_waveforms, balanced_labels)
    """
    from tqdm import tqdm
    
    # Count samples per class
    class_counts = Counter(labels)
    print(f"\nOriginal class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  Class {label}: {count} samples")
    
    # Determine target count
    if target_count is None:
        target_count = max(class_counts.values())
    
    print(f"\nBalancing to {target_count} samples per class...")
    
    balanced_waveforms = []
    balanced_labels = []
    
    # Group samples by label
    class_samples = {}
    for i, label in enumerate(labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(waveforms[i])
    
    # Balance each class
    for label in tqdm(sorted(class_samples.keys()), desc="Balancing classes"):
        samples = class_samples[label]
        current_count = len(samples)
        
        # Add all original samples
        balanced_waveforms.extend(samples)
        balanced_labels.extend([label] * current_count)
        
        # Generate augmented samples if needed
        if current_count < target_count:
            n_to_generate = target_count - current_count
            
            for _ in range(n_to_generate):
                # Select random sample from class
                idx = np.random.randint(current_count)
                original = samples[idx]
                
                # Apply augmentation
                augmented = augment_sample(original)
                
                balanced_waveforms.append(augmented)
                balanced_labels.append(label)
    
    balanced_waveforms = np.array(balanced_waveforms)
    balanced_labels = np.array(balanced_labels)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(balanced_labels))
    balanced_waveforms = balanced_waveforms[shuffle_idx]
    balanced_labels = balanced_labels[shuffle_idx]
    
    print(f"\nBalanced dataset: {len(balanced_labels)} total samples")
    
    return balanced_waveforms, balanced_labels


def augment_dataset(waveforms, labels, augmentation_factor=1.0):
    """
    Augment the entire dataset by generating additional samples.
    
    Args:
        waveforms: Array of audio waveforms
        labels: Array of labels
        augmentation_factor: Ratio of augmented samples to add (1.0 = double the dataset)
        
    Returns:
        Tuple of (augmented_waveforms, augmented_labels)
    """
    from tqdm import tqdm
    
    n_original = len(waveforms)
    n_to_generate = int(n_original * augmentation_factor)
    
    print(f"Generating {n_to_generate} augmented samples...")
    
    aug_waveforms = list(waveforms)
    aug_labels = list(labels)
    
    # Generate augmented samples
    indices = np.random.choice(n_original, size=n_to_generate, replace=True)
    
    for idx in tqdm(indices, desc="Augmenting"):
        augmented = augment_sample(waveforms[idx])
        aug_waveforms.append(augmented)
        aug_labels.append(labels[idx])
    
    return np.array(aug_waveforms), np.array(aug_labels)


if __name__ == "__main__":
    # Test augmentation functions
    test_signal = np.random.randn(AUDIO_LENGTH).astype(np.float32)
    
    print("Testing augmentation functions...")
    print(f"Original shape: {test_signal.shape}")
    
    shifted = time_shift(test_signal)
    print(f"Time shifted shape: {shifted.shape}")
    
    pitched = pitch_shift(test_signal)
    print(f"Pitch shifted shape: {pitched.shape}")
    
    stretched = time_stretch(test_signal)
    print(f"Time stretched shape: {stretched.shape}")
    
    combined = augment_sample(test_signal)
    print(f"Combined augmentation shape: {combined.shape}")
    
    print("\nAll augmentation tests passed!")
