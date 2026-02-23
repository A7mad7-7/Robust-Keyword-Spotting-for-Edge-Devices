"""
AudioProcessor - Unified class for audio preprocessing and feature extraction.
Designed for easy reuse in both training and real-time inference.
"""
import os
import numpy as np
import librosa
import json

from config import (
    SAMPLE_RATE, AUDIO_LENGTH, N_MELS, N_FFT, HOP_LENGTH, WIN_LENGTH,
    PRE_EMPHASIS_COEF, NOISE_INJECTION_RATIO, SNR_MIN, SNR_MAX, RANDOM_SEED
)


class AudioProcessor:
    """
    Unified audio processor for preprocessing and feature extraction.
    
    This class handles:
    - Audio loading and resampling
    - Padding/truncating to fixed length
    - Pre-emphasis filtering
    - Log Mel-Spectrogram extraction
    - Z-score normalization
    - Noise injection
    
    Designed for easy reuse in real-time inference.
    
    Usage:
        # Training
        processor = AudioProcessor()
        features = processor.process(audio_file_or_array)
        processor.fit_scaler(all_train_features)
        normalized = processor.normalize(features)
        processor.save('processor_config.json')
        
        # Inference
        processor = AudioProcessor.load('processor_config.json')
        features = processor.process(audio_input)
        normalized = processor.normalize(features)
    """
    
    def __init__(self, 
                 sample_rate=SAMPLE_RATE,
                 audio_length=AUDIO_LENGTH,
                 n_mels=N_MELS,
                 n_fft=N_FFT,
                 hop_length=HOP_LENGTH,
                 win_length=WIN_LENGTH,
                 pre_emphasis_coef=PRE_EMPHASIS_COEF):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            audio_length: Target audio length in samples
            n_mels: Number of mel filterbanks
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            pre_emphasis_coef: Pre-emphasis filter coefficient
        """
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pre_emphasis_coef = pre_emphasis_coef
        
        # Scaler parameters (computed from training data)
        self.scaler_mean = None
        self.scaler_std = None
        
        # Calculate expected feature shape
        n_frames = 1 + (audio_length - n_fft) // hop_length
        self.feature_shape = (n_mels, n_frames)
        
        np.random.seed(RANDOM_SEED)
    
    def load_audio(self, audio_input):
        """
        Load audio from file path or accept numpy array.
        
        Args:
            audio_input: File path (str) or numpy array
            
        Returns:
            Numpy array of audio samples
        """
        if isinstance(audio_input, str):
            # Check if it's a .npy file (silence segments)
            if audio_input.endswith('.npy'):
                return np.load(audio_input)
            
            # Load audio file and resample to target sample rate
            audio, sr = librosa.load(audio_input, sr=self.sample_rate)
            return audio
        elif isinstance(audio_input, np.ndarray):
            return audio_input.astype(np.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")
    
    def fix_length(self, waveform):
        """
        Pad or truncate waveform to exactly audio_length samples.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Waveform with exactly audio_length samples
        """
        if len(waveform) > self.audio_length:
            # Truncate (center crop for better coverage)
            start = (len(waveform) - self.audio_length) // 2
            return waveform[start:start + self.audio_length]
        elif len(waveform) < self.audio_length:
            # Pad with zeros (center padding)
            pad_left = (self.audio_length - len(waveform)) // 2
            pad_right = self.audio_length - len(waveform) - pad_left
            return np.pad(waveform, (pad_left, pad_right), mode='constant')
        return waveform
    
    def apply_pre_emphasis(self, waveform):
        """
        Apply pre-emphasis filter to boost high frequencies.
        
        Pre-emphasis: y[n] = x[n] - coef * x[n-1]
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Pre-emphasized waveform
        """
        return np.append(waveform[0], waveform[1:] - self.pre_emphasis_coef * waveform[:-1])
    
    def preprocess(self, waveform):
        """
        Full preprocessing pipeline: fix length + pre-emphasis.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Preprocessed waveform of shape (audio_length,)
        """
        # Fix length first
        waveform = self.fix_length(waveform)
        
        # Apply pre-emphasis
        waveform = self.apply_pre_emphasis(waveform)
        
        return waveform
    
    def extract_features(self, waveform):
        """
        Extract Log Mel-Spectrogram features from waveform.
        
        Args:
            waveform: Preprocessed audio waveform
            
        Returns:
            Log Mel-Spectrogram of shape (n_mels, n_frames)
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmax=self.sample_rate // 2
        )
        
        # Convert to log scale (add small epsilon to avoid log(0))
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def normalize(self, features):
        """
        Apply Z-score normalization using training statistics.
        
        Args:
            features: Feature array to normalize
            
        Returns:
            Normalized features with mean≈0, std≈1
        """
        if self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        return (features - self.scaler_mean) / (self.scaler_std + 1e-8)
    
    def fit_scaler(self, train_features):
        """
        Compute normalization statistics from training data.
        
        Args:
            train_features: Array of training features, shape (n_samples, n_mels, n_frames)
        """
        # Compute mean and std across all samples and time frames
        # Result shape: (n_mels, 1) for broadcasting
        self.scaler_mean = np.mean(train_features, axis=(0, 2), keepdims=True)
        self.scaler_std = np.std(train_features, axis=(0, 2), keepdims=True)
        
        # Reshape for easier broadcasting
        self.scaler_mean = self.scaler_mean.squeeze(0)  # (n_mels, 1)
        self.scaler_std = self.scaler_std.squeeze(0)    # (n_mels, 1)
        
        print(f"Scaler fitted - Mean shape: {self.scaler_mean.shape}, Std shape: {self.scaler_std.shape}")
    
    def process(self, audio_input, normalize=False):
        """
        Full pipeline: load → preprocess → extract features (→ normalize).
        
        This is the main method for real-time inference.
        
        Args:
            audio_input: File path or numpy array
            normalize: Whether to apply normalization (requires fitted scaler)
            
        Returns:
            Log Mel-Spectrogram features
        """
        # Load audio
        waveform = self.load_audio(audio_input)
        
        # Preprocess
        waveform = self.preprocess(waveform)
        
        # Extract features
        features = self.extract_features(waveform)
        
        # Optionally normalize
        if normalize:
            features = self.normalize(features)
        
        return features
    
    def process_batch(self, audio_inputs, normalize=False, show_progress=True):
        """
        Process a batch of audio inputs.
        
        Args:
            audio_inputs: List of file paths or numpy arrays
            normalize: Whether to apply normalization
            show_progress: Whether to show progress bar
            
        Returns:
            Array of features, shape (n_samples, n_mels, n_frames)
        """
        from tqdm import tqdm
        
        iterator = tqdm(audio_inputs, desc="Processing audio") if show_progress else audio_inputs
        
        features = []
        for audio_input in iterator:
            feat = self.process(audio_input, normalize=normalize)
            features.append(feat)
        
        return np.array(features)
    
    def save(self, path):
        """
        Save processor configuration and scaler parameters.
        
        Args:
            path: Path to save the configuration (JSON file)
        """
        config = {
            'sample_rate': self.sample_rate,
            'audio_length': self.audio_length,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'pre_emphasis_coef': self.pre_emphasis_coef,
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Processor configuration saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load processor from saved configuration.
        
        Args:
            path: Path to the saved configuration
            
        Returns:
            AudioProcessor instance with loaded configuration
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        processor = cls(
            sample_rate=config['sample_rate'],
            audio_length=config['audio_length'],
            n_mels=config['n_mels'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length'],
            pre_emphasis_coef=config['pre_emphasis_coef']
        )
        
        if config['scaler_mean'] is not None:
            processor.scaler_mean = np.array(config['scaler_mean'])
            processor.scaler_std = np.array(config['scaler_std'])
        
        print(f"Processor configuration loaded from {path}")
        return processor
    
    def get_feature_shape(self):
        """Return the expected feature shape."""
        return self.feature_shape


# Noise injection functions (separate from class for training pipeline)

def inject_noise(waveform, noise_waveform, snr_db):
    """
    Mix background noise with audio at specified SNR.
    
    Args:
        waveform: Clean audio waveform
        noise_waveform: Background noise waveform (same length)
        snr_db: Signal-to-Noise Ratio in dB
        
    Returns:
        Noisy waveform
    """
    # Calculate signal and noise power
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise_waveform ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        return waveform
    
    # Calculate required noise scaling factor
    # SNR = 10 * log10(signal_power / noise_power)
    # noise_scale = sqrt(signal_power / (10^(snr_db/10) * noise_power))
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    
    # Mix signal and scaled noise
    noisy = waveform + noise_scale * noise_waveform
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    return noisy


def load_all_noise_files(noise_paths, sample_rate=SAMPLE_RATE):
    """
    Load all noise files into memory as full waveforms.
    
    Args:
        noise_paths: List of paths to noise files
        sample_rate: Target sample rate
        
    Returns:
        List of numpy arrays containing full noise waveforms
    """
    noise_waveforms = []
    print(f"Loading {len(noise_paths)} background noise files...")
    
    for path in noise_paths:
        try:
            # Load full file
            noise, _ = librosa.load(path, sr=sample_rate)
            noise_waveforms.append(noise)
        except Exception as e:
            print(f"Warning: Failed to load noise file {path}: {e}")
            
    return noise_waveforms


def create_noisy_dataset(clean_waveforms, noise_waveforms, injection_ratio=NOISE_INJECTION_RATIO,
                         snr_min=SNR_MIN, snr_max=SNR_MAX):
    """
    Create noisy versions of audio samples with dynamic slicing.
    
    Args:
        clean_waveforms: Array of clean waveforms
        noise_waveforms: List of FULL noise waveforms (not paths)
        injection_ratio: Ratio of samples to add noise to
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        
    Returns:
        Tuple of (noisy_waveforms, noise_mask) where noise_mask indicates which samples have noise
    """
    if not noise_waveforms:
        print("Warning: No background noise loaded! Skipping noise injection.")
        return clean_waveforms, np.zeros(len(clean_waveforms), dtype=bool)

    n_samples = len(clean_waveforms)
    target_length = len(clean_waveforms[0])
    
    noisy_waveforms = []
    noise_mask = np.zeros(n_samples, dtype=bool)
    
    # Determine which samples get noise
    n_noisy = int(n_samples * injection_ratio)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)
    noise_mask[noisy_indices] = True
    
    for i, waveform in enumerate(clean_waveforms):
        if noise_mask[i]:
            # 1. Select random noise file
            noise_full = noise_waveforms[np.random.randint(len(noise_waveforms))]
            
            # 2. Slice random segment
            if len(noise_full) > target_length:
                start = np.random.randint(0, len(noise_full) - target_length)
                noise_segment = noise_full[start:start + target_length]
            else:
                # Pad if noise is shorter than target (unlikely for background noise files but safe to handle)
                noise_segment = np.pad(noise_full, (0, target_length - len(noise_full)), mode='wrap')
            
            # 3. Inject at random SNR
            snr = np.random.uniform(snr_min, snr_max)
            noisy = inject_noise(waveform, noise_segment, snr)
            noisy_waveforms.append(noisy)
        else:
            noisy_waveforms.append(waveform)
    
    return np.array(noisy_waveforms), noise_mask


if __name__ == "__main__":
    # Test the AudioProcessor
    processor = AudioProcessor()
    
    print(f"AudioProcessor initialized:")
    print(f"  Sample rate: {processor.sample_rate} Hz")
    print(f"  Audio length: {processor.audio_length} samples ({processor.audio_length/processor.sample_rate} seconds)")
    print(f"  Feature shape: {processor.feature_shape}")
    
    # Test with a synthetic signal
    test_signal = np.random.randn(16000).astype(np.float32)
    features = processor.process(test_signal)
    print(f"\nTest signal processed:")
    print(f"  Input shape: {test_signal.shape}")
    print(f"  Output shape: {features.shape}")
