"""
Data loader for Google Speech Commands dataset.
Handles loading file paths, creating special classes, and stratified splitting.
"""
import os
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

import librosa

from config import (
    RAW_DATA_DIR, SILENCE_DIR, TARGET_KEYWORDS, UNKNOWN_LABEL, SILENCE_LABEL,
    SAMPLE_RATE, AUDIO_LENGTH, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    LABEL_TO_IDX, RANDOM_SEED
)


class SpeechCommandsDataLoader:
    """
    Data loader for the Google Speech Commands dataset.
    
    Handles:
    - Loading file paths for target keywords
    - Creating 'unknown' class from remaining words
    - Creating 'silence' class from background noise
    - Stratified train/val/test splitting
    """
    
    def __init__(self, data_dir=RAW_DATA_DIR, silence_dir=SILENCE_DIR, random_seed=RANDOM_SEED):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the extracted speech commands dataset
            silence_dir: Path to save sliced silence segments
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.silence_dir = silence_dir
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.file_paths = []
        self.labels = []
        self.background_noise_files = []
    
    def load_file_paths(self):
        """
        Scan the dataset directory and collect all audio file paths.
        
        Returns:
            Tuple of (file_paths, labels)
        """
        print("Loading file paths...")
        
        target_files = defaultdict(list)
        unknown_files = []
        
        # Get all word directories
        for word_dir in os.listdir(self.data_dir):
            word_path = os.path.join(self.data_dir, word_dir)
            
            if not os.path.isdir(word_path):
                continue
            
            # Handle background noise separately
            if word_dir == '_background_noise_':
                for f in os.listdir(word_path):
                    if f.endswith('.wav'):
                        self.background_noise_files.append(os.path.join(word_path, f))
                continue
            
            # Get all wav files in this directory
            wav_files = [os.path.join(word_path, f) 
                        for f in os.listdir(word_path) 
                        if f.endswith('.wav')]
            
            if word_dir in TARGET_KEYWORDS:
                target_files[word_dir].extend(wav_files)
            else:
                unknown_files.extend([(f, UNKNOWN_LABEL) for f in wav_files])
        
        # Add target keyword files
        for keyword in TARGET_KEYWORDS:
            for filepath in target_files[keyword]:
                self.file_paths.append(filepath)
                self.labels.append(keyword)
        
        print(f"Found {len(self.file_paths)} target keyword samples")
        print(f"Found {len(unknown_files)} unknown word samples")
        print(f"Found {len(self.background_noise_files)} background noise files")
        
        return target_files, unknown_files
    
    def create_unknown_class(self, unknown_files, sample_ratio=0.1):
        """
        Sample from non-target keywords to create the 'unknown' class.
        
        Args:
            unknown_files: List of tuples (filepath, label) for unknown words
            sample_ratio: Ratio of unknown samples to include (relative to target class size)
            
        Returns:
            Number of unknown samples added
        """
        print("Creating 'unknown' class...")
        
        # Calculate average number of samples per target keyword
        current_samples = len(self.file_paths)
        avg_per_class = current_samples // len(TARGET_KEYWORDS)
        
        # Sample approximately the same number of unknown samples
        n_unknown_samples = min(avg_per_class, len(unknown_files))
        
        # Random sample
        sampled = random.sample(unknown_files, n_unknown_samples)
        
        for filepath, label in sampled:
            self.file_paths.append(filepath)
            self.labels.append(label)
        
        print(f"Added {n_unknown_samples} 'unknown' samples")
        return n_unknown_samples
    
    def create_silence_class(self):
        """
        Slice background noise files into 1-second segments to create 'silence' class.
        
        Returns:
            Number of silence samples created
        """
        print("Creating 'silence' class from background noise...")
        
        os.makedirs(self.silence_dir, exist_ok=True)
        
        silence_count = 0
        segment_length = AUDIO_LENGTH  # 1 second = 16000 samples
        
        for noise_file in self.background_noise_files:
            # Load the full audio file
            audio, sr = librosa.load(noise_file, sr=SAMPLE_RATE)
            
            # Calculate number of full 1-second segments
            n_segments = len(audio) // segment_length
            
            # Extract each segment
            for i in range(n_segments):
                start = i * segment_length
                end = start + segment_length
                segment = audio[start:end]
                
                # Save segment
                segment_filename = f"silence_{os.path.basename(noise_file).replace('.wav', '')}_{i}.npy"
                segment_path = os.path.join(self.silence_dir, segment_filename)
                
                # Save as numpy array for faster loading later
                np.save(segment_path, segment)
                
                self.file_paths.append(segment_path)
                self.labels.append(SILENCE_LABEL)
                silence_count += 1
        
        print(f"Created {silence_count} 'silence' segments")
        return silence_count
    
    def stratified_split(self):
        """
        Split data into train (70%), val (20%), test (10%) using stratified split.
        
        Returns:
            Dictionary with train/val/test file paths and labels
        """
        print("Performing stratified split...")
        
        # Convert labels to indices for stratification
        label_indices = [LABEL_TO_IDX[label] for label in self.labels]
        
        # First split: separate test set (10%)
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            self.file_paths, label_indices,
            test_size=TEST_SPLIT,
            stratify=label_indices,
            random_state=self.random_seed
        )
        
        # Second split: separate val from train (20% of original = 20/90 of remaining)
        val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=self.random_seed
        )
        
        # Print split statistics
        print(f"\nSplit Statistics:")
        print(f"  Train: {len(train_files)} samples ({len(train_files)/len(self.file_paths)*100:.1f}%)")
        print(f"  Val:   {len(val_files)} samples ({len(val_files)/len(self.file_paths)*100:.1f}%)")
        print(f"  Test:  {len(test_files)} samples ({len(test_files)/len(self.file_paths)*100:.1f}%)")
        
        # Print class distribution
        self._print_class_distribution({
            'train': train_labels,
            'val': val_labels,
            'test': test_labels
        })
        
        return {
            'train_files': train_files,
            'train_labels': np.array(train_labels),
            'val_files': val_files,
            'val_labels': np.array(val_labels),
            'test_files': test_files,
            'test_labels': np.array(test_labels)
        }
    
    def _print_class_distribution(self, splits_dict):
        """Print class distribution for each split."""
        from config import IDX_TO_LABEL
        
        print("\nClass Distribution:")
        for split_name, labels in splits_dict.items():
            print(f"\n  {split_name.upper()}:")
            unique, counts = np.unique(labels, return_counts=True)
            for idx, count in zip(unique, counts):
                print(f"    {IDX_TO_LABEL[idx]}: {count}")
    
    def get_background_noise_paths(self):
        """Return paths to background noise files for noise injection."""
        return self.background_noise_files
    
    def prepare_dataset(self):
        """
        Full pipeline to prepare the dataset.
        
        Returns:
            Dictionary with all split data and metadata
        """
        # Load file paths
        target_files, unknown_files = self.load_file_paths()
        
        # Create special classes
        self.create_unknown_class(unknown_files)
        self.create_silence_class()
        
        # Perform stratified split
        splits = self.stratified_split()
        
        # Add background noise paths for noise injection
        splits['background_noise_files'] = self.background_noise_files
        
        return splits


if __name__ == "__main__":
    # Test the data loader
    loader = SpeechCommandsDataLoader()
    splits = loader.prepare_dataset()
    
    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print(f"Total files: {len(loader.file_paths)}")
