"""
Configuration settings for the Keyword Spotting Data Pipeline.
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'speech_commands')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SILENCE_DIR = os.path.join(DATA_DIR, 'silence_segments')

# Dataset URL
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Target keywords (10 classes)
TARGET_KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# Special classes
UNKNOWN_LABEL = 'unknown'
SILENCE_LABEL = 'silence'

# Audio settings
SAMPLE_RATE = 16000
AUDIO_LENGTH = 16000  # 1 second at 16kHz

# Feature extraction
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms hop
WIN_LENGTH = 400  # 25ms window

# Pre-emphasis
PRE_EMPHASIS_COEF = 0.97

# Data splits
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Noise injection
NOISE_INJECTION_RATIO = 0.5
SNR_MIN = 0   # dB
SNR_MAX = 10  # dB

# Random seed for reproducibility
RANDOM_SEED = 42

# Label mapping (will be populated during data loading)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(TARGET_KEYWORDS + [UNKNOWN_LABEL, SILENCE_LABEL])}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE_1 = 0.5
DROPOUT_RATE_2 = 0.25
LR_PATIENCE = 5

# Evaluation / Reporting
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
