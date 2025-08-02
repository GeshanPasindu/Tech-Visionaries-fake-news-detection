
import torch
import os

# --- Paths and Directories ---

BASE_DIR = 'C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/' # From src/, this goes up to the lratf_deepfake_detector/ root
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
FRAMES_DIR = os.path.join(PROCESSED_DATA_DIR, 'frames')
LANDMARKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'landmarks')
METADATA_PATH = os.path.join(BASE_DIR, 'data/raw/DFDC/metadata.json')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models/student_lratf/')
MOTION_VECTORS_DIR = os.path.join(PROCESSED_DATA_DIR, 'motion_vectors')

# --- Training Hyperparameters ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
NUM_EPOCHS = 350
NUM_WORKERS = 4



# --- Model & Data Settings ---
# The number of frames to sample from each video during training
SEQUENCE_LENGTH =32
# The size of the processed frames
IMAGE_SIZE = 224
# Split ratio for the dataset
TRAIN_SPLIT = 0.80
VALID_SPLIT = 0.10
BACKBONE_OUT_FEATURES = 576


os.makedirs(MODEL_SAVE_DIR, exist_ok=True)



