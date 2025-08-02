import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Import necessary components from your project files
from model import LRATF
from data_loader import DeepfakeDataset
from config import *
from torchvision import transforms
from torch.utils.data import DataLoader

def plot_confusion_matrix(cm, save_path, title='Confusion Matrix'):
    """Saves a visual heatmap of the confusion matrix."""
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    ax.set_title(title)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def get_test_loader():
    """
    Creates a DataLoader for the test set.
    """
    print("Loading and preparing test data...")
    metadata = pd.read_json(TEST_METADATA_PATH).T
    df = metadata.copy()
    df['label'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

    # Filter for the 'test' split
    test_df = df[df['split'] == 'test'].copy()

    def is_valid_sample(video_id):
        video_basename = os.path.splitext(video_id)[0]
        frame_dir_path = os.path.normpath(os.path.join(TEST_FRAMES_DIR, video_basename))
        landmark_dir_path = os.path.normpath(os.path.join(TEST_LANDMARKS_DIR, video_basename))
        motion_vector_path = os.path.normpath(os.path.join(TEST_MOTION_VECTORS_DIR, f'{video_basename}_mv.npy'))
        
        return (os.path.isdir(frame_dir_path) and
                os.path.isdir(landmark_dir_path) and
                os.path.exists(motion_vector_path) and
                len(os.listdir(frame_dir_path)) > 0)

    print("Validating test data files...")
    test_df['is_valid'] = test_df.index.to_series().apply(is_valid_sample)
    valid_test_df = test_df[test_df['is_valid']].copy()
    print(f"Found {len(valid_test_df)} valid videos in the test set.")

    if len(valid_test_df) == 0:
        print("No valid test data found. Exiting.")
        return None

    # Define the transformation for the test set (no data augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = DeepfakeDataset(video_ids=valid_test_df.index.tolist(), labels=valid_test_df['label'].tolist(), transform=test_transform)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return test_loader

def test_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, landmarks, motion_vectors, labels in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            landmarks = landmarks.to(device)
            motion_vectors = motion_vectors.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(frames, landmarks, motion_vectors)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate metrics
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.5  # Cannot calculate AUC if only one class is present

    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, binary_preds)
    cm = confusion_matrix(all_labels, binary_preds)
    report = classification_report(all_labels, binary_preds, target_names=['REAL', 'FAKE'], zero_division=0)

    return acc, auc, cm, report

def main():
    # --- IMPORTANT: SET THE PATH TO YOUR BEST SAVED MODEL ---
    MODEL_PATH = "C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/models/student_lratf/best_model_epoch_43_auc_0.8442.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please update the MODEL_PATH variable with the correct path to your trained .pth file.")
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")

    # Initialize the model architecture
    model = LRATF().to(DEVICE)
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # Get the test data loader
    test_loader = get_test_loader()
    
    if test_loader is None:
        return

    # Run the evaluation
    test_acc, test_auc, test_cm, test_report = test_model(model, test_loader, DEVICE)

    # Print the final results
    print("\n" + "="*30)
    print("      FINAL MODEL EVALUATION      ")
    print("="*30)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print("\nTest Classification Report:")
    print(test_report)
    
    print("\nTest Confusion Matrix:")
    print(test_cm)
    
    # Save the confusion matrix as an image
    cm_save_path = os.path.join(MODEL_SAVE_DIR, 'final_test_confusion_matrix.png')
    plot_confusion_matrix(test_cm, cm_save_path, title='Final Test Set Confusion Matrix')

if __name__ == "__main__":
    main()