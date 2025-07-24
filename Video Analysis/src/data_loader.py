import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from config import *

class DeepfakeDataset(Dataset):
    def __init__(self, video_ids, labels, transform=None):
        self.transform = transform
        self.video_ids = video_ids
        self.labels = labels
        print(f"Initialized dataset split. Found {len(self.video_ids)} valid videos.")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        label = self.labels[idx]

        video_basename = os.path.splitext(video_id)[0]
        frame_folder = os.path.normpath(os.path.join(FRAMES_DIR, video_basename))
        landmark_folder = os.path.normpath(os.path.join(LANDMARKS_DIR, video_basename))
        motion_vector_path = os.path.normpath(os.path.join(MOTION_VECTORS_DIR, f'{video_basename}_mv.npy'))
        
        frame_files = sorted(os.listdir(frame_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
        landmark_files = sorted(os.listdir(landmark_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
        all_motion_vectors = np.load(motion_vector_path, allow_pickle=True)

        num_available_items = min(len(frame_files), len(landmark_files), len(all_motion_vectors))

        start_index = 0
        if num_available_items > SEQUENCE_LENGTH:
            start_index = random.randint(0, num_available_items - SEQUENCE_LENGTH)
        
        end_index = min(start_index + SEQUENCE_LENGTH, num_available_items)

        frames, landmarks_list = [], []
        
        frame_files_to_load = frame_files[start_index:end_index]
        landmark_files_to_load = landmark_files[start_index:end_index]

        for frame_file, landmark_file in zip(frame_files_to_load, landmark_files_to_load):
            frame_path = os.path.join(frame_folder, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            
            landmark_path = os.path.join(landmark_folder, landmark_file)
            landmarks_list.append(np.load(landmark_path))

        landmarks = np.array(landmarks_list) if landmarks_list else np.empty((0, 478, 3))
        motion_vectors = all_motion_vectors[start_index:end_index]

        padding_needed = SEQUENCE_LENGTH - len(frames)
        if padding_needed > 0:
            if frames:
                black_frame = torch.zeros_like(frames[0])
                frames.extend([black_frame] * padding_needed)
            else:
                dummy_frame = torch.zeros((3, 224, 224))
                if self.transform and hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if isinstance(t, transforms.Normalize):
                            dummy_frame = t(dummy_frame)
                            break
                frames.extend([dummy_frame] * padding_needed)

            lm_padding_shape = (padding_needed, landmarks.shape[1], landmarks.shape[2]) if landmarks.size > 0 else (padding_needed, 478, 3)
            landmarks = np.concatenate([landmarks, np.zeros(lm_padding_shape)], axis=0)
            
            mv_padding_shape = (padding_needed, motion_vectors.shape[1], motion_vectors.shape[2]) if motion_vectors.size > 0 else (padding_needed, 512, 2)
            motion_vectors = np.concatenate([motion_vectors, np.zeros(mv_padding_shape)], axis=0)

        frames_tensor = torch.stack(frames, dim=0) 
        landmarks_tensor = torch.from_numpy(landmarks).float()
        motion_vectors_tensor = torch.from_numpy(motion_vectors).float()

        return frames_tensor, landmarks_tensor, motion_vectors_tensor, torch.tensor(label, dtype=torch.float32)

def get_data_loaders():
    print("Creating data loaders...")
    metadata = pd.read_json(METADATA_PATH).T
    df = metadata.copy()
    df['label'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

    def is_valid_sample(video_id):
        video_basename = os.path.splitext(video_id)[0]
        frame_dir_path = os.path.normpath(os.path.join(FRAMES_DIR, video_basename))
        landmark_dir_path = os.path.normpath(os.path.join(LANDMARKS_DIR, video_basename))
        motion_vector_path = os.path.normpath(os.path.join(MOTION_VECTORS_DIR, f'{video_basename}_mv.npy'))
        
        return (os.path.isdir(frame_dir_path) and
                os.path.isdir(landmark_dir_path) and
                os.path.exists(motion_vector_path))

    print("Validating data files...")
    df['is_valid'] = df.index.to_series().apply(is_valid_sample)
    valid_df = df[df['is_valid']].copy()
    print(f"Found {len(valid_df)} videos with complete data out of {len(df)} total.")
    
    train_val_df, test_df = train_test_split(valid_df, test_size=(1 - TRAIN_SPLIT), random_state=42, stratify=valid_df['label'])
    val_test_ratio = VALID_SPLIT / (1 - TRAIN_SPLIT)
    val_df, test_df = train_test_split(test_df, test_size=(1-val_test_ratio), random_state=42, stratify=test_df['label'])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Using the same transform for train, validation, and test sets
    train_dataset = DeepfakeDataset(video_ids=train_val_df.index.tolist(), labels=train_val_df['label'].tolist(), transform=val_transform)
    val_dataset = DeepfakeDataset(video_ids=val_df.index.tolist(), labels=val_df['label'].tolist(), transform=val_transform)
    test_dataset = DeepfakeDataset(video_ids=test_df.index.tolist(), labels=test_df['label'].tolist(), transform=val_transform)

    class_counts = train_val_df['label'].value_counts().to_dict()
    num_samples = len(train_dataset)
    
    weights = [1.0 / class_counts[label] for label in train_val_df['label'].tolist()]
    sample_weights = torch.DoubleTensor(weights)
    
    sampler = WeightedRandomSampler(sample_weights, num_samples)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Data loading complete. Using stratified sampling for training set.")
    return train_loader, val_loader, test_loader