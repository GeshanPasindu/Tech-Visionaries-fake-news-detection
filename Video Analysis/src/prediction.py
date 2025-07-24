import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import av
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# Required for models and transforms
from torchvision.ops import roi_align
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from facenet_pytorch import MTCNN

# Required for MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
TEMP_DIR = "temp_prediction_data"
SEQUENCE_LENGTH = 128  # This should match the sequence length used in training
IMAGE_SIZE = 224
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Model Definition ---
# <<< MODIFIED: This must EXACTLY match the model used for training >>>
class MotionVectorEncoder(nn.Module):
    def __init__(self, input_dim=2, embedding_dim=64):
        super(MotionVectorEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, embedding_dim))
    def forward(self, mv_sequence):
        b, s, n, d = mv_sequence.shape
        mv_sequence = mv_sequence.view(-1, 2)
        embeddings = self.mlp(mv_sequence).view(b * s, n, -1)
        pooled, _ = torch.max(embeddings, dim=1)
        return pooled.view(b, s, -1)

class LRATF(nn.Module):
    def __init__(self, num_classes=1, backbone_out_features=576):
        super(LRATF, self).__init__()
        # Use pre-trained weights for the feature extractor
        self.feature_extractor = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        
        # LSTMs without dropout for loading a model trained to fix underfitting
        self.mouth_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.nose_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.eyes_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.eyebrows_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.chin_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        
        self.mv_encoder = MotionVectorEncoder(input_dim=2, embedding_dim=64)
        self.mv_lstm = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)

        self.attention_embedding_dim = 128
        self.visual_proj = nn.Linear(128, self.attention_embedding_dim) 
        self.motion_proj = nn.Linear(64, self.attention_embedding_dim)

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.attention_embedding_dim, num_heads=4, batch_first=True)

        # Classifier without dropout
        self.classifier = nn.Sequential(
            nn.Linear(6 * self.attention_embedding_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def get_region_boxes(self, landmarks, img_shape):
        h, w = img_shape
        regions = [
            landmarks[:, 61:80, :2], landmarks[:, 291:300, :2], landmarks[:, 362:388, :2],
            torch.cat([landmarks[:, 63:70, :2], landmarks[:, 293:300, :2]], dim=1),
            landmarks[:, [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 397], :2]
        ]
        boxes = []
        for region_pts in regions:
            x_coords, y_coords = region_pts[:, :, 0] * w, region_pts[:, :, 1] * h
            boxes.append(torch.stack([torch.min(x_coords, dim=1)[0], torch.min(y_coords, dim=1)[0], 
                                      torch.max(x_coords, dim=1)[0], torch.max(y_coords, dim=1)[0]], dim=1))
        return boxes

    def forward(self, x, landmarks, motion_vectors):
        b, s, _, h, w = x.shape
        regional_features_seq = [[] for _ in range(5)]
        for t in range(s):
            features = self.feature_extractor(x[:, t])
            boxes = self.get_region_boxes(landmarks[:, t], (h, w))
            box_indices = torch.arange(b, device=x.device).view(-1, 1)
            for i in range(5):
                rois = torch.cat([box_indices, boxes[i]], dim=1)
                aligned_features = roi_align(features, rois, output_size=(1, 1), spatial_scale=1.0 / 16.0).squeeze()
                regional_features_seq[i].append(aligned_features)
        
        sequences = [torch.stack(seq, dim=1) for seq in regional_features_seq]
        
        _, (h_mouth, _) = self.mouth_lstm(sequences[0])
        _, (h_nose, _) = self.nose_lstm(sequences[1])
        _, (h_eyes, _) = self.eyes_lstm(sequences[2])
        _, (h_eyebrows, _) = self.eyebrows_lstm(sequences[3])
        _, (h_chin, _) = self.chin_lstm(sequences[4])
        
        mv_embeddings = self.mv_encoder(motion_vectors)
        _, (h_mv, _) = self.mv_lstm(mv_embeddings)

        h_streams = [
            torch.cat((h_mouth[-2,:,:], h_mouth[-1,:,:]), dim=1), torch.cat((h_nose[-2,:,:], h_nose[-1,:,:]), dim=1),
            torch.cat((h_eyes[-2,:,:], h_eyes[-1,:,:]), dim=1), torch.cat((h_eyebrows[-2,:,:], h_eyebrows[-1,:,:]), dim=1),
            torch.cat((h_chin[-2,:,:], h_chin[-1,:,:]), dim=1), torch.cat((h_mv[-2,:,:], h_mv[-1,:,:]), dim=1)
        ]

        proj_streams = [self.visual_proj(s) for s in h_streams[:5]] + [self.motion_proj(h_streams[5])]
        
        attention_input = torch.stack(proj_streams, dim=1)
        attention_output, _ = self.attention_layer(attention_input, attention_input, attention_input)
        
        output = self.classifier(attention_output.flatten(start_dim=1))
        return output

# --- Pre-processing Functions ---

def simulate_low_quality(frame):
    # <<< MODIFIED: Disabled for prediction to match training on high-quality data >>>
    return frame

def extract_motion_vectors(video_path, output_path):
    # This function remains the same as it extracts raw data.
    try:
        container = av.open(video_path, options={"flags2": "+export_mvs"})
        stream = container.streams.video[0]
        all_frames_mv = [np.zeros((512, 2), dtype=np.int16) if f.pict_type == av.video.frame.PictureType.I else f.side_data.get('MOTION_VECTORS').to_ndarray()[:, [5, 6]] for f in container.decode(stream)]
        
        processed_mvs = []
        for vectors in all_frames_mv:
            if vectors.shape[0] > 512:
                processed_mvs.append(vectors[:512].astype(np.int16))
            else:
                padding = np.zeros((512 - vectors.shape[0], 2), dtype=np.int16)
                processed_mvs.append(np.vstack([vectors, padding]))
        
        container.close()
        np.save(output_path, np.array(processed_mvs))
        return True
    except Exception as e:
        print(f"Error extracting motion vectors: {e}")
        return False

# <<< MODIFIED: This function now uses the MTCNN + MediaPipe workflow >>>
def process_video_with_mtcnn(video_path, frames_dir, landmarks_dir, mtcnn, landmarker):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 32: # A minimum number of frames
            return False

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(landmarks_dir, exist_ok=True)
        
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, SEQUENCE_LENGTH), dtype=int)
        frames_saved = 0

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                margin = 20
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
                
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0: continue
                
                final_frame = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                
                # Use high-quality frame for landmark detection
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
                detection_result = landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    cv2.imwrite(os.path.join(frames_dir, f'frame_{i}.jpg'), final_frame)
                    
                    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.face_landmarks[0]])
                    np.save(os.path.join(landmarks_dir, f'landmarks_{i}.npy'), landmarks_array)
                    frames_saved += 1
        
        cap.release()
        return frames_saved > 0
    except Exception as e:
        print(f"Error in process_video_with_mtcnn: {e}")
        return False

# --- Main Prediction Logic ---
def predict_on_video(video_path, model_path, device):
    print("--- 1. Initializing Models ---")
    
    # <<< MODIFIED: Initialize MTCNN here >>>
    mtcnn = MTCNN(keep_all=False, device=device)
    
    # Initialize MediaPipe (CPU for Windows compatibility)
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    # Initialize the model and load weights
    model = LRATF().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print("This may be due to a mismatch between the saved model and the current model architecture.")
        return
    model.eval()

    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print(f"\n--- 2. Pre-processing video: {os.path.basename(video_path)} ---")
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    frames_dir = os.path.join(TEMP_DIR, 'frames')
    landmarks_dir = os.path.join(TEMP_DIR, 'landmarks')
    mv_path = os.path.join(TEMP_DIR, 'motion_vectors.npy')

    # <<< MODIFIED: Call the new processing functions >>>
    success_fl = process_video_with_mtcnn(video_path, frames_dir, landmarks_dir, mtcnn, landmarker)
    success_mv = extract_motion_vectors(video_path, mv_path)

    if not (success_fl and success_mv):
        print("❌ Pre-processing failed. Cannot make a prediction.")
        shutil.rmtree(TEMP_DIR)
        return

    print("\n--- 3. Loading processed data into tensors ---")
    frames, landmarks_list = [], []
    frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    landmark_files = sorted(os.listdir(landmarks_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

    for frame_file, landmark_file in zip(frame_files, landmark_files):
        frame = Image.open(os.path.join(frames_dir, frame_file)).convert("RGB")
        frames.append(val_transform(frame))
        landmarks_list.append(np.load(os.path.join(landmarks_dir, landmark_file)))

    all_motion_vectors = np.load(mv_path)
    motion_vectors = all_motion_vectors[:len(frames)]

    padding_needed = SEQUENCE_LENGTH - len(frames)
    if padding_needed > 0:
        if not frames: # Handle case where no frames were processed
            print("❌ No valid frames to process after preprocessing.")
            return
        frames.extend([torch.zeros_like(frames[0])] * padding_needed)
        landmarks_list.extend([np.zeros_like(landmarks_list[0])] * padding_needed)
        motion_vectors = np.concatenate([motion_vectors, np.zeros((padding_needed, 512, 2))])

    frames_tensor = torch.stack(frames[:SEQUENCE_LENGTH]).unsqueeze(0).to(device)
    landmarks_tensor = torch.from_numpy(np.array(landmarks_list[:SEQUENCE_LENGTH])).float().unsqueeze(0).to(device)
    motion_vectors_tensor = torch.from_numpy(motion_vectors[:SEQUENCE_LENGTH]).float().unsqueeze(0).to(device)

    print("\n--- 4. Running model inference ---")
    with torch.no_grad():
        output = model(frames_tensor, landmarks_tensor, motion_vectors_tensor)
        confidence_score = torch.sigmoid(output).item()

    prediction = "FAKE" if confidence_score > 0.5 else "REAL"
    print("\n" + "="*30)
    print("          PREDICTION RESULT")
    print("="*30)
    print(f"  VIDEO: {os.path.basename(video_path)}")
    print(f"  PREDICTION: {prediction}")
    print(f"  CONFIDENCE (as FAKE): {confidence_score:.2%}")
    print("="*30 + "\n")

    shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    # Download MediaPipe model if it doesn't exist
    if not os.path.exists('face_landmarker.task'):
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(model_url, 'face_landmarker.task')

    root = tk.Tk()
    root.withdraw()
    
    print("Please provide the path to the trained model weights (.pth file)...")
    MODEL_PATH = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Models", "*.pth")])

    if not MODEL_PATH:
        print("No model file selected. Exiting.")
    else:
        print("Please select a video file to analyze...")
        video_to_test = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*")])
        
        if not video_to_test:
            print("No video file selected. Exiting.")
        else:
            predict_on_video(video_path=video_to_test, model_path=MODEL_PATH, device=DEVICE)