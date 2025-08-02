import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import av
import shutil
from PIL import Image
from torchvision.ops import roi_align
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from facenet_pytorch import MTCNN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration (can be moved to a separate config.py) ---
TEMP_DIR = "temp_prediction_data"
SEQUENCE_LENGTH = 256
IMAGE_SIZE = 224
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Model Definition ---
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
        self.feature_extractor = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        
        # --- Updated to 10 regional LSTMs ---
        self.upper_lip_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.lower_lip_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_eye_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_eye_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.nose_bridge_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_cheek_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_cheek_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.left_eyebrow_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.right_eyebrow_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        self.chin_jawline_lstm = nn.LSTM(input_size=backbone_out_features, hidden_size=64, bidirectional=True, batch_first=True)
        
        self.mv_encoder = MotionVectorEncoder(input_dim=2, embedding_dim=64)
        self.mv_lstm = nn.LSTM(input_size=64, hidden_size=32, bidirectional=True, batch_first=True)
        
        self.attention_embedding_dim = 128
        self.visual_proj = nn.Linear(128, self.attention_embedding_dim)
        self.motion_proj = nn.Linear(64, self.attention_embedding_dim)
        
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.attention_embedding_dim, num_heads=4, batch_first=True)
        
        # --- Updated classifier input size (10 visual streams + 1 motion stream = 11) ---
        self.classifier = nn.Sequential(
            nn.Linear(11 * self.attention_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def get_region_boxes(self, landmarks, img_shape):
        h, w = img_shape
        regions = [
            landmarks[:, [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], :2],  # Upper Lip
            landmarks[:, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291], :2],  # Lower Lip
            landmarks[:, [362, 382, 381, 380, 374, 373, 390, 249, 263], :2],  # Left Eye
            landmarks[:, [33, 7, 163, 144, 145, 153, 154, 155, 133], :2],  # Right Eye
            landmarks[:, [6, 197, 195, 5, 4, 1, 19, 94, 2], :2],  # Nose Bridge & Tip
            landmarks[:, [200, 426, 431, 411, 444, 396, 266, 350, 446], :2],  # Left Cheek
            landmarks[:, [45, 234, 248, 243, 44, 107, 59, 10, 336], :2],  # Right Cheek
            landmarks[:, [276, 283, 282, 295, 300, 293], :2],  # Left Eyebrow
            landmarks[:, [46, 53, 52, 65, 70, 63], :2],  # Right Eyebrow
            landmarks[:, [152, 172, 176, 148, 150, 149, 176, 172, 152], :2]  # Chin & Jawline
        ]
        
        boxes = []
        for region_pts in regions:
            x_coords, y_coords = region_pts[:, :, 0] * w, region_pts[:, :, 1] * h
            boxes.append(torch.stack([torch.min(x_coords, dim=1)[0], torch.min(y_coords, dim=1)[0],
                                     torch.max(x_coords, dim=1)[0], torch.max(y_coords, dim=1)[0]], dim=1))
        return boxes

    def forward(self, x, landmarks, motion_vectors):
        b, s, _, h, w = x.shape
        regional_features_seq = [[] for _ in range(10)]
        for t in range(s):
            features = self.feature_extractor(x[:, t])
            boxes = self.get_region_boxes(landmarks[:, t], (h, w))
            box_indices = torch.arange(b, device=x.device).view(-1, 1)
            for i in range(10): # Loop over 10 regions
                rois = torch.cat([box_indices, boxes[i]], dim=1)
                aligned_features = roi_align(features, rois.float(), output_size=(1, 1), spatial_scale=1.0 / 16.0).squeeze(-1).squeeze(-1)
                regional_features_seq[i].append(aligned_features)

        sequences = [torch.stack(seq, dim=1) for seq in regional_features_seq]
        
        _, (h_upper_lip, _) = self.upper_lip_lstm(sequences[0])
        _, (h_lower_lip, _) = self.lower_lip_lstm(sequences[1])
        _, (h_left_eye, _) = self.left_eye_lstm(sequences[2])
        _, (h_right_eye, _) = self.right_eye_lstm(sequences[3])
        _, (h_nose_bridge, _) = self.nose_bridge_lstm(sequences[4])
        _, (h_left_cheek, _) = self.left_cheek_lstm(sequences[5])
        _, (h_right_cheek, _) = self.right_cheek_lstm(sequences[6])
        _, (h_left_eyebrow, _) = self.left_eyebrow_lstm(sequences[7])
        _, (h_right_eyebrow, _) = self.right_eyebrow_lstm(sequences[8])
        _, (h_chin_jawline, _) = self.chin_jawline_lstm(sequences[9])
        
        mv_embeddings = self.mv_encoder(motion_vectors)
        _, (h_mv, _) = self.mv_lstm(mv_embeddings)
        
        h_streams = [
            torch.cat((h_upper_lip[-2,:,:], h_upper_lip[-1,:,:]), dim=1),
            torch.cat((h_lower_lip[-2,:,:], h_lower_lip[-1,:,:]), dim=1),
            torch.cat((h_left_eye[-2,:,:], h_left_eye[-1,:,:]), dim=1),
            torch.cat((h_right_eye[-2,:,:], h_right_eye[-1,:,:]), dim=1),
            torch.cat((h_nose_bridge[-2,:,:], h_nose_bridge[-1,:,:]), dim=1),
            torch.cat((h_left_cheek[-2,:,:], h_left_cheek[-1,:,:]), dim=1),
            torch.cat((h_right_cheek[-2,:,:], h_right_cheek[-1,:,:]), dim=1),
            torch.cat((h_left_eyebrow[-2,:,:], h_left_eyebrow[-1,:,:]), dim=1),
            torch.cat((h_right_eyebrow[-2,:,:], h_right_eyebrow[-1,:,:]), dim=1),
            torch.cat((h_chin_jawline[-2,:,:], h_chin_jawline[-1,:,:]), dim=1),
            torch.cat((h_mv[-2,:,:], h_mv[-1,:,:]), dim=1)
        ]
        
        proj_streams = [self.visual_proj(s) for s in h_streams[:10]] + [self.motion_proj(h_streams[10])]
        
        attention_input = torch.stack(proj_streams, dim=1)
        attention_output, _ = self.attention_layer(attention_input, attention_input, attention_input)
        
        output = self.classifier(attention_output.flatten(start_dim=1))
        return output

class DeepfakePredictor:
    def __init__(self, model_path, device=DEVICE):
        self.device = device
        self.model = self.load_model(model_path)
        self.mtcnn = MTCNN(keep_all=False, device=device)
        self.landmarker = self.load_landmarker()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        model = LRATF().to(self.device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            raise

    def load_landmarker(self):
        model_asset_path = 'face_landmarker.task'
        if not os.path.exists(model_asset_path):
            import urllib.request
            print("Downloading MediaPipe face landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(model_url, model_asset_path)
        
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        return vision.FaceLandmarker.create_from_options(options)

    def preprocess_video(self, video_path):
        temp_dir = "temp_prediction_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        frames_dir = os.path.join(temp_dir, 'frames')
        landmarks_dir = os.path.join(temp_dir, 'landmarks')
        mv_path = os.path.join(temp_dir, 'motion_vectors.npy')

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(landmarks_dir, exist_ok=True)

        # 1. Extract frames and landmarks
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 32:
            shutil.rmtree(temp_dir)
            return None, None, None, "Video is too short for analysis."
        
        frame_indices = np.linspace(0, total_frames - 1, min(total_frames, SEQUENCE_LENGTH), dtype=int)
        frames_saved = 0
        first_face_thumbnail = None

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(frame_rgb)

            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                margin = 20
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
                
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0: continue
                
                if first_face_thumbnail is None:
                    first_face_thumbnail = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                detection_result = self.landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    final_frame = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(frames_dir, f'frame_{i}.jpg'), final_frame)
                    
                    landmarks = detection_result.face_landmarks[0]
                    crop_h, crop_w, _ = cropped_face.shape
                    full_h, full_w, _ = frame.shape
                    
                    adjusted_landmarks = [[(lm.x * crop_w + x1) / full_w, (lm.y * crop_h + y1) / full_h, lm.z] for lm in landmarks]
                    np.save(os.path.join(landmarks_dir, f'landmarks_{i}.npy'), np.array(adjusted_landmarks))
                    frames_saved += 1
        
        cap.release()
        if frames_saved == 0:
            shutil.rmtree(temp_dir)
            return None, None, None, "No faces were detected in the video."

        # 2. Extract motion vectors
        success_mv = extract_motion_vectors_with_optical_flow(video_path, mv_path)
        if not success_mv:
            shutil.rmtree(temp_dir)
            return None, None, None, "Failed to extract motion vectors."

        # 3. Load data into tensors
        frames, landmarks_list = [], []
        frame_files = sorted(os.listdir(frames_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        landmark_files = sorted(os.listdir(landmarks_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        all_motion_vectors = np.load(mv_path)

        num_items = min(len(frame_files), len(landmark_files), len(all_motion_vectors))

        for i in range(num_items):
            frame = Image.open(os.path.join(frames_dir, frame_files[i])).convert("RGB")
            frames.append(self.transform(frame))
            landmarks_list.append(np.load(os.path.join(landmarks_dir, landmark_files[i])))
        
        motion_vectors = all_motion_vectors[:num_items]

        padding_needed = SEQUENCE_LENGTH - len(frames)
        if padding_needed > 0:
            frames.extend([torch.zeros_like(frames[0])] * padding_needed)
            landmarks_list.extend([np.zeros_like(landmarks_list[0])] * padding_needed)
            motion_vectors = np.concatenate([motion_vectors, np.zeros((padding_needed, motion_vectors.shape[1], 2), dtype=np.int16)])

        frames_tensor = torch.stack(frames[:SEQUENCE_LENGTH]).unsqueeze(0).to(self.device)
        landmarks_tensor = torch.from_numpy(np.array(landmarks_list[:SEQUENCE_LENGTH])).float().unsqueeze(0).to(self.device)
        motion_vectors_tensor = torch.from_numpy(motion_vectors[:SEQUENCE_LENGTH]).float().unsqueeze(0).to(self.device)

        shutil.rmtree(temp_dir)

        return frames_tensor, landmarks_tensor, motion_vectors_tensor, first_face_thumbnail

    def predict(self, frames_tensor, landmarks_tensor, motion_vectors_tensor):
        with torch.no_grad():
            output = self.model(frames_tensor, landmarks_tensor, motion_vectors_tensor)
            confidence_score = torch.sigmoid(output).item()
        
        prediction = "FAKE" if confidence_score > 0.5 else "REAL"
        return prediction, confidence_score

def extract_motion_vectors_with_optical_flow(video_path, output_path):
    # This is the CPU version for broader compatibility.
    try:
        cap = cv2.VideoCapture(video_path)
        all_frames_mv = []
        max_vectors_per_frame = 512
        ret, prev_frame = cap.read()
        if not ret: return False
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_count = 0
        while frame_count < SEQUENCE_LENGTH:
            ret, curr_frame = cap.read()
            if not ret: break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            h, w, _ = flow.shape
            step = int(np.sqrt((h * w) / max_vectors_per_frame))
            y_coords, x_coords = np.arange(0, h, step), np.arange(0, w, step)
            
            sampled_vectors = [flow[y, x] for y in y_coords for x in x_coords][:max_vectors_per_frame]
            
            if len(sampled_vectors) < max_vectors_per_frame:
                sampled_vectors.extend([[0, 0]] * (max_vectors_per_frame - len(sampled_vectors)))

            all_frames_mv.append(np.array(sampled_vectors, dtype=np.int16))
            prev_gray = curr_gray
            frame_count += 1

        cap.release()
        if not all_frames_mv: return False
        
        np.save(output_path, np.array(all_frames_mv))
        return True
    except Exception as e:
        print(f"Error extracting optical flow: {e}")
        return False