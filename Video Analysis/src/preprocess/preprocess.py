import os
import cv2
import torch
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import time
import urllib.request
import av

# --- Configuration ---
model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
model_path = "face_landmarker.task"
# Define paths based on your new folder structure
BASE_DIR = 'C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/' # From src/data/, this goes up to ratf_deepfake_detector/
RAW_VIDEO_DIR = os.path.join(BASE_DIR,'data/raw/DFDC')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
METADATA_PATH = os.path.join(RAW_VIDEO_DIR, 'metadata.json')

# Create output directories
FRAMES_DIR = os.path.join(PROCESSED_DATA_DIR, 'frames')
LANDMARKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'landmarks')
MOTION_VECTORS_DIR = os.path.join(PROCESSED_DATA_DIR, 'motion_vectors')

# Preprocessing settings
TARGET_FRAMES = 128   
IMAGE_SIZE = 224    

# --- Low-Quality Video Simulation Settings ---
COMPRESSION_QUALITY = 60  # JPEG quality (0-100), lower is worse
DOWNSAMPLE_RATIO = 0.5   # Reduce resolution by 50% before upsampling

# --- GPU/CUDA Configuration ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {DEVICE}')


def simulate_low_quality(frame):
    """Applies transformations to a frame to simulate low-quality social media video."""
    # 1. Simulate low resolution
    height, width, _ = frame.shape
    small_dims = (int(width * DOWNSAMPLE_RATIO), int(height * DOWNSAMPLE_RATIO))
    frame_downsampled = cv2.resize(frame, small_dims, interpolation=cv2.INTER_NEAREST)
    frame_upsampled = cv2.resize(frame_downsampled, (width, height), interpolation=cv2.INTER_NEAREST)

    # 2. Simulate compression artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), COMPRESSION_QUALITY]
    result, encimg = cv2.imencode('.jpg', frame_upsampled, encode_param)
    frame_compressed = cv2.imdecode(encimg, 1)

    return frame_compressed


def extract_motion_vectors(video_path, output_dir):
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f'{video_basename}_mv.npy')

    if os.path.exists(output_path):
        print("Output already exists.")
        return

    try:
        # Enable export_mvs via the container's options
        container = av.open(video_path, options={"flags2": "+export_mvs"})
        stream = container.streams.video[0]

        all_frames_mv = []
        max_vectors_per_frame = 512

        for frame in container.decode(stream):
            # if frame.pict_type.name == 'I':
            if str(frame.pict_type) == 'I':
                all_frames_mv.append(np.zeros((max_vectors_per_frame, 2), dtype=np.int16))
                continue

            motion_vectors = None
            for side_data in frame.side_data:
                if side_data.type.name == "MOTION_VECTORS":
                    motion_vectors = side_data.to_ndarray()
                    break

            if motion_vectors is None or len(motion_vectors) == 0:
                all_frames_mv.append(np.zeros((max_vectors_per_frame, 2), dtype=np.int16))
                continue

            vectors = motion_vectors[:, [5, 6]]  # dx, dy

            if vectors.shape[0] > max_vectors_per_frame:
                vectors = vectors[:max_vectors_per_frame]
            else:
                padding = np.zeros((max_vectors_per_frame - vectors.shape[0], 2))
                vectors = np.vstack([vectors, padding])

            all_frames_mv.append(vectors.astype(np.int16))

        container.close()

        if not all_frames_mv:
            raise ValueError("No motion vectors found in any frames.")

        final_mv_array = np.array(all_frames_mv)
        np.save(output_path, final_mv_array)
        # print(f"Saved motion vectors to {output_path}")

    except Exception as e:
        print(f"Error extracting motion vectors for {video_basename}: {e}")
        np.save(output_path, np.zeros((1, 512, 2), dtype=np.int16))  # fallback

# def process_video_frames_and_landmarks(video_path, frames_dir, landmarks_dir, landmarker):
#     """
#     Extracts frames and facial landmarks from a single video, applies low-quality simulation,
#     and saves the results.
#     """
#     video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
#     # Create subdirectories for this specific video's frames and landmarks
#     video_frames_dir = os.path.join(frames_dir, video_basename)
#     video_landmarks_dir = os.path.join(landmarks_dir, video_basename)
    
#     # Check if the last frame exists to see if it's already processed
#     last_frame_path = os.path.join(video_frames_dir, f'frame_{TARGET_FRAMES-1}.jpg')
#     if os.path.exists(last_frame_path):
#         return
        
#     os.makedirs(video_frames_dir, exist_ok=True)
#     os.makedirs(video_landmarks_dir, exist_ok=True)

#     try:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         if total_frames < TARGET_FRAMES:
#             cap.release()
#             return

#         frame_indices = np.linspace(0, total_frames - 1, TARGET_FRAMES, dtype=int)
        
#         for i, frame_idx in enumerate(frame_indices):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if not ret:
#                 continue
            
#             # --- Apply Social Media Simulation ---
#             low_quality_frame = simulate_low_quality(frame)

#             # --- Facial Landmark Detection ---
#             # Convert to MediaPipe's Image format
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(low_quality_frame, cv2.COLOR_BGR2RGB))
#             detection_result = landmarker.detect(mp_image)

#             # --- Cropping and Saving ---
#             if detection_result.face_landmarks:
#                 # Get the first detected face
#                 landmarks = detection_result.face_landmarks[0]
                
#                 # Get bounding box from landmarks to crop the face
#                 x_coords = [landmark.x * low_quality_frame.shape[1] for landmark in landmarks]
#                 y_coords = [landmark.y * low_quality_frame.shape[0] for landmark in landmarks]
#                 x_min, x_max = int(min(x_coords)), int(max(x_coords))
#                 y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
#                 # Add a small margin
#                 margin = 20
#                 x_min = max(0, x_min - margin)
#                 y_min = max(0, y_min - margin)
#                 x_max = min(low_quality_frame.shape[1], x_max + margin)
#                 y_max = min(low_quality_frame.shape[0], y_max + margin)

#                 cropped_face = low_quality_frame[y_min:y_max, x_min:x_max]

#                 if cropped_face.size > 0:
#                     # Resize final cropped face to target size and save
#                     final_frame = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
#                     frame_path = os.path.join(video_frames_dir, f'frame_{i}.jpg')
#                     cv2.imwrite(frame_path, final_frame)

#                     # Save landmarks as a numpy array
#                     landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
#                     landmark_path = os.path.join(video_landmarks_dir, f'landmarks_{i}.npy')
#                     np.save(landmark_path, landmarks_array)
        
#         cap.release()

#     except Exception as e:
#         print(f"Error processing frames/landmarks for {video_basename}: {e}")


def process_video_frames_and_landmarks(video_path, frames_dir, landmarks_dir, landmarker):
    """
    Extracts frames and facial landmarks from a single video, applies low-quality simulation,
    and saves the results.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    video_frames_dir = os.path.join(frames_dir, video_basename)
    video_landmarks_dir = os.path.join(landmarks_dir, video_basename)
    
    # Check if the video has already been processed by looking for the landmarks folder
    if os.path.exists(video_landmarks_dir) and os.listdir(video_landmarks_dir):
        return
        
    os.makedirs(video_frames_dir, exist_ok=True)
    os.makedirs(video_landmarks_dir, exist_ok=True)

    try:
        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            # If ret is False, it means we're at the end of the video
            if not ret:
                break
            

            # --- Facial Landmark Detection ---
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = landmarker.detect(mp_image)
            lowFrame =  simulate_low_quality(frame)

            # --- Cropping and Saving ---
            if detection_result.face_landmarks:
                # Loop through each detected face
                for face_idx, landmarks in enumerate(detection_result.face_landmarks):
                    x_coords = [lm.x * lowFrame.shape[1] for lm in landmarks]
                    y_coords = [lm.y * lowFrame.shape[0] for lm in landmarks]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    margin = 20
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(lowFrame.shape[1], x_max + margin)
                    y_max = min(lowFrame.shape[0], y_max + margin)

                    cropped_face = lowFrame[y_min:y_max, x_min:x_max]

                    if cropped_face.size > 0:
                        final_frame = cv2.resize(cropped_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                        
                        frame_path = os.path.join(video_frames_dir, f'frame_{frame_index}_face_{face_idx}.jpg')
                        cv2.imwrite(frame_path, final_frame)

                        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                        landmark_path = os.path.join(video_landmarks_dir, f'landmarks_{frame_index}_face_{face_idx}.npy')
                        np.save(landmark_path, landmarks_array)
            
            frame_index += 1
            
        cap.release()

    except Exception as e:
        print(f"Error processing frames/landmarks for {video_basename}: {e}")

def main():
    """Main function to orchestrate the entire preprocessing workflow."""
    print("🚀 Starting preprocessing pipeline...")
    start_time = time.time()

    # Create all necessary directories
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    os.makedirs(MOTION_VECTORS_DIR, exist_ok=True)

    # Load DFDC metadata
    try:
        metadata_df = pd.read_json(METADATA_PATH).T
        metadata_df['video_id'] = metadata_df.index
        # Filter for videos that are part of the original training set
        metadata_df = metadata_df[metadata_df['original'].notna()]
    except Exception as e:
        print(f"❌ Error loading metadata.json: {e}")
        return

    # --- Initialize MediaPipe Face Landmarker ---
    print("Initializing MediaPipe Face Landmarker...")
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task') # Download this file
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=4,
                                           min_face_detection_confidence=0.1) 
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✅ MediaPipe Initialized.")


    video_files = list(metadata_df.index)
    
    # for testing the pipeline
    # video_files = video_files[:10] 

    for video_filename in tqdm(video_files, desc="Processing Videos"):
        # The DFDC dataset has subfolders, so we need to find the full path
        video_path = None
        for root, dirs, files in os.walk(RAW_VIDEO_DIR):
            if video_filename in files:
                video_path = os.path.join(root, video_filename)
                break
        
        if not video_path:
            continue

        # --- Task 1: Extract Motion Vectors ---
        extract_motion_vectors(video_path, MOTION_VECTORS_DIR)

        # --- Task 2: Extract Frames and Landmarks ---
        process_video_frames_and_landmarks(video_path, FRAMES_DIR, LANDMARKS_DIR, landmarker)

    end_time = time.time()
    print("\n--- ✅ Preprocessing Complete ---")
    print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Processed data saved in:")
    print(f"  - Frames: {FRAMES_DIR}")
    print(f"  - Landmarks: {LANDMARKS_DIR}")
    print(f"  - Motion Vectors: {MOTION_VECTORS_DIR}")


if __name__ == '__main__':
    # Download the mediapipe model file before running
    if not os.path.exists('face_landmarker.task'):
        
         print("Downloading MediaPipe face landmarker model...")
         urllib.request.urlretrieve(model_url, model_path)
    main()