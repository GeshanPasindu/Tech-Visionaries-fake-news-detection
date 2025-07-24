import os
import cv2
import torch
import pandas as pd
import numpy as np
import time
import av

# Required for MediaPipe and MTCNN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = 'C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/'
RAW_VIDEO_DIR = os.path.join(BASE_DIR, 'data/raw/DFDC')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed/')
METADATA_PATH = os.path.join(RAW_VIDEO_DIR, 'metadata.json')

# Create output directories
FRAMES_DIR = os.path.join(PROCESSED_DATA_DIR, 'frames')
LANDMARKS_DIR = os.path.join(PROCESSED_DATA_DIR, 'landmarks')
MOTION_VECTORS_DIR = os.path.join(PROCESSED_DATA_DIR, 'motion_vectors')

# Preprocessing settings
TARGET_FRAMES = 128   # How many frames to sample from each video
IMAGE_SIZE = 224      # The final size of the cropped face images

# --- GPU/CUDA Configuration ---
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {DEVICE}')


# def extract_motion_vectors(video_path, output_dir):
#     """Extracts and saves motion vectors from a video file."""
#     video_basename = os.path.splitext(os.path.basename(video_path))[0]
#     output_path = os.path.join(output_dir, f'{video_basename}_mv.npy')

#     if os.path.exists(output_path):
#         return

#     try:
#         container = av.open(video_path, options={"flags2": "+export_mvs"})
#         stream = container.streams.video[0]

#         all_frames_mv = []
#         max_vectors_per_frame = 512

#         for frame in container.decode(stream):
#             if frame.pict_type == av.video.frame.PictureType.I:
#                 all_frames_mv.append(np.zeros((max_vectors_per_frame, 2), dtype=np.int16))
#                 continue

#             motion_vectors = None
#             for side_data in frame.side_data:
#                 if side_data.type.name == "MOTION_VECTORS":
#                     motion_vectors = side_data.to_ndarray()
#                     break

#             if motion_vectors is None or len(motion_vectors) == 0:
#                 all_frames_mv.append(np.zeros((max_vectors_per_frame, 2), dtype=np.int16))
#                 continue

#             vectors = motion_vectors[:, [5, 6]]  # dx, dy
#             padding_needed = max_vectors_per_frame - vectors.shape[0]

#             if padding_needed < 0:
#                 vectors = vectors[:max_vectors_per_frame]
#             else:
#                 padding = np.zeros((padding_needed, 2))
#                 vectors = np.vstack([vectors, padding])

#             all_frames_mv.append(vectors.astype(np.int16))

#         container.close()

#         if not all_frames_mv:
#             raise ValueError("No motion vectors found in any frames.")

#         np.save(output_path, np.array(all_frames_mv))

#     except Exception as e:
#         print(f"Error extracting motion vectors for {video_basename}: {e}")
#         np.save(output_path, np.zeros((1, max_vectors_per_frame, 2), dtype=np.int16))

# def extract_motion_vectors_with_optical_flow(video_path, output_dir):
#     """
#     Calculates dense optical flow using GPU (CUDA) enabled OpenCV.
#     """
#     video_basename = os.path.splitext(os.path.basename(video_path))[0]
#     output_path = os.path.join(output_dir, f'{video_basename}_mv.npy')

#     if os.path.exists(output_path):
#         return

#     try:
#         cap = cv2.VideoCapture(video_path)
#         all_frames_mv = []
#         max_vectors_per_frame = 512

#         ret, prev_frame = cap.read()
#         if not ret:
#             np.save(output_path, np.zeros((TARGET_FRAMES, max_vectors_per_frame, 2), dtype=np.int16))
#             return
            
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        
#         gpu_prev_gray = cv2.cuda_GpuMat()
#         gpu_prev_gray.upload(prev_gray)

        
#         cuda_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)

#         while True:
#             ret, curr_frame = cap.read()
#             if not ret:
#                 break

#             curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
#             gpu_curr_gray = cv2.cuda_GpuMat()
#             gpu_curr_gray.upload(curr_gray)

#             gpu_flow = cuda_flow.calc(gpu_prev_gray, gpu_curr_gray, None)

#             flow = gpu_flow.download()

#             height, width, _ = flow.shape
#             step = int(np.sqrt((height * width) / max_vectors_per_frame))
#             y_coords = np.arange(0, height, step)
#             x_coords = np.arange(0, width, step)
            
#             sampled_vectors = []
#             for y in y_coords:
#                 for x in x_coords:
#                     if len(sampled_vectors) < max_vectors_per_frame:
#                         dx, dy = flow[y, x]
#                         sampled_vectors.append([dx, dy])

#             if len(sampled_vectors) < max_vectors_per_frame:
#                 padding_needed = max_vectors_per_frame - len(sampled_vectors)
#                 sampled_vectors.extend([[0, 0]] * padding_needed)

#             all_frames_mv.append(np.array(sampled_vectors, dtype=np.int16))

#             # <<< 6. Update the previous frame on the GPU >>>
#             gpu_prev_gray = gpu_curr_gray

#         cap.release()
        
#         if len(all_frames_mv) < TARGET_FRAMES:
#              padding_needed = TARGET_FRAMES - len(all_frames_mv)
#              zero_padding_frame = np.zeros((max_vectors_per_frame, 2), dtype=np.int16)
#              all_frames_mv.extend([zero_padding_frame] * padding_needed)

#         final_mv_array = np.array(all_frames_mv[:TARGET_FRAMES])
#         np.save(output_path, final_mv_array)

#     except Exception as e:
#         print(f"Error extracting GPU optical flow for {video_basename}: {e}")
#         np.save(output_path, np.zeros((TARGET_FRAMES, max_vectors_per_frame, 2), dtype=np.int16))

def extract_motion_vectors_with_optical_flow(video_path, output_dir):
    """
    Calculates and saves dense optical flow as motion vectors using the CPU.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f'{video_basename}_mv.npy')

    if os.path.exists(output_path):
        return

    try:
        cap = cv2.VideoCapture(video_path)
        all_frames_mv = []
        max_vectors_per_frame = 512

        ret, prev_frame = cap.read()
        if not ret:
            np.save(output_path, np.zeros((TARGET_FRAMES, max_vectors_per_frame, 2), dtype=np.int16))
            return
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Standard CPU-based optical flow calculation
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            # Sampling logic remains the same
            height, width, _ = flow.shape
            step = int(np.sqrt((height * width) / max_vectors_per_frame))
            y_coords = np.arange(0, height, step)
            x_coords = np.arange(0, width, step)
            
            sampled_vectors = []
            for y in y_coords:
                for x in x_coords:
                    if len(sampled_vectors) < max_vectors_per_frame:
                        dx, dy = flow[y, x]
                        sampled_vectors.append([dx, dy])

            if len(sampled_vectors) < max_vectors_per_frame:
                padding_needed = max_vectors_per_frame - len(sampled_vectors)
                sampled_vectors.extend([[0, 0]] * padding_needed)

            all_frames_mv.append(np.array(sampled_vectors, dtype=np.int16))

            prev_gray = curr_gray

        cap.release()
        
        if len(all_frames_mv) < TARGET_FRAMES:
             padding_needed = TARGET_FRAMES - len(all_frames_mv)
             zero_padding_frame = np.zeros((max_vectors_per_frame, 2), dtype=np.int16)
             all_frames_mv.extend([zero_padding_frame] * padding_needed)

        final_mv_array = np.array(all_frames_mv[:TARGET_FRAMES])
        np.save(output_path, final_mv_array)

    except Exception as e:
        print(f"Error extracting optical flow for {video_basename}: {e}")
        np.save(output_path, np.zeros((TARGET_FRAMES, max_vectors_per_frame, 2), dtype=np.int16))


def process_video_with_mtcnn(video_path, frames_dir, landmarks_dir, mtcnn, landmarker):
    """
    Extracts frames and landmarks using MTCNN for detection and MediaPipe for landmarks.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    video_frames_dir = os.path.join(frames_dir, video_basename)
    video_landmarks_dir = os.path.join(landmarks_dir, video_basename)

    if os.path.exists(video_landmarks_dir) and os.listdir(video_landmarks_dir):
        return

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < TARGET_FRAMES:
            cap.release()
            return

        os.makedirs(video_frames_dir, exist_ok=True)
        os.makedirs(video_landmarks_dir, exist_ok=True)

        frame_indices = np.linspace(0, total_frames - 1, TARGET_FRAMES, dtype=int)
        frames_saved = 0

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Step 1: Detect face with MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)

            if boxes is not None:
                x1, y1, x2, y2 = map(int, boxes[0])
                margin = 20
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
                mtcnn_cropped_face = frame[y1:y2, x1:x2]

                if mtcnn_cropped_face.size == 0:
                    continue

                # Step 2: Get landmarks from the cropped face using MediaPipe
                face_for_landmarks_rgb = cv2.cvtColor(mtcnn_cropped_face, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_for_landmarks_rgb)
                detection_result = landmarker.detect(mp_image)

                if detection_result.face_landmarks:
                    # Save the resized face for the model
                    final_frame = cv2.resize(mtcnn_cropped_face, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(video_frames_dir, f'frame_{i}.jpg'), final_frame)

                    # Step 3: Adjust landmark coordinates to be relative to the FULL frame
                    landmarks = detection_result.face_landmarks[0]
                    crop_h, crop_w, _ = mtcnn_cropped_face.shape
                    full_h, full_w, _ = frame.shape
                    
                    adjusted_landmarks = []
                    for lm in landmarks:
                        abs_x = (lm.x * crop_w) + x1
                        abs_y = (lm.y * crop_h) + y1
                        adjusted_landmarks.append([abs_x / full_w, abs_y / full_h, lm.z])

                    # np.save(os.path.join(landmarks_dir, f'landmarks_{i}.npy'), np.array(adjusted_landmarks))
                    # frames_saved += 1

                    np.save(os.path.join(video_landmarks_dir, f'landmarks_{i}.npy'), np.array(adjusted_landmarks))
                    frames_saved += 1
        
        cap.release()
        if frames_saved == 0:
            os.rmdir(video_frames_dir)
            os.rmdir(video_landmarks_dir)
            
    except Exception as e:
        print(f"Error processing video {video_basename} with MTCNN: {e}")


def main():
    """Main function to orchestrate the entire preprocessing workflow."""
    print("🚀 Starting preprocessing pipeline...")
    start_time = time.time()

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    os.makedirs(MOTION_VECTORS_DIR, exist_ok=True)

    try:
        metadata_df = pd.read_json(METADATA_PATH).T
        metadata_df['video_id'] = metadata_df.index
        metadata_df = metadata_df[metadata_df['split'] == 'train']
    except Exception as e:
        print(f"❌ Error loading metadata.json: {e}")
        return

    # Initialize MTCNN for face detection
    print("Initializing MTCNN Face Detector...")
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    print("✅ MTCNN Initialized.")

    # Initialize MediaPipe Face Landmarker
    print("Initializing MediaPipe Face Landmarker...")
    model_asset_path = 'face_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✅ MediaPipe Initialized.")

    video_files = list(metadata_df.index)
    
    for video_filename in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(RAW_VIDEO_DIR, video_filename)
        
        if not os.path.exists(video_path):
            found_path = None
            for root, _, files in os.walk(RAW_VIDEO_DIR):
                if video_filename in files:
                    found_path = os.path.join(root, video_filename)
                    break
            if not found_path:
                print(f"Warning: Video file not found: {video_filename}")
                continue
            video_path = found_path
        
        # Call the processing functions
        # extract_motion_vectors(video_path, MOTION_VECTORS_DIR)
        extract_motion_vectors_with_optical_flow(video_path, MOTION_VECTORS_DIR)
        process_video_with_mtcnn(video_path, FRAMES_DIR, LANDMARKS_DIR, mtcnn, landmarker)

    end_time = time.time()
    print("\n--- ✅ Preprocessing Complete ---")
    print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    # Download MediaPipe model if it doesn't exist
    face_landmarker_model_path = 'face_landmarker.task'
    if not os.path.exists(face_landmarker_model_path):
        import urllib.request
        print("Downloading MediaPipe face landmarker model...")
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(model_url, face_landmarker_model_path)
    
    main()