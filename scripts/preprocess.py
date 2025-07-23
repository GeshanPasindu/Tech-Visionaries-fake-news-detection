# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Define paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset")
# PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")


# # Ensure processed dataset folder exists
# os.makedirs(PROCESSED_PATH, exist_ok=True)

# # Preprocessing Function
# def preprocess_image(image_path, save_path, size=(380, 380)):
#     """
#     Reads an image, resizes it, converts it to grayscale, remove noises and saves it.
#     """
#     try:
#         # Load image
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"Skipping: {image_path} (Not a valid image)")
#             return
        
#         # Resize
#         img_resized = cv2.resize(img, size)

#         # Convert to grayscale
#         #img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#         #remove noises
#         #image_denoised = cv2.medianBlur(img_gray, 3)

#         # Save processed image
#         cv2.imwrite(save_path, img_resized)

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")

# # Process all images
# def process_dataset():
#     for category in ["Authentic", "Tampered"]:
#         input_folder = os.path.join(DATASET_PATH, category)
#         output_folder = os.path.join(PROCESSED_PATH, category)
#         os.makedirs(output_folder, exist_ok=True)

#         print(f"Processing {category} images...")

#         for image_name in tqdm(os.listdir(input_folder)):
#             image_path = os.path.join(input_folder, image_name)
#             save_path = os.path.join(output_folder, image_name)

#             preprocess_image(image_path, save_path)

# # Run script
# if __name__ == "__main__":
#     process_dataset()
#     print("✅ Preprocessing complete! Processed images saved in 'processed_dataset/'")


import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset")
PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")

# Ensure processed dataset folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Preprocessing Function (Resize & Save for EfficientNet-B4)
def preprocess_image(image_path, save_path, size=(380, 380)):
    """
    Reads an image, resizes it to (380, 380), and saves as RGB.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping: {image_path} (Not a valid image)")
            return
        
        # Resize and ensure RGB (EfficientNet expects 3 channels)
        img_resized = cv2.resize(img, size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # OpenCV loads as BGR
        
        # Save processed image
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))  # Save as BGR for consistency

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process all images (split into train/test later)
def process_dataset():
    for category in ["Authentic", "Tampered"]:
        input_folder = os.path.join(DATASET_PATH, category)
        output_folder = os.path.join(PROCESSED_PATH, category)
        os.makedirs(output_folder, exist_ok=True)

        print(f"Processing {category} images...")
        for image_name in tqdm(os.listdir(input_folder)):
            image_path = os.path.join(input_folder, image_name)
            save_path = os.path.join(output_folder, image_name)
            preprocess_image(image_path, save_path)

if __name__ == "__main__":
    process_dataset()
    print("✅ Preprocessing complete! Images saved in 'processed_dataset/'")
