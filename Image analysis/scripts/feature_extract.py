# import os
# import cv2
# import numpy as np
# import pywt
# from skimage.feature import local_binary_pattern

# # Paths
# INPUT_DIR = "C:/ImageTamperingDetection/processed_dataset/"
# FEATURE_DIR = "C:/ImageTamperingDetection/processed_dataset/features/"

# # Ensure feature directory exists
# os.makedirs(FEATURE_DIR, exist_ok=True)

# # Function to extract feature vector from RGB image
# def extract_feature_vector_rgb(img_rgb):
#     feature_vector = []

#     for i in range(3):  # Loop over R, G, B channels
#         channel = img_rgb[:, :, i]

#         # Local Binary Patterns 
#         radius = 1
#         n_points = 8 * radius
#         lbp = local_binary_pattern(channel, n_points, radius, method="uniform")
#         lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
#         lbp_hist = lbp_hist.astype("float")
#         lbp_hist /= (lbp_hist.sum() + 1e-6)
#         feature_vector.extend(lbp_hist)

#         # Discrete Wavelet Transform
#         coeffs2 = pywt.dwt2(channel, 'haar')
#         LL, (LH, HL, HH) = coeffs2
#         feature_vector.extend([HH.mean(), HH.std(), np.max(HH), np.min(HH)])

#         # Canny edge count
#         edges = cv2.Canny(channel, 100, 200)
#         edge_count = np.sum(edges > 0)
#         feature_vector.append(edge_count)

#         # Histogram 
#         # hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
#         # hist = hist.flatten()
#         # hist /= (hist.sum() + 1e-6)
#         # feature_vector.extend(hist.tolist())

#     return np.array(feature_vector)

# # Function to process a category folder
# def extract_features(category):
#     input_folder = os.path.join(INPUT_DIR, category)
#     output_folder = os.path.join(FEATURE_DIR, category)
#     os.makedirs(output_folder, exist_ok=True)

#     feature_data = []
#     labels = []

#     for img_name in os.listdir(input_folder):
#         img_path = os.path.join(input_folder, img_name)

#         # Read RGB image
#         img_rgb = cv2.imread(img_path)

#         if img_rgb is None:
#             print(f"❌ Skipped corrupted or invalid image: {img_name}")
#             continue

#         # Extract features
#         features = extract_feature_vector_rgb(img_rgb)
#         feature_data.append(features)

#         # Set label: 0 for Authentic, 1 for Tampered
#         labels.append(0 if category.lower() == "authentic" else 1)

#     # Save features and labels
#     np.save(os.path.join(output_folder, "features.npy"), np.array(feature_data))
#     np.save(os.path.join(output_folder, "labels.npy"), np.array(labels))

#     print(f"✅ Feature extraction completed for {category} images")

# # Run for both categories
# extract_features("Authentic")
# extract_features("Tampered")


import os
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")
FEATURE_PATH = os.path.join(SCRIPT_DIR, "..", "features")

# Ensure features folder exists
os.makedirs(FEATURE_PATH, exist_ok=True)

def perform_ela(image_path, save_path, quality=90):
    """
    Performs Error Level Analysis (ELA) on an image and saves the output.
    """
    try:
        original = Image.open(image_path).convert("RGB")
        
        # Save as JPEG with defined quality
        temp_path = image_path + ".temp.jpg"
        original.save(temp_path, 'JPEG', quality=quality)
        compressed = Image.open(temp_path)

        # Calculate the difference
        diff = ImageChops.difference(original, compressed)

        # Enhance the difference
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        diff = ImageEnhance.Brightness(diff).enhance(scale)
        diff.save(save_path)

        # Cleanup
        os.remove(temp_path)
    except Exception as e:
        print(f"❌ ELA failed for {image_path}: {e}")

def extract_features():
    for category in ["Authentic", "Tampered"]:
        input_folder = os.path.join(PROCESSED_PATH, category)
        output_folder = os.path.join(FEATURE_PATH, category)
        os.makedirs(output_folder, exist_ok=True)

        print(f"🔍 Generating ELA features for {category}...")
        for image_name in tqdm(os.listdir(input_folder)):
            input_image_path = os.path.join(input_folder, image_name)
            output_feature_path = os.path.join(output_folder, image_name)
            perform_ela(input_image_path, output_feature_path)

if __name__ == "__main__":
    extract_features()
    print("✅ ELA feature extraction complete! Features saved in 'features/'")
