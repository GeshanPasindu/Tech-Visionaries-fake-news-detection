import os
import cv2
import numpy as np
import pywt
from skimage.feature import local_binary_pattern

# Paths
INPUT_DIR = "C:/ImageTamperingDetection/processed_dataset/"
FEATURE_DIR = "C:/ImageTamperingDetection/processed_dataset/features/"

# Ensure feature directory exists
os.makedirs(FEATURE_DIR, exist_ok=True)

# Function to extract feature vector from RGB image
def extract_feature_vector_rgb(img_rgb):
    feature_vector = []

    for i in range(3):  # Loop over R, G, B channels
        channel = img_rgb[:, :, i]

        # LBP
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(channel, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        feature_vector.extend(lbp_hist)

        # DWT (HH component statistics)
        coeffs2 = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs2
        feature_vector.extend([HH.mean(), HH.std(), np.max(HH), np.min(HH)])

        # Canny edge count
        edges = cv2.Canny(channel, 100, 200)
        edge_count = np.sum(edges > 0)
        feature_vector.append(edge_count)

        # Histogram (grayscale-style on this channel)
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist /= (hist.sum() + 1e-6)
        feature_vector.extend(hist.tolist())

    return np.array(feature_vector)

# Function to process a category folder
def extract_features(category):
    input_folder = os.path.join(INPUT_DIR, category)
    output_folder = os.path.join(FEATURE_DIR, category)
    os.makedirs(output_folder, exist_ok=True)

    feature_data = []
    labels = []

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        # Read RGB image
        img_rgb = cv2.imread(img_path)

        if img_rgb is None:
            print(f"❌ Skipped corrupted or invalid image: {img_name}")
            continue

        # Extract features
        features = extract_feature_vector_rgb(img_rgb)
        feature_data.append(features)

        # Set label: 0 for Authentic, 1 for Tampered
        labels.append(0 if category.lower() == "authentic" else 1)

    # Save features and labels
    np.save(os.path.join(output_folder, "features.npy"), np.array(feature_data))
    np.save(os.path.join(output_folder, "labels.npy"), np.array(labels))

    print(f"✅ Feature extraction completed for {category} images")

# Run for both categories
extract_features("Authentic")
extract_features("Tampered")
