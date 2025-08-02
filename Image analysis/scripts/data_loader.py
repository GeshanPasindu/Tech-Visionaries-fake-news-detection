# import os
# import numpy as np
# import cv2
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical

# # Define paths
# DATASET_PATH = "../dataset/"
# CATEGORIES = ["Authentic", "Tampered"]
# IMG_SIZE = 224  # VGG16 expects (224x224)

# def load_dataset():
#     X_train, X_test, y_train, y_test = [], [], [], []

#     for category in CATEGORIES:
#         path = os.path.join(DATASET_PATH, category)
#         label = CATEGORIES.index(category)  # 0 = Authentic, 1 = Tampered
        
#         images = []
#         labels = []

#         for img_name in os.listdir(path):
#             img_path = os.path.join(path, img_name)
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize for VGG16
#             images.append(img)
#             labels.append(label)
        
#         images = np.array(images) / 255.0  # Normalize pixel values
#         labels = np.array(labels)

#         # Split separately for each category
#         X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(images, labels, test_size=0.2, random_state=42)

#         # Append category-wise splits
#         X_train.extend(X_train_cat)
#         X_test.extend(X_test_cat)
#         y_train.extend(y_train_cat)
#         y_test.extend(y_test_cat)

#     # Convert lists to numpy arrays
#     X_train = np.array(X_train)
#     X_test = np.array(X_test)
#     y_train = np.array(y_train)
#     y_test = np.array(y_test)

#     # One-hot encode labels
#     y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
#     y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

#     return X_train, X_test, y_train, y_test

# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = load_dataset()
#     print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")



import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # Only needed for create_data_generators

# --- Configuration for Data Loading ---
PROCESSED_DATA_ROOT = '../processed_dataset'
IMAGE_SIZE = (224, 224) # Must match the size used in preprocess.py
TEST_SPLIT_RATIO = 0.15
VALIDATION_SPLIT_RATIO = 0.15 / (1 - TEST_SPLIT_RATIO) # Split from remaining train_val set
RANDOM_SEED = 42

def load_ela_dataset():
    """
    Loads ELA images and their labels from the processed_dataset directory.
    Includes robust checks for image dimensions.
    """
    X = [] # To store image data
    y = [] # To store labels

    expected_shape = IMAGE_SIZE + (3,) # Expected shape: (height, width, 3 channels)
    problematic_files = [] # To log files that caused issues

    # Load Authentic ELA images
    au_dir = os.path.join(PROCESSED_DATA_ROOT, 'Authentic')
    print(f"Loading Authentic ELA images from: {au_dir}")
    for img_name in os.listdir(au_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(au_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                if img_array.shape == expected_shape:
                    X.append(img_array)
                    y.append(0) # Label 0 for Authentic
                else:
                    problematic_files.append(f"Authentic/{img_name} (Incorrect shape: {img_array.shape})")
            except Exception as e:
                problematic_files.append(f"Authentic/{img_name} (Error loading: {e})")

    # Load Tampered ELA images
    tp_dir = os.path.join(PROCESSED_DATA_ROOT, 'Tampered')
    print(f"Loading Tampered ELA images from: {tp_dir}")
    for img_name in os.listdir(tp_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(tp_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                if img_array.shape == expected_shape:
                    X.append(img_array)
                    y.append(1) # Label 1 for Tampered
                else:
                    problematic_files.append(f"Tampered/{img_name} (Incorrect shape: {img_array.shape})")
            except Exception as e:
                problematic_files.append(f"Tampered/{img_name} (Error loading: {e})")

    # Convert lists to numpy arrays ONLY if X is not empty
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        # Normalize pixel values to [0, 1]
        X = X / 255.0
    else:
        print("Warning: No valid images were loaded into the dataset. X is empty.")
        X = np.array([])
        y = np.array([])


    print(f"Loaded {len(X)} ELA images. Authentic: {np.sum(y == 0) if len(y) > 0 else 0}, Tampered: {np.sum(y == 1) if len(y) > 0 else 0}")

    if problematic_files:
        print("\n--- Problematic Files Encountered (Skipped) ---")
        for f in problematic_files:
            print(f)
        print(f"Total problematic files: {len(problematic_files)}")
        print("Consider inspecting these files in your 'processed_dataset' directory.")

    return X, y

def split_dataset(X, y):
    """
    Splits the dataset into training, validation, and test sets.
    """
    if len(X) == 0:
        print("Cannot split empty dataset.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT_RATIO,
        stratify=y_train_val, random_state=RANDOM_SEED
    )
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# Removed create_data_generators from here to keep data_loader focused on loading and splitting.
# It can be used directly in train_model.py

# --- Main execution for testing ---
if __name__ == "__main__":
    print("Running data_loader.py as main for testing...")
    X, y = load_ela_dataset()
    if len(X) > 0:
        X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    else:
        print("No data to split.")