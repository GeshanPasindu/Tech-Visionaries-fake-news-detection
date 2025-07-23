import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
DATASET_PATH = "../dataset/"
CATEGORIES = ["Authentic", "Tampered"]
IMG_SIZE = 224  # VGG16 expects (224x224)

def load_dataset():
    X_train, X_test, y_train, y_test = [], [], [], []

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        label = CATEGORIES.index(category)  # 0 = Authentic, 1 = Tampered
        
        images = []
        labels = []

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize for VGG16
            images.append(img)
            labels.append(label)
        
        images = np.array(images) / 255.0  # Normalize pixel values
        labels = np.array(labels)

        # Split separately for each category
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(images, labels, test_size=0.2, random_state=42)

        # Append category-wise splits
        X_train.extend(X_train_cat)
        X_test.extend(X_test_cat)
        y_train.extend(y_train_cat)
        y_test.extend(y_test_cat)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=len(CATEGORIES))
    y_test = to_categorical(y_test, num_classes=len(CATEGORIES))

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
