# import numpy as np
# import tensorflow as tf
# import cv2
# import os
# from data_loader import load_dataset

# # Load trained model
# model = tf.keras.models.load_model("../vgg16_model.h5")

# # Load test data
# X_train, X_test, y_train, y_test = load_dataset()

# # Evaluate model
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {acc:.4f}")

# # Predict on a single image (example)
# def predict_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (224, 224))
#     img = np.expand_dims(img, axis=0) / 255.0  # Normalize

#     prediction = model.predict(img)
#     class_idx = np.argmax(prediction)
#     categories = ["Authentic", "Tampered"]
    
#     return categories[class_idx]

# # Test prediction
# test_img = "../processed_dataset/Authentic/Au_ani_00001.jpg"  # Change this
# print(f"Predicted class: {predict_image(test_img)}")




# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import classification_report, confusion_matrix
# from data_loader import load_dataset

# # Load trained model
# model = tf.keras.models.load_model("../vgg16_model.h5")

# # Load test data
# X_train, X_test, y_train, y_test = load_dataset()

# # Evaluate model
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {acc:.4f}")

# # Get model predictions on test data
# y_pred = model.predict(X_test)

# # Convert predicted probabilities to class labels
# y_pred_labels = np.argmax(y_pred, axis=1)
# y_true_labels = np.argmax(y_test, axis=1)

# # Print classification report
# print("Classification Report:")
# print(classification_report(y_true_labels, y_pred_labels, target_names=["Authentic", "Tampered"]))

# # Print confusion matrix
# print("Confusion Matrix:")
# print(confusion_matrix(y_true_labels, y_pred_labels))



# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split

# # Define class names
# CLASS_NAMES = ["Authentic", "Tampered"]

# # Paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FEATURE_DIR = os.path.join(SCRIPT_DIR, "..", "processed_dataset", "features")
# MODEL_PATH = os.path.join(SCRIPT_DIR, "feature_based_model.h5")

# # Load features and labels
# def load_feature_data(feature_dir):
#     features = []
#     labels = []

#     for idx, class_name in enumerate(CLASS_NAMES):
#         class_path = os.path.join(feature_dir, class_name)
#         feature_file = os.path.join(class_path, "features.npy")
#         label_file = os.path.join(class_path, "labels.npy")

#         if not os.path.exists(feature_file) or not os.path.exists(label_file):
#             print(f"❌ Missing feature files for: {class_name}")
#             continue

#         f = np.load(feature_file)
#         l = np.load(label_file)

#         features.append(f)
#         labels.append(l)

#     X = np.vstack(features)
#     y = np.hstack(labels)

#     return X, y

# # Load data
# print("📦 Loading feature data for testing...")
# X, y = load_feature_data(FEATURE_DIR)

# # One-hot encode labels for model input
# y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)

# # Split into test set (reuse same split as training)
# _, X_test, _, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y, random_state=42)

# # Load trained model
# print("📥 Loading trained model...")
# model = load_model(MODEL_PATH)

# # Predict
# print("🔍 Running predictions on test data...")
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # Evaluation
# print("📊 Classification Report:")
# print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# print("📉 Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

# accuracy = accuracy_score(y_true, y_pred)
# print(f"✅ Test Accuracy: {accuracy:.4f}")


# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# # Paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")
# MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_model.h5")

# # Load test data (20% held out from training)
# test_datagen = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
# )

# test_generator = test_datagen.flow_from_directory(
#     PROCESSED_PATH,
#     target_size=(380, 380),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation',  # Same as validation_split=0.2 in training
#     shuffle=False  # Important for correct labels
# )

# # Load trained model
# model = load_model(MODEL_PATH)

# # Evaluate
# print("📊 Evaluating model on test set...")
# loss, accuracy = model.evaluate(test_generator)
# print(f"✅ Test Accuracy: {accuracy:.4f}")

# # Predictions
# y_pred = (model.predict(test_generator) > 0.5).astype("int32")
# y_true = test_generator.classes

# # Classification Report
# print("\n📋 Classification Report:")
# print(classification_report(y_true, y_pred, target_names=["Authentic", "Tampered"]))

# # Confusion Matrix
# print("📉 Confusion Matrix:")
# print(confusion_matrix(y_true, y_pred))

import os
import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")
MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_model.h5")
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 16
NUM_CLASSES = 2

# Load image paths and labels
class_names = sorted(os.listdir(DATASET_PATH))
image_paths = []
labels = []

for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.isdir(class_dir):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for file in glob(os.path.join(class_dir, ext)):
                image_paths.append(file)
                labels.append(idx)

image_paths = np.array(image_paths)
labels = np.array(labels)

# Preprocessing
def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    return image, tf.one_hot(label, NUM_CLASSES)

def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Build test dataset (reuse full dataset for testing in this example)
test_dataset = build_dataset(image_paths, labels)

# Load model
model = load_model(MODEL_PATH)

# Evaluate
loss, acc = model.evaluate(test_dataset)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
