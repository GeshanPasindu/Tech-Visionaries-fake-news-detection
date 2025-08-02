
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

# # One-hot encode labels
# y_categorical = tf.keras.utils.to_categorical(y, num_classes=2)

# # Split into test set
# _, X_test, _, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y, random_state=42)

# # Load trained model
# print("📥 Loading trained model...")
# model = load_model(MODEL_PATH)

# # Predict
# print("🔍 Running predictions on test data...")
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# tn, fp, fn, tp = cm.ravel()
# total = tn + fp + fn + tp

# # Print results
# print("\n📊 Classification Report:")
# print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# print("\n📉 Confusion Matrix:")
# print(cm)

# print("\n🔢 Detailed Percentages:")
# print(f"✅ True Positive (Tampered correctly classified): {tp} ({tp / total:.2%})")
# print(f"❌ False Positive (Authentic misclassified as Tampered): {fp} ({fp / total:.2%})")
# print(f"❌ False Negative (Tampered misclassified as Authentic): {fn} ({fn / total:.2%})")
# print(f"✅ True Negative (Authentic correctly classified): {tn} ({tn / total:.2%})")

# accuracy = accuracy_score(y_true, y_pred)
# print(f"\n🎯 Test Accuracy: {accuracy:.4f}")


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---------------- Constants ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "features")  # ELA preprocessed image path
MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_forgery_detector.h5")
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 16

# ---------------- Load Image Paths and Labels ----------------
label_map = {"Authentic": 0, "Tampered": 1}
reverse_label_map = {v: k for k, v in label_map.items()}
image_paths = []
labels = []

# Gather image file paths for each class
for class_name, label in label_map.items():
    class_dir = os.path.join(DATASET_PATH, class_name)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files = tf.io.gfile.glob(os.path.join(class_dir, ext))
        image_paths.extend(files)
        labels.extend([label] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels)

# ---------------- Split into Train/Test ----------------
_, test_paths, _, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# ---------------- Image Preprocessing ----------------
def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, tf.cast(label, tf.float32)

def build_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ---------------- Load Test Data ----------------
test_dataset = build_dataset(test_paths, test_labels)

# ---------------- Load Trained Model ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- Evaluate ----------------
print("🔍 Running predictions on test set...")
pred_probs = model.predict(test_dataset)
preds = (pred_probs > 0.5).astype(int).flatten()

# ---------------- Accuracy ----------------
accuracy = accuracy_score(test_labels, preds)
print(f"\n✅ Accuracy: {accuracy:.4f}\n")

# ---------------- Classification Report ----------------
print("📊 Classification Report:")
print(classification_report(test_labels, preds, target_names=["Authentic", "Tampered"]))

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(test_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Authentic", "Tampered"],
            yticklabels=["Authentic", "Tampered"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

