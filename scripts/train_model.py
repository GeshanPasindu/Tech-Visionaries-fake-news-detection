# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.optimizers import Adam
# from data_loader import load_dataset  

# # Load dataset
# X_train, X_test, y_train, y_test = load_dataset()

# # Load VGG16 (pretrained model without top layers)
# base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# # Freeze the base layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom layers
# x = Flatten()(base_model.output)
# x = Dense(512, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(128, activation="relu")(x)
# x = Dropout(0.5)(x)

# # Output layer (using softmax for binary classification)
# output = Dense(2, activation="softmax")(x)  # For binary classification (Authentic vs Tampered)

# # Compile model
# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# # Train model
# model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# # Save trained model
# model.save("../vgg16_model.h5")
# print("Model training complete. Model saved as vgg16_model.h5")


# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# # Set path to processed feature directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# FEATURE_DIR = os.path.join(SCRIPT_DIR, "..", "processed_dataset", "features")

# # Class names
# CLASS_NAMES = ["Authentic", "Tampered"]

# # Load pre-extracted features and labels
# def load_feature_data(feature_dir):
#     features = []
#     labels = []

#     for class_name in CLASS_NAMES:
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

#     # One-hot encode labels
#     y = tf.keras.utils.to_categorical(y, num_classes=2)
#     return X, y

# print("🔄 Loading feature vectors...")
# X, y = load_feature_data(FEATURE_DIR)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# print(f"✅ Loaded features. Train: {len(X_train)}, Test: {len(X_test)}")
# print(f"🧬 Feature dimension: {X.shape[1]}")

# # Build a simple fully connected model
# model = Sequential([
#     Dense(256, activation='relu', input_shape=(X.shape[1],)),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(2, activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# print("🚀 Starting training...")
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# # Save the trained model
# model_path = os.path.join(SCRIPT_DIR, "feature_based_model.h5")
# model.save(model_path)
# print(f"✅ Model saved to {model_path}")

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.applications import EfficientNetB4
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split

# # Paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROCESSED_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")

# # Data Generator with EfficientNet Preprocessing
# train_datagen = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
#     validation_split=0.2,  # 80% train, 20% validation
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Load data from directory
# train_generator = train_datagen.flow_from_directory(
#     PROCESSED_PATH,
#     target_size=(380, 380),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     PROCESSED_PATH,
#     target_size=(380, 380),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )

# # Load EfficientNet-B4 (pretrained on ImageNet)
# base_model = EfficientNetB4(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(380, 380, 3)
# )

# # Freeze base model (optional: fine-tune later)
# base_model.trainable = False

# # Build model
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Binary classification
# ])

# # Compile
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Train
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10,
#     verbose=1
# )

# # Save model
# model_path = os.path.join(SCRIPT_DIR, "efficientnet_b4_model.h5")
# model.save(model_path)
# print(f"✅ Model saved to {model_path}")



# import os
# import numpy as np
# import tensorflow as tf
# from glob import glob
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # Constants
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")
# IMAGE_SIZE = (380, 380)
# BATCH_SIZE = 16
# EPOCHS = 10
# NUM_CLASSES = 2
# MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_model.h5")

# # Load image paths and labels
# class_names = sorted(os.listdir(DATASET_PATH))
# image_paths = []
# labels = []

# for idx, class_name in enumerate(class_names):
#     class_dir = os.path.join(DATASET_PATH, class_name)
#     if os.path.isdir(class_dir):
#         for ext in ("*.jpg", "*.jpeg", "*.png"):
#             for file in glob(os.path.join(class_dir, ext)):
#                 image_paths.append(file)
#                 labels.append(idx)

# image_paths = np.array(image_paths)
# labels = np.array(labels)

# # Split dataset
# train_paths, val_paths, train_labels, val_labels = train_test_split(
#     image_paths, labels, test_size=0.2, stratify=labels, random_state=42
# )


# def parse_image(filename, label):
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, IMAGE_SIZE)
#     image = preprocess_input(image)
#     return image, tf.one_hot(label, NUM_CLASSES)

# def build_dataset(paths, labels, is_train=True):
#     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
#     ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
#     if is_train:
#         ds = ds.shuffle(buffer_size=1000)
#     ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#     return ds

# train_dataset = build_dataset(train_paths, train_labels, is_train=True)
# val_dataset = build_dataset(val_paths, val_labels, is_train=False)

# # Model
# base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(380, 380, 3))
# base_model.trainable = False  

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# output = Dense(NUM_CLASSES, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# # Callbacks
# callbacks = [
#     ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
#     EarlyStopping(patience=5, restore_best_weights=True)
# ]

# # Train
# model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)




# import os
# import numpy as np
# import tensorflow as tf
# from glob import glob
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # Constants
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "processed_dataset")
# IMAGE_SIZE = (380, 380)
# BATCH_SIZE = 16
# EPOCHS = 10
# MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_finetuned.h5")

# # Explicit label mapping
# label_map = {"Authentic": 0, "Tampered": 1}
# image_paths = []
# labels = []

# # Load image paths and labels
# for class_name, label in label_map.items():
#     class_dir = os.path.join(DATASET_PATH, class_name)
#     for ext in ("*.jpg", "*.jpeg", "*.png"):
#         for file in glob(os.path.join(class_dir, ext)):
#             image_paths.append(file)
#             labels.append(label)

# image_paths = np.array(image_paths)
# labels = np.array(labels)

# # Split dataset
# train_paths, val_paths, train_labels, val_labels = train_test_split(
#     image_paths, labels, test_size=0.2, stratify=labels, random_state=42
# )

# # Preprocessing function
# def parse_image(filename, label):
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, IMAGE_SIZE)
#     image = preprocess_input(image)
#     return image, tf.cast(label, tf.float32)

# def build_dataset(paths, labels, is_train=True):
#     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
#     ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
#     if is_train:
#         ds = ds.shuffle(buffer_size=1000)
#     ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#     return ds

# train_dataset = build_dataset(train_paths, train_labels, is_train=True)
# val_dataset = build_dataset(val_paths, val_labels, is_train=False)

# # Build model
# base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(380, 380, 3))
# base_model.trainable = False  # Freeze initially

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# output = Dense(1, activation='sigmoid')(x)  # Binary output

# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# # Callbacks
# callbacks = [
#     ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
#     EarlyStopping(patience=5, restore_best_weights=True)
# ]

# # Train initial frozen model
# model.fit(train_dataset, validation_data=val_dataset, epochs=5, callbacks=callbacks)

# # 🔄 Fine-tune top 20 layers
# base_model.trainable = True
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# # Continue training
# model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)

# print("✅ Model training and fine-tuning complete.")


import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "features")  # ← Use ELA features
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_forgery_detector.h5")

# Explicit label mapping
label_map = {"Authentic": 0, "Tampered": 1}
image_paths = []
labels = []

# Load image paths and labels (include TIFF formats)
for class_name, label in label_map.items():
    class_dir = os.path.join(DATASET_PATH, class_name)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        files = glob(os.path.join(class_dir, ext))
        image_paths.extend(files)
        labels.extend([label] * len(files))

image_paths = np.array(image_paths)
labels = np.array(labels)

# Split dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Preprocessing function (supports all image types)
def parse_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)  # format-agnostic
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    return image, tf.cast(label, tf.float32)

def build_dataset(paths, labels, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Build training and validation datasets
train_dataset = build_dataset(train_paths, train_labels, is_train=True)
val_dataset = build_dataset(val_paths, val_labels, is_train=False)

# Build model
base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(380, 380, 3))
base_model.trainable = False  # Freeze initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", mode="max"),
    EarlyStopping(patience=5, restore_best_weights=True)
]

# Step 1: Train with frozen base
model.fit(train_dataset, validation_data=val_dataset, epochs=5, callbacks=callbacks)

# Step 2: Fine-tune top 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)

print("✅ Model training and fine-tuning complete with .jpg and .tif ELA features.")


