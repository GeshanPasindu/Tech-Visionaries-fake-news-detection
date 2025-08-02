
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from PIL import Image, ImageChops, ImageEnhance
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tkinter import Tk, filedialog

# # ---------------------- Paths ----------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_forgery_detector.h5")
# TEMP_ELA_PATH = os.path.join(SCRIPT_DIR, "temp_ela.jpg")

# # ---------------------- Load Model ----------------------
# model = load_model(MODEL_PATH)

# # ---------------------- ELA Preprocessing ----------------------
# def perform_ela(image_path, quality=90):
#     original = Image.open(image_path).convert("RGB")
#     temp_jpg = image_path + ".temp.jpg"
#     original.save(temp_jpg, 'JPEG', quality=quality)
#     compressed = Image.open(temp_jpg)

#     diff = ImageChops.difference(original, compressed)
#     extrema = diff.getextrema()
#     max_diff = max([e[1] for e in extrema])
#     max_diff = max_diff if max_diff != 0 else 1
#     diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)

#     diff.save(TEMP_ELA_PATH)
#     os.remove(temp_jpg)
#     return TEMP_ELA_PATH

# # ---------------------- Image Preprocessing ----------------------
# def preprocess_image(image_path, target_size=(380, 380)):
#     ela_path = perform_ela(image_path)
#     img = cv2.imread(ela_path)
#     if img is None:
#         raise ValueError("ELA image could not be loaded.")
#     img = cv2.resize(img, target_size)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_array = np.expand_dims(img_rgb, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array, img_rgb

# # ---------------------- Grad-CAM Generation ----------------------
# def generate_gradcam(model, img_tensor, original_img, save_path="heatmap_output.jpg"):
#     last_conv_layer_name = "block6a_activation"  # You can adjust this if needed

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
#     heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     if original_img.shape[2] == 4:
#         original_img = original_img[:, :, :3]

#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
#     cv2.imwrite(save_path, superimposed_img)
#     print(f"✅ Heatmap saved to {save_path}")

#     # Display heatmap
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.title("Tampered Region Heatmap")
#     plt.show()

# # ---------------------- Prediction ----------------------
# def make_prediction(img_path):
#     print(f"\n🔍 Predicting: {img_path}")
#     try:
#         img_tensor, original_rgb = preprocess_image(img_path)
#         prediction = model.predict(img_tensor, verbose=0)[0][0]
#         label = "Tampered" if prediction > 0.5 else "Authentic"
#         print(f"Prediction: {label} ({prediction:.4f})")

#         if label == "Tampered":
#             generate_gradcam(model, img_tensor, original_rgb)
#         else:
#             print("✅ No tampering detected. No heatmap generated.")

#     except Exception as e:
#         print(f"❌ Error: {e}")

# # ---------------------- File Dialog Interface ----------------------
# if __name__ == "__main__":
#     root = Tk()
#     root.withdraw()
#     selected_file = filedialog.askopenfilename(
#         title="Select an image for prediction",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
#     )

#     if selected_file:
#         make_prediction(selected_file)
#     else:
#         print("No file selected.")



# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from PIL import Image, ImageChops, ImageEnhance
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tkinter import Tk, filedialog

# # ---------------------- Paths ----------------------
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(SCRIPT_DIR, "efficientnet_b4_forgery_detector.h5")
# TEMP_ELA_PATH = os.path.join(SCRIPT_DIR, "temp_ela.jpg")

# # ---------------------- Load Model ----------------------
# model = load_model(MODEL_PATH)

# # ---------------------- ELA Preprocessing ----------------------
# def perform_ela(image_path, quality=90):
#     original = Image.open(image_path).convert("RGB")
#     temp_jpg = image_path + ".temp.jpg"
#     original.save(temp_jpg, 'JPEG', quality=quality)
#     compressed = Image.open(temp_jpg)

#     diff = ImageChops.difference(original, compressed)
#     extrema = diff.getextrema()
#     max_diff = max([e[1] for e in extrema])
#     max_diff = max_diff if max_diff != 0 else 1
#     diff = ImageEnhance.Brightness(diff).enhance(255.0 / max_diff)

#     diff.save(TEMP_ELA_PATH)
#     os.remove(temp_jpg)
#     return TEMP_ELA_PATH

# # ---------------------- Image Preprocessing ----------------------
# def preprocess_image(image_path, target_size=(380, 380)):
#     ela_path = perform_ela(image_path)
#     ela_img = cv2.imread(ela_path)
#     if ela_img is None:
#         raise ValueError("ELA image could not be loaded.")
#     ela_img = cv2.resize(ela_img, target_size)
#     ela_rgb = cv2.cvtColor(ela_img, cv2.COLOR_BGR2RGB)

#     ela_input = np.expand_dims(ela_rgb, axis=0)
#     ela_input = preprocess_input(ela_input)

#     # Also load and resize the original image for visualization
#     original_img = cv2.imread(image_path)
#     original_img = cv2.resize(original_img, target_size)
#     original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

#     return ela_input, original_rgb

# # ---------------------- Grad-CAM Generation ----------------------
# def generate_gradcam(model, img_tensor, original_img, save_path="heatmap_output.jpg"):
#     last_conv_layer_name = "block6a_activation"  # Can adjust based on architecture

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
#     heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Ensure original image is 3-channel
#     if original_img.shape[2] == 4:
#         original_img = original_img[:, :, :3]

#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
#     cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
#     print(f"✅ Heatmap saved to {save_path}")

#     # Display heatmap
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.title("Tampered Region Heatmap")
#     plt.show()

# # ---------------------- Prediction ----------------------
# def make_prediction(img_path):
#     print(f"\n🔍 Predicting: {img_path}")
#     try:
#         img_tensor, original_rgb = preprocess_image(img_path)
#         prediction = model.predict(img_tensor, verbose=0)[0][0]
#         label = "Tampered" if prediction > 0.5 else "Authentic"
#         print(f"Prediction: {label} ({prediction:.4f})")

#         if label == "Tampered":
#             generate_gradcam(model, img_tensor, original_rgb)
#         else:
#             print("✅ No tampering detected. No heatmap generated.")

#     except Exception as e:
#         print(f"❌ Error: {e}")

# # ---------------------- File Dialog Interface ----------------------
# if __name__ == "__main__":
#     root = Tk()
#     root.withdraw()
#     selected_file = filedialog.askopenfilename(
#         title="Select an image for prediction",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
#     )

#     if selected_file:
#         make_prediction(selected_file)
#     else:
#         print("No file selected.")



# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tkinter import Tk, filedialog

# # ---------------------- Load Trained Model ----------------------
# MODEL_PATH = './efficientnet_b4_finetuned.h5'
# model = load_model(MODEL_PATH)

# # ---------------------- Image Preprocessing ----------------------
# def preprocess_image(image_path, target_size=(380, 380)):
#     """
#     Preprocess input image exactly like in training:
#     - Read with cv2
#     - Resize
#     - Convert BGR to RGB
#     - Apply EfficientNet preprocessing
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError("Image could not be loaded.")
#     img = cv2.resize(img, target_size)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_array = np.expand_dims(img_rgb, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array, img_rgb

# # ---------------------- Grad-CAM Generation ----------------------
# def generate_gradcam(model, img_tensor, original_img, save_path="heatmap_output.jpg"):
#     # Use a better layer for localization (e.g., 'block6a_activation')
#     last_conv_layer_name = "block6a_activation"

#     grad_model = tf.keras.models.Model(
#         inputs=model.input,
#         outputs=[model.get_layer(last_conv_layer_name).output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
#     heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Overlay heatmap
#     if original_img.shape[2] == 4:
#         original_img = original_img[:, :, :3]
#     superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
#     cv2.imwrite(save_path, superimposed_img)
#     print(f"Heatmap saved to {save_path}")

#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.title("Tampered Region Heatmap")
#     plt.show()

# # ---------------------- Prediction ----------------------
# def make_prediction(img_path):
#     img_tensor, orig_img = preprocess_image(img_path)
#     prediction = model.predict(img_tensor)[0][0]
#     label = "Tampered" if prediction > 0.5 else "Authentic"
#     print(f"Prediction: {label} ({prediction:.4f})")

#     if label == "Tampered":
#         generate_gradcam(model, img_tensor, orig_img, save_path="heatmap_output.jpg")
#     else:
#         print("No heatmap generated for authentic image.")

# # ---------------------- File Dialog Interface ----------------------
# if __name__ == "__main__":
#     root = Tk()
#     root.withdraw()
#     selected_file = filedialog.askopenfilename(
#         title="Select an image",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
#     )

#     if selected_file:
#         make_prediction(selected_file)
#     else:
#         print("No file selected.")



import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tkinter import Tk, filedialog

# ---------------------- Load Trained Model ----------------------
MODEL_PATH = './efficientnet_b4_finetuned.h5'
model = load_model(MODEL_PATH)

# ---------------------- Image Preprocessing ----------------------
def preprocess_image(image_path, target_size=(380, 380)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
    img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img_rgb

# ---------------------- Grad-CAM and Annotation ----------------------
def generate_gradcam(model, img_tensor, original_img, save_path="heatmap_output.jpg"):
    last_conv_layer_name = "block6a_activation"

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))

    # Threshold the heatmap to identify strong activations
    threshold = 0.5
    heatmap_bin = (heatmap > threshold).astype(np.uint8) * 255

    # Create contours to highlight tampered area
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Generate heatmap overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # Draw contours on heatmap
    cv2.drawContours(superimposed_img, contours, -1, (0, 0, 255), 2)

    # Add label to the image
    cv2.putText(superimposed_img, "Tampered Region", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    # Save and display
    cv2.imwrite(save_path, superimposed_img)
    print(f"Heatmap saved to {save_path}")

    # Show legend and image
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Highlighted Tampered Region (in Red)")
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", label='Detected Tampered Area')
    plt.legend(handles=[red_patch], loc='lower right')
    plt.tight_layout()
    plt.show()

# ---------------------- Prediction ----------------------
def make_prediction(img_path):
    img_tensor, orig_img = preprocess_image(img_path)
    prediction = model.predict(img_tensor)[0][0]
    label = "Tampered" if prediction > 0.5 else "Authentic"
    print(f"Prediction: {label} ({prediction:.4f})")

    if label == "Tampered":
        generate_gradcam(model, img_tensor, orig_img, save_path="heatmap_output.jpg")
    else:
        print("No heatmap generated for authentic image.")

# ---------------------- File Dialog Interface ----------------------
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    selected_file = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if selected_file:
        make_prediction(selected_file)
    else:
        print("No file selected.")
