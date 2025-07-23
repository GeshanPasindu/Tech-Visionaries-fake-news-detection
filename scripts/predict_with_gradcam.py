# import tensorflow as tf
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os
# from tensorflow.keras.applications import EfficientNetB4
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# # Load trained model
# model = load_model('../efficientnet_b4_model.h5')  

# # Define image size expected by EfficientNetB4
# IMG_SIZE = 380

# def load_and_preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img)
#     img_array = preprocess_input(img_array)
#     return np.expand_dims(img_array, axis=0), img

# def make_prediction(img_path):
#     img_tensor, orig_img = load_and_preprocess_image(img_path)
#     prediction = model.predict(img_tensor)[0][0]
#     label = "Tampered" if prediction > 0.5 else "Authentic"
#     print(f"Prediction: {label} ({prediction:.4f})")

#     if label == "Tampered":
#         generate_gradcam(model, img_tensor, orig_img, save_path="heatmap_output.jpg")
#     else:
#         print("No heatmap generated for authentic image.")

# def generate_gradcam(model, img_tensor, original_img, save_path="heatmap.jpg"):
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer("top_conv").output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_tensor)
#         loss = predictions[:, 0]

#     grads = tape.gradient(loss, conv_outputs)[0]
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     heatmap = cv2.resize(heatmap.numpy(), (original_img.size[0], original_img.size[1]))
#     heatmap = np.uint8(255 * heatmap)

#     heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     original_array = np.array(original_img)
#     if original_array.shape[2] == 4:  # remove alpha channel
#         original_array = original_array[:, :, :3]

#     superimposed_img = cv2.addWeighted(original_array, 0.6, heatmap_colored, 0.4, 0)
#     cv2.imwrite(save_path, superimposed_img)
#     print(f"Heatmap saved to {save_path}")
#     plt.imshow(superimposed_img)
#     plt.axis('off')
#     plt.title("Tampered Region Heatmap")
#     plt.show()

# if __name__ == "__main__":
#     # Example usage:
#     test_image_path = "../dataset/Tampered/sample1.jpg"  # Change this to your test image
#     make_prediction(test_image_path)


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tkinter import Tk, filedialog

# Load trained model
model = load_model('../efficientnet_b4_model.h5')

# Define image size expected by EfficientNetB4
IMG_SIZE = 380

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

def make_prediction(img_path):
    img_tensor, orig_img = load_and_preprocess_image(img_path)
    prediction = model.predict(img_tensor)[0][0]
    label = "Tampered" if prediction > 0.5 else "Authentic"
    print(f"Prediction: {label} ({prediction:.4f})")

    if label == "Tampered":
        generate_gradcam(model, img_tensor, orig_img, save_path="heatmap_output.jpg")
    else:
        print("No heatmap generated for authentic image.")

def generate_gradcam(model, img_tensor, original_img, save_path="heatmap.jpg"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer("top_conv").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_array = np.array(original_img)
    if original_array.shape[2] == 4:
        original_array = original_array[:, :, :3]

    superimposed_img = cv2.addWeighted(original_array, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(save_path, superimposed_img)
    print(f"Heatmap saved to {save_path}")
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Tampered Region Heatmap")
    plt.show()

if __name__ == "__main__":
    # Open a file dialog for the user to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    selected_file = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if selected_file:
        make_prediction(selected_file)
    else:
        print("No file selected.")
