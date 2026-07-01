import streamlit as st
import sys
import os
import tempfile
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import streamlit.components.v1 as components
import plotly.graph_objects as go

# --- Suppress TensorFlow Logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Path Setup ---
# (Assuming other modules are in folders relative to this app)
# You can adjust these paths if your file structure is different.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", 'Video Analysis', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "..", 'Text Analysis')))

# --- Module Imports (with error handling) ---
try:
    from prediction import DeepfakePredictor, DEVICE
    from predict import predict_news_with_explanation
except ImportError as e:
    st.error(f"Failed to import a necessary module: {e}. Please check your project structure.")
    # To allow the app to run partially, we can stub the missing functions
    def DeepfakePredictor(*args, **kwargs): return None
    def predict_news_with_explanation(*args, **kwargs): return None, [], None, [], 0, 0, 0
    DEVICE = 'cpu'

# ==============================================================================
# IMAGE FORGERY DETECTION LOGIC (INTEGRATED AND FIXED)
# ==============================================================================

@st.cache_resource
def load_image_model(model_path):
    """Loads the Keras model and caches it."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading image forgery model: {e}")
        return None

def preprocess_for_gradcam(image_path, target_size=(380, 380)):
    """Preprocesses an image for model prediction and Grad-CAM visualization."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded. Check the file path and format.")
    
    # Keep original image for visualization, resize it
    img_resized = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Preprocess for the model
    img_array = np.expand_dims(img_rgb, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array, img_rgb

def generate_gradcam_heatmap(model, img_tensor, original_img, last_conv_layer_name="top_conv"):
    """Generates and overlays a Grad-CAM heatmap on the original image."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # Normalize
    heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
    
    # Create colored heatmap and overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

    # Find and draw contours
    heatmap_bin = (heatmap > 0.6).astype(np.uint8) * 255 # Threshold for cleaner contours
    contours, _ = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(superimposed_img, contours, -1, (0, 0, 255), 2) # Draw red contours
    
    return superimposed_img

def make_image_prediction(model, image_path):
    """
    Makes a prediction on an image, generates a heatmap if tampered,
    and returns the results.
    """
    try:
        img_tensor, orig_img_resized = preprocess_for_gradcam(image_path)
        
        # Prediction
        prediction_confidence = model.predict(img_tensor, verbose=0)[0][0]
        label = "Tampered" if prediction_confidence > 0.5 else "Authentic"
        
        heatmap_image = None
        if label == "Tampered":
            # Generate heatmap only if tampered
            heatmap_image = generate_gradcam_heatmap(model, img_tensor, orig_img_resized)

        # Returns: label (str), confidence (float), heatmap image (np.array) or None
        return label, prediction_confidence, heatmap_image

    except Exception as e:
        st.error(f"An error occurred during image prediction: {e}")
        return None, None, None

# ==============================================================================
# STREAMLIT UI COMPONENTS
# ==============================================================================

def show_circular_confidence(confidence, label="DEEPFAKE"):
    percent = int(confidence * 100)
    color_start, color_end = ("#ff4b1f", "#ff9068") if label == "DEEPFAKE" else ("#1f9cff", "#68bbff")
    html_code = f"""
    <div style="text-align: center; margin-top: 10px;">
        <svg width="150" height="150">
            <circle cx="75" cy="75" r="65" stroke="#eee" stroke-width="15" fill="none"/>
            <circle cx="75" cy="75" r="65" stroke="url(#grad1)" stroke-width="15" fill="none"
                    stroke-dasharray="{int(2 * 3.1415 * 65)}"
                    stroke-dashoffset="{int(2 * 3.1415 * 65 * (1 - confidence))}"
                    stroke-linecap="round" transform="rotate(-90 75 75)"/>
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:{color_start};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:{color_end};stop-opacity:1" />
                </linearGradient>
            </defs>
            <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="28px" fill="{color_start}">{percent}%</text>
        </svg>
        <p style="margin-top: -10px;">Confidence Score ({label})</p>
    </div>
    """
    components.html(html_code, height=200)

def show_sentiment_gauge(value, title):
    # Determine color based on value
    if value > 0.3: bar_color = "green"
    elif value < -0.3: bar_color = "red"
    else: bar_color = "lightgray"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(128, 128, 128, 0.2)'},
                {'range': [0.3, 1], 'color': 'rgba(0, 128, 0, 0.2)'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

def show_word_importance_table(features, label):
    st.markdown("##### Word Importance")
    df = pd.DataFrame(features, columns=['Word', 'Weight'])
    st.dataframe(df, use_container_width=True)

# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================

def main():
    st.set_page_config(page_title="🔍 Multi-Modal Fake News Detector", layout="wide")
    st.title("🧠 Multi-Modal Fake News Detection Suite")

    st.sidebar.title("⚙️ Model Paths")
    video_model_path_default = "C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/models/student_lratf/best_model_epoch_146_auc_0.9692.pth"
    image_model_path_default = "C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Image Analysis/scripts/efficientnet_b4_finetuned.h5"

    video_model_path = st.sidebar.text_input("Video Model Path (.pth)", value=video_model_path_default)
    image_model_path = st.sidebar.text_input("Image Model Path (.h5)", value=image_model_path_default)

    # --- Load Models ---
    if os.path.exists(video_model_path) and DeepfakePredictor is not None:
        video_predictor = DeepfakePredictor(model_path=video_model_path, device=DEVICE)
    else:
        video_predictor = None
        st.sidebar.warning("Video model not found or module missing.")

    if os.path.exists(image_model_path):
        image_model = load_image_model(image_model_path)
    else:
        image_model = None
        st.sidebar.warning("Image model not found.")

    tab1, tab2, tab3 = st.tabs(["🎥 Deepfake Video", "📰 Fake Text", "🖼️ Image Forgery"])

    with tab1:
        st.header("🎬 Upload and Analyze a Video")
        uploaded_file = st.file_uploader("Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"], key="video_uploader")

        if uploaded_file and video_predictor:
            st.video(uploaded_file)
            if st.button("🚀 Analyze Video"):
                with st.spinner("Processing video... This may take a moment."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        temp_video_path = tfile.name
                    
                    try:
                        frames, landmarks, motion, face_thumb = video_predictor.preprocess_video(temp_video_path)
                        if frames is not None:
                            pred, conf = video_predictor.predict(frames, landmarks, motion)
                            st.subheader("📊 Result")
                            if pred == "FAKE":
                                st.error("⚠️ This video is likely a **DEEPFAKE**.")
                            else:
                                st.success("✅ This video appears to be **REAL**.")
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if face_thumb: st.image(face_thumb, caption="Detected Face", width=130)
                            with col2:
                                show_circular_confidence(conf, label="DEEPFAKE" if pred == "FAKE" else "REAL")
                        else:
                            st.error(f"Processing failed. Could not detect a face in the video.")
                    finally:
                        os.remove(temp_video_path)

    with tab2:
        st.header("📰 Fake News Text Analysis")
        user_input_text = st.text_area("Enter text here:", height=150)
        if st.button("🔬 Analyze Text"):
            if user_input_text and predict_news_with_explanation is not None:
                with st.spinner("Analyzing text..."):
                    pred, features, para, _, txt_sent, emj_sent, hyb_sent = predict_news_with_explanation(user_input_text)
                    st.subheader("📊 Analysis Result")
                    if pred == "FAKE":
                        st.error("⚠️ This text is likely **FAKE NEWS**.")
                    else:
                        st.success("✅ This text appears to be **REAL**.")
                    
                    st.markdown("##### Justification")
                    st.markdown(f"> {para}")

                    st.markdown("##### Sentiment Breakdown")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a: show_sentiment_gauge(txt_sent, "Text Sentiment")
                    with col_b: show_sentiment_gauge(emj_sent, "Emoji Sentiment")
                    with col_c: show_sentiment_gauge(hyb_sent, "Hybrid Sentiment")
                    
                    if features: show_word_importance_table(features, pred)
            else:
                st.warning("Please enter some text to analyze.")

    with tab3:
        st.header("🖼️ Image Forgery Detection")
        uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")

        if uploaded_file and image_model:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            if st.button("🔬 Analyze Image"):
                with st.spinner("Analyzing image for forgery..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
                        tfile.write(uploaded_file.read())
                        temp_image_path = tfile.name
                    
                    try:
                        label, confidence, heatmap_img = make_image_prediction(image_model, temp_image_path)
                        
                        st.subheader("📊 Analysis Result")
                        if label == "Tampered":
                            st.error(f"⚠️ This image is likely **TAMPERED** (Confidence: {confidence:.2f})")
                            st.info("The heatmap below highlights suspected alterations. Red contours mark the most likely tampered regions.")
                            if heatmap_img is not None:
                                # Convert BGR (from OpenCV) to RGB for Streamlit
                                heatmap_img_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                                st.image(heatmap_img_rgb, caption="Forgery Heatmap (Grad-CAM)", use_container_width=True)
                        elif label == "Authentic":
                            st.success(f"✅ This image appears to be **AUTHENTIC** (Confidence: {1 - confidence:.2f})")
                        else:
                            st.warning("Could not analyze the image.")
                    finally:
                        os.remove(temp_image_path)

if __name__ == "__main__":
    main()