import streamlit as st
import sys
import os
import tempfile
from PIL import Image
import streamlit.components.v1 as components

# --- Path Setup ---
video_analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'Video Analysis', 'src'))
if video_analysis_path not in sys.path:
    sys.path.insert(0, video_analysis_path)

try:
    from prediction import DeepfakePredictor, DEVICE
except ImportError as e:
    st.error(f"Error importing modules: {e}. Please ensure the 'Video Analysis/src' folder is in the correct location and contains an '__init__.py' file.")
    st.stop()

@st.cache_resource
def load_predictor(model_path):
    try:
        predictor = DeepfakePredictor(model_path=model_path, device=DEVICE)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Circular Confidence Display ---
def show_circular_confidence(confidence):
    percent = int(confidence * 100)
    html_code = f"""
    <div style="text-align: center; margin-top: 10px;">
        <svg width="150" height="150">
            <circle cx="75" cy="75" r="65" stroke="#eee" stroke-width="15" fill="none"/>
            <circle cx="75" cy="75" r="65" stroke="url(#grad1)" stroke-width="15" fill="none"
                    stroke-dasharray="{int(2 * 3.1415 * 65)}"
                    stroke-dashoffset="{int(2 * 3.1415 * 65 * (1 - confidence))}"
                    stroke-linecap="round"
                    transform="rotate(-90 75 75)"/>
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#ff4b1f;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ff9068;stop-opacity:1" />
                </linearGradient>
            </defs>
            <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="28px" fill="#ff4b1f">{percent}%</text>
        </svg>
        <p style="margin-top: -10px;">Confidence Score (Fake)</p>
    </div>
    """
    components.html(html_code, height=200)

# --- Main App ---
def main():
    st.set_page_config(page_title="🔍 Multi-Modal Fake News Detector", layout="wide")
    st.title("🧠 Multi-Modal Fake News Detection Suite")
    st.markdown("""
        Detect and analyze suspicious content using state-of-the-art AI models across videos, images, and text.
        Use the tabs below to navigate through modalities.
    """)

    st.sidebar.title("⚙️ Configuration")
    model_path_default = "C:/Users/MSI/Documents/Git/Tech-Visionaries-fake-news-detection/Video Analysis/models/student_lratf/best_model_epoch_307_auc_0.9682.pth"
    model_path = st.sidebar.text_input("Model Path (.pth)", value=model_path_default)

    if not os.path.exists(model_path):
        st.sidebar.error("❌ Model file not found.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🎥 Deepfake Video", "🖼️ Image Forgery", "📰 Fake Text"])

    with tab1:
        st.header("🎬 Upload and Analyze a Video")
        uploaded_file = st.file_uploader("📤 Upload a video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

        if uploaded_file:
            st.video(uploaded_file)

            if st.button("🚀 Analyze Video"):
                predictor = load_predictor(model_path)
                if predictor:
                    with st.spinner("🔍 Processing video... This may take a few moments."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                            tfile.write(uploaded_file.read())
                            temp_video_path = tfile.name

                        frames_tensor, landmarks_tensor, motion_vectors_tensor, face_thumbnail = predictor.preprocess_video(temp_video_path)

                        if frames_tensor is not None:
                            prediction, confidence = predictor.predict(frames_tensor, landmarks_tensor, motion_vectors_tensor)

                            st.subheader("📊 Result")
                            if prediction == "FAKE":
                                st.error("⚠️ This video is likely a **DEEPFAKE**.")
                            else:
                                st.success("✅ This video appears to be **REAL**.")

                            if face_thumbnail is not None:
                                st.image(face_thumbnail, caption="Detected Face", width=130)

                            show_circular_confidence(confidence)

                            st.caption("Higher values → higher likelihood of being fake.")
                        else:
                            st.error(f"Processing failed. Reason: {face_thumbnail}")

                        os.remove(temp_video_path)

    with tab2:
        st.header("🖼️ Image Forgery Detection")
        st.info("🚧 Feature under development. Stay tuned!")

    with tab3:
        st.header("📰 Fake News Text Analysis")
        st.info("🚧 Feature under development. Stay tuned!")

if __name__ == "__main__":
    main()
