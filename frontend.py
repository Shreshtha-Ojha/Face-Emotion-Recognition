import streamlit as st
import requests
from PIL import Image
import io
import random  # For generating dummy emotions in the mock API

# Styling with CSS
st.markdown("""
    <style>
    /* Background */
    .stApp { background-color: #1e1e2e; color: #ffffff; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #252532; color: white; }
    
    /* Titles */
    .title { text-align: center; font-size: 40px; font-weight: bold; color: #4CAF50; }
    .subheader { text-align: center; font-size: 20px; color: #c7c7c7; }

    /* File Upload & Webcam Section */
    .stFileUploader, .stCamera {
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #2e2e3e;
        color: white;
    }
    
    /* Button Styling */ 
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        margin-top: 10px;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.05); }

    /* Image Styling */
    .stImage img {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 255, 0, 0.3);
        margin-top: 20px;
    }

    /* Detected Emotion Box */
    .emotion-box {
        background-color: #2e2e3e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 8px rgba(0, 255, 0, 0.2);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #252532;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>       
    """,
    unsafe_allow_html=True
)

# Title and Subheader
st.markdown('<div class="title"> Face Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image or use the webcam to analyze emotions.</div>', unsafe_allow_html=True)

# Image Upload Options
option = st.radio("How would you like to provide an image?", ("Upload Image", "Use Webcam"))
selected_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)

elif option == "Use Webcam":
    image_from_camera = st.camera_input("Take a photo")
    if image_from_camera:
        selected_image = Image.open(image_from_camera)

# If an image is selected, display and process it
if selected_image:
    st.image(selected_image, caption="📸 Selected Image", use_container_width=True)

    img_bytes = io.BytesIO()
    selected_image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Mock API Response (temporary, replace with actual API call)
    MOCK_EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Neutral"]
    result = {"faces": [{"id": 1, "emotion": random.choice(MOCK_EMOTIONS)}]}

    # Display the detected emotion
    if result.get("faces"):
        detected_emotion = result["faces"][0]["emotion"]
        st.markdown(f"""
        <div class="emotion-box">
            <h3>🎭 Detected Emotion:</h3>
            <h2 style="color:#4CAF50;"><b>{detected_emotion}</b></h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No face detected!!")

# Footer
st.markdown('<div class="footer">Developed by MoodLens | © 2025 All Rights Reserved</div>', unsafe_allow_html=True)
