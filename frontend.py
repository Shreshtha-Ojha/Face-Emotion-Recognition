import streamlit as st
from PIL import Image
import io
from Backend.Main import predict_emotion_from_image
import numpy as np
import cv2
import time

# Styling with CSS
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
       background-color: #1e1e2e;
       color: #ffffff;
    }
    [data-testid="stSidebar"] { background-color: #252532; color: white; }
    .title { text-align: center; font-size: 40px; font-weight: bold; color: #4CAF50; }
    .subheader { text-align: center; font-size: 20px; color: #c7c7c7; }
    [data-testid="stFileUploader"], [data-testid="stCameraInput"]{
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        background-color: #2e2e3e;
        color: white;
    }
            
    [data-testid="stRadio"] label {
        color: white !important;  
        font-size: 18px;  
        //opacity: 1 !important; 
        font-weight: bold;  
        filter: brightness(2.5); 
    }        

    button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        margin-top: 10px !important;
        transition: 0.3s !important;
        border: none !important;
        opacity: 1 !important;
        display: block !important;
    }
    button:hover { 
        background-color: #45a049 !important; 
        transform: scale(1.05) !important; 
    }
    .emotion-box {
        background-color: #2e2e3e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 4px 8px rgba(0, 255, 0, 0.2);
    }
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
    """, unsafe_allow_html=True)

st.markdown('<div class="title"> Face Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image, use the webcam, or enable real-time detection.</div>', unsafe_allow_html=True)

option = st.radio("How would you like to provide an image?", ("Upload Image", "Use Webcam", "Real-Time Webcam"))
selected_image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)

elif option == "Use Webcam":
    image_from_camera = st.camera_input("Take a photo")
    if image_from_camera:
        selected_image = Image.open(image_from_camera)

elif option == "Real-Time Webcam":
    stframe = st.empty()
    run = st.checkbox("Start Real-Time Detection")
    if run:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(roi_gray, (48, 48)).astype('float32') / 255.0
                face_resized = np.expand_dims(face_resized, axis=-1)
                face_resized = np.expand_dims(face_resized, axis=0)

                emotion = predict_emotion_from_image(frame)
                if emotion:
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")

        cap.release()



if selected_image is not None:
    st.image(selected_image, caption="📸 Selected Image", use_container_width=True)
    selected_image = selected_image.convert('RGB')
    opencv_image = np.array(selected_image)
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)

    # Ensure the grayscale image is in the correct format
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)

    detected_emotion = predict_emotion_from_image(gray_image)

    if detected_emotion:
        st.markdown(f"""
        <div class="emotion-box">
            <h3>🎭 Detected Emotion:</h3>
            <h2 style="color:#4CAF50;"><b>{detected_emotion}</b></h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("No face detected!")
else:
    if option != "Real-Time Webcam":
        st.warning("Please upload or capture an image first.")

st.markdown('<div class="footer">Developed by MoodLens | © 2025 All Rights Reserved</div>', unsafe_allow_html=True)
