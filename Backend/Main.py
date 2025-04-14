import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained model directly
model_path = os.path.join(os.path.dirname(__file__), 'model_final.h5')
model = load_model(model_path)  # Only load the model once

# Emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

def start_emotion_detector():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    print("Press 'q' to exit the webcam feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ensure the frame is in the correct format (grayscale)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame  # Already grayscale

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

def predict_emotion_from_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(roi_gray, (48, 48)).astype('float32') / 255.0
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)

        prediction = model.predict(face_resized, verbose=0)
        emotion_index = int(np.argmax(prediction))
        return emotion_dict[emotion_index]

    return None



if __name__ == "__main__":
    start_emotion_detector()
