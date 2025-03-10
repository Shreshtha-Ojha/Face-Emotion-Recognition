from flask import Flask, request, jsonify
import random

app = Flask(__name__)

#mock api fot testing
MOCK_EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Neutral"]

@app.route('/predict', methods=['POST'])
def predict():
    # Checking for image
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    #mock response
    detected_emotion = random.choice(MOCK_EMOTIONS)
    response = {
        "faces": [
            {"id": 1, "emotion": detected_emotion}
        ]
    }
    return jsonify({"emotion": detected_emotion})

if __name__ == '__main__':
    app.run(debug=True)
