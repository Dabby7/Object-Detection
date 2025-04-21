from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
from PIL import Image
import cv2
import os
import joblib  # <-- Use joblib instead of keras

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the joblib model
model = joblib.load("svm_model.pkl")  # Make sure your model is compatible

# CIFAR-10 class labels (adjust based on your actual model classes)
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Webcam capture
camera = cv2.VideoCapture(0)

# Preprocess image for joblib model (adjust as per your model’s requirements)
def preprocess_image(image_path):
    img = Image.open(image_path).resize((40, 32)).convert('L')  # match training input
    img = np.array(img).astype('float32') / 255.0
    img = img.flatten()
    return [img]


# Generate webcam frames
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ret, frame = camera.read()
        if ret:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
            cv2.imwrite(image_path, frame)

            img = preprocess_image(image_path)
            prediction = model.predict(img)[0]

            # Handle models with `predict_proba`
            if hasattr(model, "predict_proba"):
                confidence = np.max(model.predict_proba(img)) * 100
            else:
                confidence = 100.0  # Assume full confidence if not available

            predicted_class = class_labels[prediction]

            return render_template('index.html', image=image_path, label=predicted_class, confidence=confidence)

    return render_template('index.html', image=None)

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run app
if __name__ == '__main__':
    app.run(debug=True)
