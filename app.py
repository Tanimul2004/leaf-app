import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Ensure upload directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and define class labels (adjust labels to match your training set)
MODEL_PATH = 'leaf_classifier.h5'
model = load_model(MODEL_PATH)
CLASS_NAMES = ['jackfruit', 'mango', 'unknown']  # Ensure this matches your model's target indices

def preprocess_and_predict(img_path):
    # Load and resize image to match MobileNetV2 input size
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_array(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale matching training preprocessing

    # Predict
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return CLASS_NAMES[predicted_class_idx], confidence

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Run model prediction
        predicted_class, confidence = preprocess_and_predict(file_path)

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=confidence,
            image_path=file_path
        )

if __name__ == '__main__':
    app.run(debug=True, port=5000)