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

CONFIDENCE_THRESHOLD = 0.75  # 75% minimum required confidence

def preprocess_and_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    max_confidence = float(np.max(predictions[0]))
    predicted_class_idx = np.argmax(predictions[0])
    
    # Check if confidence meets the threshold
    if max_confidence < CONFIDENCE_THRESHOLD:
        return "Unknown / Invalid Image", round(100 * max_confidence, 2)

    return CLASS_NAMES[predicted_class_idx], round(100 * max_confidence, 2)

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