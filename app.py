from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your model
model = load_model("model/resnet50_transfer_learning_model.h5")  # Make sure this relative path is correct

# Preprocessing function
def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img_data = file.read()
        preprocessed_image = prepare_image(img_data)
        preds = model.predict(preprocessed_image)[0]
        predicted_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Local dev mode
if __name__ == '__main__':
    app.run(debug=True)
