from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Load your trained model (adjust the path if needed)
model = load_model("model/resnet50_transfer_learning_model.h5")

# Define class labels (adjust to your classes)
class_labels = ['Bardhaman', 'Dol', 'Gajadanta', 'Kaput', 'Karkat', 'Makar1', 'Makar2', 'Samput']

@app.route('/')
def home():
    return "Sattriya Gesture Recognition API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # Read the file content as bytes and load it as an image
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    
    # Optional: reset the file pointer if you need to read the file again later
    file.seek(0)
    
    # Convert image to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img_array = preprocess_input(img_array)
    
    # Predict
    preds = model.predict(img_array)
    preds = preds[0]
    
    max_index = np.argmax(preds)
    confidence = float(preds[max_index])
    predicted_class = class_labels[max_index]
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
