from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from PIL import Image
from io import BytesIO
import base64
import gdown
import tensorflow as tf
import os
import requests

# ====== Flask API ======
app = Flask(__name__)

# ====== Custom layer if model has one ======
class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ====== Download model from Google Drive using gdown ======
def download_model_if_needed():
    model_path = 'resnet.h5'
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        file_id = '122bsDkj6wzqK6KKSyDq5D7vrJ25KxSH6'
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        print("Model downloaded.")
    else:
        print("Model already exists, no need to download.")

# ====== Load model ======
download_model_if_needed()
model = load_model('resnet.h5', custom_objects={'Cast': Cast})

# ====== Class dictionary and thresholds ======
indices_class = {
    0: 'baoluc',
    1: 'hentai',
    2: 'phanbiet',
    3: 'sex-nude',
    4: 'wound'
}

thresholds = {
    'baoluc': 0.9,
    'hentai': 0.9355,
    'phanbiet': 0.9,
    'sex-nude': 0.93,
    'wound': 0.9
}

def predict_image(input_image):
    print(input_image)
    try:
        print("Received input:", input_image[:100])  # Print first 100 characters
        img = None

        if input_image.startswith('http://') or input_image.startswith('https://'):
            headers = {'User-Agent': 'Mozilla/5.0'}
            print("Downloading image from URL...")
            response = requests.get(input_image, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            print("Image from URL opened successfully.")

        elif input_image.startswith('data:image'):
            print("Processing image from base64 data:image...")
            header, encoded = input_image.split(',', 1)
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            print("Base64 image opened successfully.")

        else:
            print("Processing image from base64 raw string...")
            img_data = base64.b64decode(input_image)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            print("Raw base64 image opened successfully.")

        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("Predicting...")
        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds, axis=1)[0]
        predicted_prob = preds[0][predicted_class_index]
        predicted_class_name = indices_class[predicted_class_index]
        print(f"Prediction result: {predicted_class_name} ({predicted_prob:.4f})")

        threshold = thresholds.get(predicted_class_name, 0.9)
        if predicted_prob < threshold:
            predicted_class = None
        else:
            predicted_class = predicted_class_name

        return {
            'predicted_class': predicted_class,
            'probability': float(predicted_prob)
        }

    except requests.exceptions.RequestException as req_err:
        print(f"Error downloading image from URL: {req_err}")
        raise ValueError("Unable to download image from URL. Make sure the image URL is valid and public.")
    except Exception as e:
        print(f"Error processing image:", e)
        raise ValueError(f"Error processing image: {str(e)}")


@app.route('/', methods=['GET', 'HEAD'])
def home():
    return jsonify({"message": "API server is running. Send POST /predict to get predictions."})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Please provide an image as a URL or base64 string in the "image" field'}), 400

        result = predict_image(data['image'])

        return jsonify({
            'is_approved': result['predicted_class'] is None,
            'confidence': result['probability'],
            'reason': result['predicted_class']
        })
    except Exception as e:
        print("Error in /predict API:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running'})

# ====== Run app ======
if __name__ == '__main__':
    print("Port from environment:", os.environ.get("PORT"))
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
