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

# ====== Custom layer nếu model có ======
class Cast(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ====== Hàm tải model từ Google Drive bằng gdown ======
def download_model_if_needed():
    model_path = 'resnet.h5'
    if not os.path.exists(model_path):
        print("🔽 Đang tải model từ Google Drive...")
        file_id = '122bsDkj6wzqK6KKSyDq5D7vrJ25KxSH6'
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        print("✅ Tải xong model.")
    else:
        print("✅ Model đã tồn tại, không cần tải lại.")

# ====== Tải và load model ======
download_model_if_needed()
model = load_model('resnet.h5', custom_objects={'Cast': Cast})

# ====== Từ điển nhãn và ngưỡng ======
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

# ====== Hàm xử lý ảnh ======
def predict_image(input_image):
    try:
        if input_image.startswith('http://') or input_image.startswith('https://'):
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(input_image, headers=headers)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        elif input_image.startswith('data:image'):
            header, encoded = input_image.split(',', 1)
            img_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(img_data)).convert('RGB')
        else:
            img_data = base64.b64decode(input_image)
            img = Image.open(BytesIO(img_data)).convert('RGB')

        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class_index = np.argmax(preds, axis=1)[0]
        predicted_prob = preds[0][predicted_class_index]
        predicted_class_name = indices_class[predicted_class_index]

        threshold = thresholds.get(predicted_class_name, 0.9)
        if predicted_prob < threshold:
            predicted_class = None
        else:
            predicted_class = predicted_class_name

        return {
            'predicted_class': predicted_class,
            'probability': float(predicted_prob)
        }

    except Exception as e:
        raise ValueError(f"Lỗi xử lý ảnh: {str(e)}")

# ====== Flask API ======
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Vui lòng gửi ảnh dạng URL hoặc base64 trong trường "image"'}), 400
    try:
        result = predict_image(data['image'])
        return jsonify({
            'is_approved': result['predicted_class'] is not None,
            'confidence': result['probability'],
            'reason': result['predicted_class']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Server is running 🚀'})

# ====== Run app ======
if __name__ == '__main__':
    app.run(debug=True)
