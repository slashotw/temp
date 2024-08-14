from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import io
import traceback

app = Flask(__name__)

# 加載預訓練的 MobileNetV2 模型
model = MobileNetV2(weights='imagenet')

# 擴展貓科動物列表
cat_breeds = [
    'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat'
]



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def predict_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data)).resize((224, 224))
        x = np.expand_dims(np.array(img), axis=0)
        x = preprocess_input(x)
        with tf.device('/CPU:0'):  # 強制使用 CPU，避免 GPU 相關錯誤
            preds = model.predict(x)
        return decode_predictions(preds, top=5)[0]
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            image_data = file.read()
            predictions = predict_image(image_data)
            if predictions is None:
                return jsonify({'error': 'Error processing image'}), 500

            is_cat = any(breed in pred[1].lower() for pred in predictions for breed in cat_breeds)
            results = {
                'is_cat': is_cat,
                'predictions': [{'class': pred[1], 'probability': float(pred[2])} for pred in predictions]
            }
            return jsonify(results)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    return send_file('templates/index.html')

@app.route('/imagenet_classes.txt')
def imagenet_classes():
    return send_file('imagenet_classes.txt')

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0",port=80)