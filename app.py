import io
import os

import numpy as np
from PIL import Image
from flask import Flask, request, redirect, flash, jsonify
from keras.preprocessing import image
from keras.src.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from werkzeug.utils import secure_filename



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Dosya parçası yok')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Dosya seçilmedi')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return 'Dosya başarıyla yüklendi'
    else:
        return 'İzin verilen dosya türleri: png, jpg, jpeg, gif'


model = MobileNetV2(weights='imagenet')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya parçası yok'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya parçası yok'}), 400

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array_expanded_dims)

    predictions = model.predict(preprocessed_img)
    results = decode_predictions(predictions, top=3)[0]

    return jsonify({'predictions': [{'class': result[1], 'confidence': float(result[2])} for result in results]})

if __name__ == '__main__':
    app.run(debug=True)
