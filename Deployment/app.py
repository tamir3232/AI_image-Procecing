from flask import Flask, render_template, request, session
from utils import *
from werkzeug.utils import secure_filename
from PIL import Image
import io
import os



app = Flask(__name__)
app.secret_key = 'mysecretkey123'
labels = gen_labels()
# Route untuk halaman utama
@app.route('/')
def home():
 
    return render_template('home.html')

# Route untuk memproses prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data gambar dari objek BytesIO
    image_data = request.files['image'].read()
    # Membuat objek BytesIO dari data gambar
    image_bytes = io.BytesIO(image_data)

    # Memuat objek gambar menggunakan PIL
    image = Image.open(image_bytes)
    img = preprocess(image)
    model = model_arc()
    model.load_weights("../weights/model.h5")
    # Proses gambar menggunakan model
    prediction = model.predict(img[np.newaxis, ...])
    prediksi = labels[np.argmax(prediction[0], axis=-1)]
    # Render template hasil prediksi dengan hasil dari model
    return render_template('result.html', prediction=prediksi)

if __name__ == '__main__':
    app.run(debug=True)