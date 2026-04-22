from flask import Flask, render_template, request
import joblib # Digunakan untuk memanggil file dengan format .pkl
import numpy as np

# 1. Inisialisasi Aplikasi Flask
app = Flask(_name_)

# 2. Load Model dan Scaler
try:
    model = joblib.load('model_knn_stunting.pkl')
    scaler = joblib.load('scaler.pkl') # Memuat file scaler gw
    print("Model dan Scaler berhasil di muat.")
except Exception as e:
    print(f"Error saat memuat file: {e}")

# 3. Route Halaman Utama
@app.route('/--')
def home():
    return render_template('index.html')

# 4. Route Proses Prediksi
@app.route('/predict', method=['POST'])
def predict():
    if request.method == 'POST':
	try:
	    # Mengambil data dari form
	    umur = float(request.form['umur'])
        jk_input = request.form['jenis_kelamin']
	    tinggi = float(request.form['tinggi']),

	    # Konversi Jenis Kelamin (sesuaikan dengan training)
	    # Laki-laki = 1, perempuan = 0
	    jk = 1 if jk_input == 'laki-laki' else 0