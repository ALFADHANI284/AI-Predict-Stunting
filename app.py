from flask import Flask, render_template, request
import joblib # Digunakan untuk memanggil file dengan format .pkl
import numpy as np

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# 2. Load Model dan Scaler
try:
    model = joblib.load('model_knn.pkl')
    scaler = joblib.load('scaler.pkl') # Memuat file scaler gw
    print("Model dan Scaler berhasil di muat.")
except Exception as e:
    print(f"Error saat memuat file: {e}")

# 3. Route Halaman Utama
@app.route('/')
def home():
    return render_template('index.html')

# 4. Route Proses Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Mengambil data dari form
            umur = float(request.form['umur'])
            jk_input = request.form['jenis_kelamin']
            tinggi = float(request.form['tinggi'])

            # Konversi Jenis Kelamin (Sesuaikan dengan training anda)
            # .lower() ditambahkan agar tidak error jika user input huruf kapital
            #Laki Laki  = 1, Perempuan = 0
            jk = 1 if jk_input.lower() == 'laki-laki' else 0
        
            # Susun fitur dalam bentuk array 2D
            # Urutan harus sama dengan saat training
            fitur_raw = np.array([[umur, jk, tinggi]])

            # Langkah Krusial: Scaling
            # Mengubah data input user menjai skala yang sma dengan data training
            fitur_scaled = scaler.transform(fitur_raw)
        
            # Melakukan prediksi
            prediksi = model.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            # Mapping Label
            status_map = {
                0: "Normal",
                1: "Sangat Pendek",
                2: "Pendek",
                3: "Tinggi"
            }
            hasil_teks = status_map.get(angka_hasil, f"kode: {angka_hasil}")
            
            return render_template('index.html', prediction_text=f'Hasil Analisis: {hasil_teks}')

        except Exception as e:
            # Jika terjadi error di dalam blok try
            return render_template('index.html', prediction_text=f'Kesalahan Sistem: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)

