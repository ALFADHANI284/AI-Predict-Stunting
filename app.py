from flask import Flask, render_template, request
import joblib 
import numpy as np

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# 2. Load Model dan Scaler
try:
    model = joblib.load('model_knn.pkl')
    scaler = joblib.load('scaler.pkl')
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

            # Validasi Backend (jaga-jaga jika validasi HTML tembus)
            if not (1 <= umur <= 36):
                return render_template('index.html', error="Maaf Bun, aplikasi ini khusus untuk usia 1 - 36 Bulan ya.")
            if not (45 <= tinggi <= 110):
                return render_template('index.html', error="Maaf Bun, tinggi badan yang dimasukkan harus antara 45 - 110 cm.")

            # Konversi Jenis Kelamin
            # Laki Laki = 1, Perempuan = 0
            jk = 1 if jk_input.lower() == 'laki-laki' else 0
        
            # Susun fitur dan scaling
            fitur_raw = np.array([[umur, jk, tinggi]])
            fitur_scaled = scaler.transform(fitur_raw)
        
            # Melakukan prediksi
            prediksi = model.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            # Mapping Label, Warna CSS, dan Pesan untuk Bunda
            status_info = {
                0: {
                    "css": "normal", # Hijau Muda
                    "pesan": "Wah, tinggi si Kecil Normal Bun!",
                    "saran": "Pertumbuhannya sudah sangat baik. Terus jaga asupan nutrisi seimbang, berikan protein hewani seperti telur/ikan setiap hari, dan pantau terus tumbuh kembangnya di Posyandu ya, Bun!"
                },
                1: {
                    "css": "sangat-pendek", # Merah
                    "pesan": "Perhatian khusus ya Bun, si Kecil masuk kategori Sangat Pendek.",
                    "saran": "Jangan panik dulu, Bun. Segera konsultasikan ke dokter anak atau puskesmas terdekat agar mendapat penanganan yang tepat. Dengan tambahan nutrisi khusus dan kejar tumbuh, si Kecil masih bisa dikejar kok!"
                },
                2: {
                    "css": "pendek", # Pink
                    "pesan": "Halo Bun, tinggi si Kecil saat ini masuk kategori Pendek.",
                    "saran": "Yuk tingkatkan lagi asupan protein hewani si Kecil dan pastikan ia cukup istirahat. Jangan lupa rutin ke Posyandu untuk memantau perkembangannya dan konsultasi gizi ya, Bun."
                },
                3: {
                    "css": "tinggi", # Hijau Tua
                    "pesan": "Hebat Bun! Tinggi si Kecil masuk kategori Tinggi.",
                    "saran": "Pertumbuhannya sangat optimal! Tetap berikan makanan bergizi seimbang dan stimulasi bermain yang cukup agar si Kecil tumbuh cerdas, kuat, dan aktif!"
                }
            }

            # Ambil data berdasarkan hasil prediksi model
            hasil = status_info.get(angka_hasil)
            
            return render_template('index.html', hasil=hasil)

        except Exception as e:
            return render_template('index.html', error=f'Kesalahan Sistem: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)