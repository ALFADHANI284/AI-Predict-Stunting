from flask import Flask, render_template, request
import joblib 
import numpy as np

# 1. Inisialisasi Aplikasi Flask
app = Flask(__name__)

# 2. Load Model dan Scaler
try:
    # --- Model Stunting --
    model_stunting = joblib.load('models/model_knn_stunting.pkl')
    scaler_stunting = joblib.load('scaler/scaler_stunting.pkl')

    # --- Model Obesitas ---
    model_obesitas = joblib.load('models/model_knn_obesitas.pkl')
    scaler_obesitas = joblib.load('scaler/scaler_obesitas.pkl')

    print("Semua Model dan Scaler berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat file: {e}")

# 3. Route Halaman Utama
@app.route('/')
def home():
    return render_template('index.html')

# 4. ROUTE AI STUNTING
@app.route('/predict_stunting', methods=['POST'])
def predict_stunting():
    if request.method == 'POST':
        try:
            # Mengambil data dari form
            umur = float(request.form['umur'])
            jk_input = request.form['jenis_kelamin']
            tinggi = float(request.form['tinggi'])

            # Validasi Backend
            if not (1 <= umur <= 36):
                return render_template('index.html', error_stunting="Maaf Bun, aplikasi ini khusus untuk usia 1 - 36 Bulan ya.")
            if not (45 <= tinggi <= 110):
                return render_template('index.html', error_stunting="Maaf Bun, tinggi badan yang dimasukkan harus antara 45 - 110 cm.")

            # Konversi Jenis Kelamin (Laki Laki = 1, Perempuan = 0)
            jk = 1 if jk_input.lower() == 'laki-laki' else 0
        
            # Susun fitur dan scaling
            fitur_raw = np.array([[umur, jk, tinggi]])
            fitur_scaled = scaler_stunting.transform(fitur_raw)
        
            # Melakukan prediksi
            prediksi = model_stunting.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            # Mapping Label
            status_info = {
                0: {
                    "css": "normal", 
                    "pesan": "Wah, tinggi si Kecil Normal Bun!", 
                    "saran": "Pertumbuhannya sudah sangat baik. Terus jaga asupan nutrisi seimbang, berikan protein hewani seperti telur/ikan setiap hari, dan pantau terus tumbuh kembangnya di Posyandu ya, Bun!"
                },
                1: {
                    "css": "sangat-pendek", 
                    "pesan": "Perhatian khusus ya Bun, si Kecil masuk kategori Sangat Pendek.", 
                    "saran": "Jangan panik dulu, Bun. Segera konsultasikan ke dokter anak atau puskesmas terdekat agar mendapat penanganan yang tepat. Dengan tambahan nutrisi khusus dan kejar tumbuh, si Kecil masih bisa dikejar kok!"
                },
                2: {
                    "css": "pendek", 
                    "pesan": "Halo Bun, tinggi si Kecil saat ini masuk kategori Pendek.", 
                    "saran": "Yuk tingkatkan lagi asupan protein hewani si Kecil dan pastikan ia cukup istirahat. Jangan lupa rutin ke Posyandu untuk memantau perkembangannya dan konsultasi gizi ya, Bun."
                },
                3: {
                    "css": "tinggi", 
                    "pesan": "Hebat Bun! Tinggi si Kecil masuk kategori Tinggi.", 
                    "saran": "Pertumbuhannya sangat optimal! Tetap berikan makanan bergizi seimbang dan stimulasi bermain yang cukup agar si Kecil tumbuh cerdas, kuat, dan aktif!"
                }
            }

            hasil = status_info.get(angka_hasil)
            return render_template('index.html', hasil_stunting=hasil)

        except Exception as e:
            return render_template('index.html', error_stunting=f'Kesalahan Sistem: {str(e)}')
        
# 5. ROUTE AI OBESITAS
@app.route('/predict_obesitas', methods=['POST'])
def predict_obesitas():
    if request.method == 'POST':
        try:
            # Mengambil data dari form baru
            nama = request.form['nama']
            umur = float(request.form['umur'])
            gender = int(request.form['gender']) # 1 untuk Male, 0 untuk Female
            tinggi = float(request.form['tinggi'])
            berat = float(request.form['berat'])

            # Susun fitur dan scaling
            fitur_raw = np.array([[umur, gender, tinggi, berat]])
            fitur_scaled = scaler_obesitas.transform(fitur_raw)
        
            # Melakukan prediksi
            prediksi = model_obesitas.predict(fitur_scaled)
            angka_hasil = int(prediksi[0])

            # Mapping Label Obesitas (Sesuaikan isinya dengan label asli dari model lo)
            # Misalnya: 0 = Underweight, 1 = Normal weight, 2 = Overweight, 3 = Obesity
            # Mapping Kategori, CSS, dan Saran (Obesitas)
            status_info_obesitas = {
                0: {
                    "css": "sangat-pendek",
                    "kategori": "Underweight (Kekurangan Berat Badan / Terlalu Kurus)",
                    "saran": "Tubuhmu butuh asupan energi ekstra. Coba tambahkan porsi protein dan camilan sehat seperti kacang-kacangan atau alpukat setiap hari."
                },
                1: {
                    "css": "normal",
                    "kategori": "Normal Weight (Berat Badan Ideal)",
                    "saran": "Kerja bagus! Tubuhmu berada di kondisi prima. Tetap jaga pola makan bergizi seimbang dan hidrasi yang cukup ya."
                },
                2: {
                    "css": "pendek",
                    "kategori": "Overweight (Kelebihan Berat Badan / Gemuk)",
                    "saran": "Sedikit perubahan akan sangat berarti. Cobalah aktif bergerak minimal 30 menit sehari dan kurangi minuman manis secara bertahap."
                },
                3: {
                    "css": "sangat-pendek",
                    "kategori": "Obesity (Obesitas / Sangat Gemuk)",
                    "saran": "Kesehatanmu adalah prioritas utama. Fokuslah pada makanan utuh dan jangan ragu untuk berdiskusi dengan ahli gizi demi program yang aman."
                }
            }

            # Ambil data dari dictionary
            data_obesitas = status_info_obesitas.get(angka_hasil)
            
            # Gabungin nama inputan user dengan pesan kategorinya
            pesan_final = f"Hai {nama}, berdasarkan data yang kamu masukkan, kamu termasuk kategori: {data_obesitas['kategori']}"

            # Bungkus ke dalam dictionary baru buat dikirim ke HTML
            hasil_akhir_obesitas = {
                "css": data_obesitas["css"],
                "pesan": pesan_final,
                "saran": data_obesitas["saran"]
            }

            return render_template('index.html', hasil_obesitas=hasil_akhir_obesitas)

        except Exception as e:
            return render_template('index.html', error_obesitas=f'Kesalahan Sistem: {str(e)}')
        
# 6. Jalankan Server
if __name__ == "__main__":
    app.run(debug=True)

