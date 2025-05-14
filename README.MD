# Waste Classification Model

Proyek ini adalah model pembelajaran mesin untuk mengklasifikasikan sampah ke dalam kategori Organik, Non-Organik, dan B3 (Bahan Berbahaya dan Beracun).

## Prasyarat

- Python 3.8 atau lebih baru
- TensorFlow 2.12.0
- Dependensi lain yang diperlukan

## Instalasi

1. Buat virtual environment:

```bash
python -m venv waste_env
```

2. Aktifkan virtual environment:

<div class="copy-button-container">
  <pre><code id="venv-command">waste_env\Scripts\activate</code></pre>
  <button class="copy-button" onclick="copyToClipboard('venv-command')">Copy</button>
</div>

3. Instal dependensi yang diperlukan:

```bash
pip install tensorflow==2.12.0 numpy matplotlib onnx==1.12.0 tf2onnx==1.9.0 protobuf==3.20.0 opencv-python
```

## Penggunaan

Untuk melatih model:

```bash
python train.py
```

Untuk mengkonversi model ke format ONNX (jika train.py gagal melakukan konversi):

```bash
python convert_model.py
```

### Menjalankan Aplikasi Klasifikasi Sampah

Aplikasi dapat menerima input dari kamera atau file gambar:

```bash
# Mode interaktif - pilih model dan sumber input
python track_waste.py

# Menggunakan kamera (default)
python track_waste.py --camera

# Menggunakan input file gambar
python track_waste.py --image

# Model terakhir yang paling akurat
python track_waste.py model.h5

# Menggunakan model tertentu
python track_waste.py model/model.h5

# Kombinasi opsi
python track_waste.py model/model.h5 --image

# Melihat daftar semua model tersedia
python track_waste.py --list

# Bantuan
python track_waste.py --help
```

### Jika Dialog Pemilihan File Bermasalah

Jika Anda mengalami masalah dengan dialog pemilihan file, gunakan alternatif berbasis konsol:

```bash
python alternative_image_loader.py
```

Ini akan menampilkan daftar gambar di direktori saat ini dan memungkinkan Anda memilih melalui konsol.

### Troubleshooting Mode Gambar

Jika mengalami masalah saat menggunakan mode gambar:

1. **Dialog Pemilihan File Tidak Muncul**: Dialog mungkin terbuka di belakang jendela lain. Periksa taskbar atau alt+tab.

2. **Program Tidak Responsif**: Tekan Ctrl+C di terminal untuk menghentikan program.

3. **Jendela OpenCV Tidak Muncul**: Pastikan gambar yang dipilih valid dan dapat dibaca.

4. **Error Saat Memilih ROI**: Klik dan tarik mouse untuk memilih area, lalu tekan ENTER. Tekan ESC untuk batal.

5. **Error Sintaks Python**: Pastikan versi Python Anda 3.6+ dan sintaks yang digunakan sesuai (misalnya gunakan "or" bukan "atau").

6. **Error Ketidakcocokan Ukuran Input**: Jika Anda melihat error seperti "expected shape=(None, 224, 224, 3), found shape=(1, 512, 512, 3)", artinya ukuran input tidak sesuai dengan yang diharapkan model. Program akan mencoba menyesuaikan ukuran secara otomatis, tetapi jika masih bermasalah, coba sesuaikan konstanta IMG_SIZE di awal file track_waste.py.

## Struktur Dataset

Dataset harus diatur dalam struktur folder berikut:

### Menggunakan Model dari Teachable Machine

Model dari [Teachable Machine](https://teachablemachine.withgoogle.com/) memerlukan penanganan khusus agar hasil prediksi lokalnya seakurat di website.

#### Panduan Cepat Menggunakan Teachable Machine Model

1. **Ekspor Model dari Teachable Machine**

   - Di website Teachable Machine, pilih **Export Model**
   - Pilih **Tensorflow → Keras**
   - Download file model (.h5) dan file metadata

2. **Persiapkan Model dengan Tool Otomatis**

   ```bash
   python teachable_guide.py
   ```

   Tool ini akan membantu Anda:

   - Memverifikasi label kelas
   - Menyimpan label dalam format yang tepat
   - Mengadaptasi model untuk akurasi optimal
   - Menguji model dengan gambar sampel

3. **Jalankan dengan Model yang Disiapkan**
   ```bash
   python track_waste.py model/teachable_your_model_adapted.h5 --image
   ```

#### Mengapa Hasil di Website dan Lokal Berbeda?

1. **Perbedaan Preprocessing**

   - Website: Menggunakan normalisasi `[-1, 1]` dengan operasi `(pixel / 127.5) - 1.0`
   - Aplikasi lokal standar: Menggunakan normalisasi `[0, 1]` dengan operasi `pixel / 255.0`

2. **Urutan Label**

   - Urutan label di model harus sama dengan di aplikasi lokal

3. **Ukuran Input**
   - Model Teachable Machine biasanya mengharapkan input 224x224 piksel

#### Solusi

Script `track_waste.py` telah diperbarui untuk:

1. Mendeteksi model Teachable Machine secara otomatis
2. Menerapkan preprocessing yang sesuai (normalisasi `[-1, 1]`)
3. Membaca label kelas dari `labels.txt` atau `metadata.json` jika tersedia
4. Menyesuaikan ukuran input sesuai kebutuhan model

Untuk akurasi terbaik, gunakan model yang diadaptasi dengan akhiran `_adapted.h5` yang dibuat oleh `teachable_guide.py` atau `model_adapter.py`.

## Troubleshooting

# ...existing code...
