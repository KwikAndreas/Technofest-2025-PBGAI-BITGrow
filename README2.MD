# 🧠 CNN Training Pipeline - `train.py`

Dokumen ini menjelaskan **alur lengkap training model Convolutional Neural Network (CNN)** dalam file `train.py`.
Model ini digunakan untuk mengklasifikasikan gambar sampah menjadi tiga kategori: `Organik`, `Non_Organik`, dan `B3`.

---

## 🚀 Tech Stack

| Teknologi           | Peran                                          |
| ------------------- | ---------------------------------------------- |
| TensorFlow/Keras    | Framework utama deep learning dan training CNN |
| NumPy               | Operasi array numerik                          |
| Matplotlib          | Visualisasi akurasi dan loss                   |
| ImageDataGenerator  | Augmentasi data gambar dan batch loader        |
| OS (Python stdlib)  | Manajemen direktori dan file lokal             |
| ONNX/tf2onnx (opsi) | Konversi model ke format universal             |
| HDF5 (.h5)          | Format penyimpanan model Keras                 |

---

## 📸 Dataset Struktur

```
dataset/
  train/
    Organik/
    Non_Organik/
    B3/
```

- Gambar diletakkan dalam subfolder sesuai dengan kelas.
- `ImageDataGenerator` akan membaca folder dan membuat label otomatis.

---

## 🔁 Alur Lengkap CNN Training

### 1. 🟩 Input Gambar

- Gambar dimuat dari folder `dataset/train/`
- Ukuran di-resize ke `(224, 224)`

### 2. 🧼 ImageDataGenerator

- Preprocessing dan augmentasi dilakukan menggunakan `ImageDataGenerator`
- Proses termasuk: normalisasi piksel, rotasi, shear, zoom, flip, dan brightness
- Dataset dibagi menjadi training dan validasi

### 3. 🧠 CNN - Feature Extraction

- Beberapa layer `Conv2D` digunakan untuk mengekstrak fitur dari gambar
- Setiap layer dilengkapi dengan aktivasi `ReLU`
- Setelah itu dilanjutkan dengan `MaxPooling2D` untuk menyaring dan mereduksi dimensi

```
Conv2D(32) + ReLU → MaxPooling2D
Conv2D(64) + ReLU → MaxPooling2D
Conv2D(128) + ReLU → MaxPooling2D
Conv2D(128) + ReLU → MaxPooling2D
```

### 4. 🧬 Flatten

- Output dari CNN (berbentuk 3D) diratakan menjadi vektor 1D

### 5. 📊 Dense Layers

```
Dense(128) + Dropout(0.5)
Dense(64)  + Dropout(0.3)
```

- Lapisan fully connected yang memproses hasil ekstraksi fitur
- Dropout membantu mencegah overfitting

### 6. 🔄 Hierarki Fitur

- CNN secara bertahap belajar dari fitur sederhana (tepi, tekstur) ke fitur kompleks (objek)
- Makin dalam layer, makin kompleks fitur yang dipelajari

### 7. 🤖 Backpropagation

- Model menghitung `loss` menggunakan `categorical_crossentropy`
- Optimizer `Adam` mengupdate bobot melalui backpropagation
- Callback seperti `EarlyStopping` dan `ReduceLROnPlateau` membantu efisiensi training

### 8. 🔮 Output (Softmax)

```
Dense(NUM_CLASSES, activation='softmax')
```

- Menghasilkan probabilitas klasifikasi gambar ke salah satu dari tiga kelas

### 9. 📊 Evaluasi dan Visualisasi

- Grafik akurasi dan loss disimpan sebagai file PNG
- Model dievaluasi terhadap data validasi

### 10. 📄 Penyimpanan Model

- Model disimpan ke: `model/waste_classifier.h5`

---

## 📈 Flowchart Proses Training

```
[Input Gambar dari Folder]
        ↓
[ImageDataGenerator: Augmentasi & Preprocessing]
        ↓
[Conv2D + ReLU (32)]
        ↓
[MaxPooling2D]
        ↓
[Conv2D + ReLU (64)]
        ↓
[MaxPooling2D]
        ↓
[Conv2D + ReLU (128)]
        ↓
[MaxPooling2D]
        ↓
[Conv2D + ReLU (128)]
        ↓
[MaxPooling2D]
        ↓
[Flatten]
        ↓
[Dense(128) + Dropout(0.5)]
        ↓
[Dense(64) + Dropout(0.3)]
        ↓
[Dense Softmax Output]
        ↓
[Backpropagation & Optimizer (Adam)]
        ↓
[Evaluasi & Visualisasi]
        ↓
[Simpan Model ke .h5]
```

---

## 🧠 Penutup

Model CNN dalam `train.py` bekerja dengan prinsip **feature hierarchy**: mengenali tepi, bentuk, hingga objek.
Training dilakukan menggunakan kombinasi augmentation, convolution, backpropagation, dan optimisasi modern seperti Adam dan early stopping.

Model ini dapat dikembangkan lebih lanjut untuk deployment, integrasi webcam, atau konversi ke ONNX/TFLite.

---

> Jika kamu ingin membuat prediksi dengan gambar baru atau deploy model ke API, langkah selanjutnya bisa diarahkan ke `inference.py` atau `app.py`.
