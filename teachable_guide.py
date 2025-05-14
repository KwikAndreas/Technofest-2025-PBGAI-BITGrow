"""
Panduan langkah demi langkah untuk menggunakan model Teachable Machine
di aplikasi track_waste.py.

Script ini akan membantu Anda:
1. Mengkonfigurasi model Teachable Machine dengan benar
2. Menyesuaikan label agar konsisten
3. Menguji model dengan gambar sampel
"""

import os
import sys
import shutil
import re
import cv2
import numpy as np
import tensorflow as tf
import json
import glob

def check_teachable_machine_model(model_path):
    """Memeriksa model Teachable Machine dan melengkapi file pendukung"""
    print(f"\n=== Memeriksa Model Teachable Machine: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Error: File model tidak ditemukan: {model_path}")
        return False
    
    # Direktori model
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # 1. Periksa/buat labels.txt
    labels_path = os.path.join(model_dir, "labels.txt")
    metadata_path = os.path.join(model_dir, "metadata.json")
    
    labels = []
    
    # Coba baca labels dari metadata jika ada
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if 'labels' in metadata:
                labels = metadata['labels']
                print(f"Label ditemukan di metadata.json: {labels}")
        except Exception as e:
            print(f"Error membaca metadata.json: {e}")
    
    # Jika tidak ada metadata atau tidak ada labels di metadata, coba baca dari labels.txt
    if not labels and os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            print(f"Label ditemukan di labels.txt: {labels}")
        except Exception as e:
            print(f"Error membaca labels.txt: {e}")
    
    # Jika masih tidak ada labels, minta input pengguna
    if not labels:
        print("\nTidak ditemukan file labels.txt atau metadata.json dengan informasi label")
        print("Masukkan label kelas untuk model Anda (satu per baris, tekan Enter dua kali setelah selesai):")
        
        while True:
            line = input()
            if not line:
                break
            labels.append(line.strip())
        
        if not labels:
            print("Tidak ada label yang dimasukkan, menggunakan label default")
            labels = ["Class 1", "Class 2", "Class 3"]
    
    # Tanyakan apakah label yang terdeteksi sudah benar
    print(f"\nLabel terdeteksi/dimasukkan: {labels}")
    confirmed = input("Apakah label ini sudah benar? (y/n, default: y): ").lower() != 'n'
    
    if not confirmed:
        print("\nMasukkan label kelas baru (satu per baris, tekan Enter dua kali setelah selesai):")
        labels = []
        while True:
            line = input()
            if not line:
                break
            labels.append(line.strip())
    
    # Simpan labels.txt dan metadata.json
    try:
        with open(labels_path, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        print(f"File labels.txt disimpan di {labels_path}")
        
        # Buat metadata.json jika belum ada
        if not os.path.exists(metadata_path):
            metadata = {"labels": labels}
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"File metadata.json disimpan di {metadata_path}")
    except Exception as e:
        print(f"Error menyimpan file label: {e}")
    
    # 2. Pastikan nama file model mengandung 'teachable' atau 'tm_'
    if not ('teachable' in model_path.lower() or 'tm_' in model_path.lower()):
        new_model_path = os.path.join(model_dir, f"teachable_{os.path.basename(model_path)}")
        try:
            shutil.copy(model_path, new_model_path)
            print(f"Model disalin dengan prefiks 'teachable_': {new_model_path}")
            model_path = new_model_path
        except Exception as e:
            print(f"Error menyalin model: {e}")
    
    # 3. Buat model yang diadaptasi menggunakan model_adapter.py jika tersedia
    try:
        import importlib
        model_adapter_spec = importlib.util.find_spec('model_adapter')
        if model_adapter_spec is not None:
            from model_adapter import adapt_teachable_machine_model
            print("\nMengadaptasi model dengan model_adapter...")
            adapted_model = adapt_teachable_machine_model(model_path)
            if adapted_model:
                print("Model berhasil diadaptasi!")
        else:
            print("\nModule model_adapter tidak ditemukan. Untuk performa terbaik, buat file model_adapter.py")
            print("Lihat panduan di README.md untuk detail lebih lanjut.")
    except Exception as e:
        print(f"Error saat mengadaptasi model: {e}")
    
    print("\n=== Konfigurasi Model Teachable Machine Selesai ===")
    return True

def test_model_with_image(model_path, image_path=None):
    """Uji model dengan gambar tertentu atau gambar yang dipilih pengguna"""
    print(f"\n=== Menguji Model: {model_path} ===")
    
    if not os.path.exists(model_path):
        print(f"Error: File model tidak ditemukan: {model_path}")
        return False
    
    # Pilih gambar jika tidak disediakan
    if not image_path:
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(glob.glob(f"*.{ext}"))
            image_paths.extend(glob.glob(f"dataset/**/*.{ext}", recursive=True))
            image_paths.extend(glob.glob(f"images/*.{ext}"))
        
        if not image_paths:
            print("Tidak ditemukan file gambar untuk pengujian")
            return False
        
        print("\nPilih gambar untuk pengujian:")
        for i, path in enumerate(image_paths[:10]):  # Batasi ke 10 gambar
            print(f"[{i}] {path}")
        
        if len(image_paths) > 10:
            print(f"... dan {len(image_paths)-10} gambar lainnya")
        
        try:
            choice = int(input("\nPilih nomor gambar (atau -1 untuk batal): "))
            if choice == -1:
                return False
            image_path = image_paths[choice]
        except (ValueError, IndexError):
            print("Pilihan tidak valid")
            return False
    
    if not os.path.exists(image_path):
        print(f"Error: File gambar tidak ditemukan: {image_path}")
        return False
    
    # Baca label model
    model_dir = os.path.dirname(model_path)
    labels_path = os.path.join(model_dir, "labels.txt")
    labels = []
    
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error membaca labels.txt: {e}")
            labels = ["Class 1", "Class 2", "Class 3"]
    else:
        # Coba baca dari metadata.json
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'labels' in metadata:
                    labels = metadata['labels']
            except Exception as e:
                print(f"Error membaca metadata.json: {e}")
                labels = ["Class 1", "Class 2", "Class 3"]
        else:
            labels = ["Class 1", "Class 2", "Class 3"]
    
    # Jalankan pengujian
    print(f"\nLabel model: {labels}")
    print(f"Menguji model dengan gambar: {image_path}")
    
    # Load model (menggunakan model yang diadaptasi jika ada)
    adapted_model_path = os.path.splitext(model_path)[0] + "_adapted.h5"
    if os.path.exists(adapted_model_path):
        print(f"Menggunakan model yang diadaptasi: {adapted_model_path}")
        model_path = adapted_model_path
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model dimuat: {model_path}")
        
        # Load dan preprocess gambar
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Tidak dapat membaca gambar {image_path}")
            return False
        
        # Tampilkan gambar
        cv2.imshow("Test Image", img)
        cv2.waitKey(1000)  # Tampilkan selama 1 detik
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (model.input_shape[1], model.input_shape[2]))
        
        # Cek apakah ini model Teachable Machine
        is_teachable = 'teachable' in model_path.lower() or 'tm_' in model_path.lower()
        is_adapted = '_adapted' in model_path.lower()
        
        # Preprocess sesuai jenis model
        if is_teachable and not is_adapted:
            # Teachable Machine preprocessing [-1, 1]
            img_processed = img_resized.astype(np.float32) / 127.5 - 1.0
            print("Menggunakan normalisasi TeachableMachine: [-1,1]")
        else:
            # Normalisasi standar [0, 1]
            img_processed = img_resized.astype(np.float32) / 255.0
            print("Menggunakan normalisasi standar: [0,1]")
        
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Prediksi
        prediction = model.predict(img_batch)[0]
        
        # Tampilkan hasil
        print("\nHasil prediksi:")
        for i, (label, score) in enumerate(zip(labels, prediction)):
            print(f"{label}: {score*100:.2f}%")
        
        # Class dengan confidence tertinggi
        max_idx = np.argmax(prediction)
        print(f"\nPrediksi: {labels[max_idx]} ({prediction[max_idx]*100:.2f}%)")
        
        # Tutup window gambar setelah prediksi
        cv2.destroyAllWindows()
        
        return True
    except Exception as e:
        print(f"Error saat menguji model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fungsi utama"""
    print("\n=== Teachable Machine Guide ===")
    print("Tool ini membantu Anda menyiapkan dan menguji model Teachable Machine")
    
    # Cari model
    model_paths = []
    for ext in ['.h5', '.keras']:
        model_paths.extend(glob.glob(f"model/*{ext}"))
        model_paths.extend(glob.glob(f"*{ext}"))
    
    if not model_paths:
        print("Tidak ditemukan file model (.h5/.keras)")
        model_path = input("Masukkan path lengkap ke file model: ")
        if not os.path.exists(model_path):
            print(f"File tidak ditemukan: {model_path}")
            return
    else:
        print("\nModel yang ditemukan:")
        for i, path in enumerate(model_paths):
            print(f"[{i}] {path}")
        
        try:
            choice = int(input("\nPilih nomor model (atau -1 untuk masukkan path manual): "))
            if choice == -1:
                model_path = input("Masukkan path lengkap ke file model: ")
                if not os.path.exists(model_path):
                    print(f"File tidak ditemukan: {model_path}")
                    return
            else:
                model_path = model_paths[choice]
        except (ValueError, IndexError):
            print("Pilihan tidak valid")
            return
    
    # Periksa dan siapkan model
    if check_teachable_machine_model(model_path):
        # Tanya jika ingin langsung menguji model
        test_choice = input("\nApakah Anda ingin menguji model dengan gambar? (y/n, default: y): ").lower() != 'n'
        if test_choice:
            test_model_with_image(model_path)
        
        # Tampilkan instruksi penggunaan
        print("\n=== Cara Menggunakan Model di track_waste.py ===")
        print(f"1. Jalankan: python track_waste.py {model_path}")
        print("2. Pastikan label yang digunakan sudah benar")
        print("3. Untuk akurasi terbaik, gunakan versi yang diadaptasi dengan akhiran '_adapted.h5'")
        
        print("\nSelamat mencoba!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
