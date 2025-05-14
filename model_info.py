"""
Utility untuk memeriksa informasi model dan memastikan ukuran input yang benar.
Gunakan tool ini untuk memeriksa model sebelum menggunakannya dalam aplikasi.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from custom_load import load_model_safely

def inspect_model(model_path):
    """Memeriksa dan menampilkan informasi tentang model"""
    print(f"Memeriksa model: {model_path}")
    
    # Cek file model
    if not os.path.exists(model_path):
        print(f"Error: File model tidak ditemukan: {model_path}")
        return False
    
    # Coba muat model
    try:
        print("Mencoba memuat model dengan custom_load...")
        model = load_model_safely(model_path)
        
        if model is None:
            print("Mencoba memuat model dengan metode standar...")
            model = load_model(model_path)
        
        # Tampilkan informasi model
        print("\n=== Informasi Model ===")
        print(f"Input shape: {model.input_shape}")
        output_shape = model.output_shape
        print(f"Output shape: {output_shape}")
        
        # Cek ukuran input
        if len(model.input_shape) == 4:
            height, width = model.input_shape[1], model.input_shape[2]
            print(f"Model mengharapkan gambar berukuran: {height}x{width} piksel")
            
            # Cek apakah ukuran sama dengan yang diharapkan di track_waste.py
            # Jika berbeda, beri saran
            with open('track_waste.py', 'r') as file:
                content = file.read()
                import re
                match = re.search(r'IMG_SIZE\s*=\s*(\d+)', content)
                if match:
                    img_size = int(match.group(1))
                    if img_size != height:
                        print(f"\nPERINGATAN: track_waste.py menggunakan IMG_SIZE={img_size}, " 
                              f"tetapi model mengharapkan {height}!")
                        print(f"Disarankan untuk mengubah IMG_SIZE menjadi {height} di track_waste.py")
        
        # Cek output
        num_classes = output_shape[-1] if len(output_shape) >= 2 else 1
        print(f"Jumlah kelas (output): {num_classes}")
        
        # Coba prediksi dengan input dummy
        print("\nMencoba prediksi dengan data dummy...")
        dummy_input = np.random.random((1,) + model.input_shape[1:])
        predictions = model.predict(dummy_input)
        print(f"Hasil prediksi (bentuk): {predictions.shape}")
        
        # Jika output adalah categorical, tampilkan prediksi untuk setiap kelas
        if len(predictions.shape) == 2 and predictions.shape[1] > 1:
            print("Probabilitas untuk setiap kelas:")
            for i in range(predictions.shape[1]):
                print(f"  Kelas {i}: {predictions[0][i]:.4f}")
        
        print("\nModel diperiksa berhasil!")
        return True
        
    except Exception as e:
        print(f"Error saat memeriksa model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Jika ada argumen, gunakan sebagai path model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Jika tidak, cari model di direktori model/ dan tampilkan pilihan
        model_dir = "model"
        if not os.path.exists(model_dir):
            print(f"Error: Direktori {model_dir} tidak ditemukan.")
            sys.exit(1)
            
        import glob
        model_files = []
        for ext in ['*.h5', '*.keras']:
            model_files.extend(glob.glob(os.path.join(model_dir, ext)))
            
        if not model_files:
            print(f"Tidak ada model ditemukan di {model_dir}")
            sys.exit(1)
            
        print("Pilih model untuk diperiksa:")
        for i, path in enumerate(model_files):
            print(f"[{i}] {os.path.basename(path)}")
            
        while True:
            try:
                choice = input("\nMasukkan nomor model: ")
                choice = int(choice)
                if 0 <= choice < len(model_files):
                    model_path = model_files[choice]
                    break
                else:
                    print(f"Pilihan tidak valid. Masukkan angka 0-{len(model_files)-1}")
            except ValueError:
                print("Masukkan nomor yang valid")
    
    # Periksa model
    inspect_model(model_path)
