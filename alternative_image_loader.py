"""
Alternative image file loader that doesn't depend on tkinter dialog.
This can be used if the tkinter dialog causes issues.
"""
import os
import sys
import cv2
import numpy as np

def list_image_files(directory="."):
    """List all image files in the specified directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    try:
        for file in os.listdir(directory):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(directory, file)
                image_files.append(full_path)
    except Exception as e:
        print(f"Error reading directory: {e}")
    
    return image_files

def select_image_console():
    """Let user select an image using console input"""
    print("\n=== Pilih File Gambar ===")
    
    current_dir = os.getcwd()
    print(f"Direktori saat ini: {current_dir}")
    
    # Ask if user wants to enter a different directory
    choice = input("Gunakan direktori ini? (y/n, default: y): ").lower()
    
    if choice == 'n':
        new_dir = input("Masukkan path direktori: ")
        if os.path.isdir(new_dir):
            current_dir = new_dir
        else:
            print(f"Direktori tidak valid: {new_dir}")
            print("Menggunakan direktori saat ini.")
    
    # Get list of image files
    image_files = list_image_files(current_dir)
    
    if not image_files:
        print(f"Tidak ada file gambar ditemukan di {current_dir}")
        
        # Allow direct entry of file path
        file_path = input("Masukkan path lengkap file gambar: ")
        if os.path.isfile(file_path):
            return file_path
        else:
            print("File tidak ditemukan.")
            return None
    
    # Show files for selection
    print("\nFile gambar yang tersedia:")
    for i, file_path in enumerate(image_files):
        print(f"[{i}] {os.path.basename(file_path)}")
    
    # Let user choose
    while True:
        try:
            choice = input("\nPilih nomor file (atau tekan Enter untuk batal): ")
            
            if not choice.strip():
                return None
                
            choice = int(choice)
            if 0 <= choice < len(image_files):
                return image_files[choice]
            else:
                print(f"Pilihan tidak valid. Masukkan angka antara 0-{len(image_files)-1}")
        except ValueError:
            print("Masukkan nomor yang valid")
    
    return None

if __name__ == "__main__":
    # Test function
    selected_file = select_image_console()
    
    if selected_file:
        print(f"File terpilih: {selected_file}")
        
        # Try to load and display the image
        img = cv2.imread(selected_file)
        if img is not None:
            print(f"Berhasil memuat gambar, ukuran: {img.shape}")
            
            # # Resize for display if needed
            # h, w = img.shape[:2]
            # if h > 800 or w > 1200:
            #     scale = min(800 / h, 1200 / w)
            #     img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            cv2.imshow("Selected Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Gagal memuat gambar.")
    else:
        print("Tidak ada file yang dipilih.")
