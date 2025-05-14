import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog
from custom_load import load_model_safely

DEFAULT_MODEL_PATH = 'model/waste_classifier.h5'
IMG_SIZE = 512
LABELS = ['Organik', 'Non_Organik', 'B3']
CONFIDENCE_THRESHOLD = 0.5
MODEL_DIR = 'model'

def select_model_from_dir(model_dir=MODEL_DIR):
    print("\n=== Pilih Model untuk Klasifikasi Sampah ===")
    
    model_files = []
    for ext in ['*.h5', '*.keras', '*/saved_model.pb']:
        model_files.extend(glob.glob(os.path.join(model_dir, ext)))
    
    if not model_files:
        print(f"Tidak ada model ditemukan di direktori {model_dir}")
        return None
        
    print("Model yang tersedia:")
    for i, model_path in enumerate(model_files):
        print(f"[{i}] {os.path.basename(model_path)}")
    
    while True:
        try:
            choice = input("\nPilih nomor model (atau tekan Enter untuk model default): ")
            if not choice.strip():
                return DEFAULT_MODEL_PATH
                
            choice = int(choice)
            if 0 <= choice < len(model_files):
                return model_files[choice]
            else:
                print(f"Pilihan tidak valid. Masukkan angka antara 0-{len(model_files)-1}")
        except ValueError:
            print("Masukkan nomor yang valid")

def select_input_source():
    print("\n=== Pilih Sumber Input ===")
    print("[1] Kamera")
    print("[2] File Gambar")
    
    while True:
        choice = input("Pilih sumber input (1/2): ").strip()
        if choice == "1":
            return "camera"
        elif choice == "2":
            return "image"
        else:
            print("Pilihan tidak valid. Masukkan 1 atau 2.")

def select_image_file():
    print("Membuka dialog pemilihan file gambar...")
    print("Jika dialog tidak muncul, cek di belakang jendela lain atau taskbar.")
    
    try:
        root = tk.Tk()
        root.withdraw()
        
        root.wm_attributes('-topmost', 1)
        
        root.update()
        
        file_path = filedialog.askopenfilename(
            title="Pilih File Gambar",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ],
            parent=root
        )
        
        root.destroy()
        
        if not file_path:
            print("Pemilihan file dibatalkan.")
            return None
            
        print(f"File terpilih: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saat memilih file: {e}")
        try:
            root.destroy()
        except:
            pass
        return None

class WasteTracker:
    def __init__(self, model_path=None, input_source="camera"):
        self.model = None
        self.input_source = input_source
        self.image_path = None
        
        if self.input_source == "image":
            self.image_path = select_image_file()
            if self.image_path is None or not os.path.exists(self.image_path):
                print("File gambar tidak valid, beralih ke mode kamera")
                self.input_source = "camera"
        
        if model_path is None:
            self.model_path = select_model_from_dir()
            if self.model_path is None:
                self.model_path = DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path
            
        print(f"Menggunakan model: {self.model_path}")
        
        self.load_model()
            
        self.tracking_box = None
        self.current_label = None
        self.confidence = 0
        self.frame_count = 0
        self.prediction_scores = [0, 0, 0]
    
    def load_model(self):
        global LABELS, IMG_SIZE
        
        if not os.path.exists(self.model_path):
            print(f"Error: Model tidak ditemukan di {self.model_path}")
            print("Gunakan model dummy untuk demo")
            self.create_dummy_model()
            return
            
        model_dir = os.path.dirname(self.model_path)
        labels_path = os.path.join(model_dir, "labels.txt")
        if os.path.exists(labels_path):
            try:
                print(f"Menemukan file labels.txt, mencoba membaca label kelas...")
                with open(labels_path, 'r') as f:
                    custom_labels = [line.strip() for line in f.readlines()]
                if custom_labels:
                    LABELS = custom_labels
                    print(f"Menggunakan label dari labels.txt: {LABELS}")
            except Exception as e:
                print(f"Error membaca labels.txt: {e}")
            
        try:
            self.is_teachable_machine = False
            filename = os.path.basename(self.model_path).lower()
            if 'teachable' in filename or 'tm_' in filename or 'converted_' in filename:
                self.is_teachable_machine = True
                print("Terdeteksi model dari Teachable Machine")
                metadata_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        if 'labels' in metadata:
                            LABELS = metadata['labels']
                            print(f"Menggunakan label dari metadata.json: {LABELS}")
                    except Exception as e:
                        print(f"Error membaca metadata.json: {e}")
                
            if self.is_teachable_machine:
                try:
                    import importlib
                    model_adapter_spec = importlib.util.find_spec('model_adapter')
                    if model_adapter_spec is not None:
                        from model_adapter import adapt_teachable_machine_model
                        print("Menggunakan model_adapter untuk model Teachable Machine")
                        self.model = adapt_teachable_machine_model(self.model_path, save_adapted=False)
                        if self.model is not None:
                            print("Model berhasil diadaptasi dengan preprocessing yang benar")
                            self.model_adapted = True
                            input_shape = self.model.input_shape
                            print(f"Model berhasil dimuat! Input shape: {input_shape}")
                            self._update_img_size(input_shape)
                            return
                        else:
                            print("Adaptasi model gagal, mencoba metode standar...")
                except ImportError:
                    print("Module model_adapter tidak tersedia, menggunakan metode standar...")
                except Exception as e:
                    print(f"Error saat menggunakan model_adapter: {e}")
                    print("Mencoba metode standar...")
            
            print(f"Memuat model dari {self.model_path} menggunakan custom_load...")
            self.model = load_model_safely(self.model_path)
            
            if self.model is not None:
                input_shape = self.model.input_shape
                print(f"Model berhasil dimuat! Input shape: {input_shape}")
                self._update_img_size(input_shape)
                self.model_adapted = False
                return
                
            print("Mencoba metode loading standar...")
            self.model = load_model(self.model_path)
            
            input_shape = self.model.input_shape
            self._update_img_size(input_shape)
            self.model_adapted = False
            
            print("Model berhasil dimuat dengan metode standar")
            
        except Exception as e:
            print(f"Error saat memuat model: {e}")
            print("Membuat model dummy untuk demo")
            self.create_dummy_model()
    
    def _update_img_size(self, input_shape):
        global IMG_SIZE
        if input_shape and len(input_shape) == 4:
            expected_size = input_shape[1]  # Height dari (None, H, W, C)
            if expected_size is not None and expected_size > 0:
                print(f"Menyesuaikan ukuran input ke {expected_size}x{expected_size} sesuai model")
                IMG_SIZE = expected_size
    
    def create_dummy_model(self):
        try:
            from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
            from tensorflow.keras.models import Model
            from tensorflow.keras.applications import MobileNetV2
            
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            predictions = Dense(len(LABELS), activation='softmax')(x)
            
            self.model = Model(inputs=base_model.input, outputs=predictions)
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            print("Model dummy berhasil dibuat")
        except Exception as e:
            print(f"Gagal membuat model dummy: {e}")
            self.model = None
            print("Menggunakan output acak untuk demo")
    
    def preprocess_image(self, frame, box=None):
        try:
            if box is None:
                img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            else:
                x, y, w, h = [int(v) for v in box]
                # Ensure coordinates are within frame bounds
                x, y = max(0, x), max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    return None
                
                img = cv2.resize(frame[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Preprocessing berdasarkan jenis model dan apakah sudah diadaptasi
            if hasattr(self, 'is_teachable_machine') and self.is_teachable_machine:
                if not hasattr(self, 'model_adapted') or not self.model_adapted:
                    # Teachable Machine menggunakan normalisasi [-1, 1]
                    img = img.astype(np.float32)
                    img = img / 127.5 - 1.0
                    print("Menggunakan normalisasi TeachableMachine: [-1,1]") if box is None else None
                else:
                    # Model sudah diadaptasi, gunakan normalisasi standar
                    img = img.astype(np.float32) / 255.0
                    print("Menggunakan normalisasi standar: [0,1] (model sudah diadaptasi)") if box is None else None
            else:
                # Normalisasi standar [0, 1]
                img = img.astype(np.float32) / 255.0
                print("Menggunakan normalisasi standar: [0,1]") if box is None else None
                
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
    
    def predict(self, frame, box=None):
        try:
            processed = self.preprocess_image(frame, box)
            if processed is None:
                return "Unknown", 0.0, [0, 0, 0]
            
            # Jika model tidak ada, gunakan output acak
            if self.model is None:
                # Generate random prediction for demo
                rand_preds = np.random.random(len(LABELS))
                rand_preds = rand_preds / np.sum(rand_preds)  # Normalize to sum 1
                label_idx = np.argmax(rand_preds)
                return LABELS[label_idx], rand_preds[label_idx], rand_preds
            
            # Debugging: tampilkan rentang nilai input
            input_min = np.min(processed)
            input_max = np.max(processed)
            print(f"DEBUG: Rentang nilai input: [{input_min:.2f}, {input_max:.2f}]")
                
            prediction = self.model.predict(processed, verbose=0)[0]
            
            # Debugging: tampilkan nilai hasil prediksi
            print(f"DEBUG: Nilai prediksi: {prediction}")
            
            label_idx = np.argmax(prediction)
            
            return LABELS[label_idx], prediction[label_idx], prediction
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            # Return random values if prediction fails
            rand_preds = np.random.random(len(LABELS))
            rand_preds = rand_preds / np.sum(rand_preds)
            label_idx = np.argmax(rand_preds)
            return LABELS[label_idx], rand_preds[label_idx], rand_preds
    
    def draw_ui(self, frame, box):
        """Gambarkan antarmuka pengguna di atas frame"""
        if box is not None:
            x, y, w, h = [int(v) for v in box]
            # Draw tracking box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw title
            cv2.putText(frame, "Klasifikasi Sampah", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw all category probabilities with progress bars
            bar_width = 150
            bar_height = 15
            bar_gap = 25
            
            # Get prediction colors: green for highest, blue for others
            colors = [(0, 0, 255)] * len(LABELS)  # Default blue for all classes
            max_idx = np.argmax(self.prediction_scores)
            colors[max_idx] = (0, 255, 0)  # Green for highest score
            
            for i, label in enumerate(LABELS):
                # Skip if out of range (for safety)
                if i >= len(self.prediction_scores):
                    continue
                    
                # Position for this category
                pos_y = 50 + (i * bar_gap)
                
                # Score for this category
                score = self.prediction_scores[i] * 100
                
                # Draw category name and percentage
                text = f"{label}: {score:.1f}%"
                cv2.putText(frame, text, (10, pos_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, text, (10, pos_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Progress bar background
                cv2.rectangle(frame, (120, pos_y-12), (120 + bar_width, pos_y+2), 
                           (0, 0, 0), -1)
                
                # Progress bar fill
                filled_width = int(bar_width * self.prediction_scores[i])
                cv2.rectangle(frame, (120, pos_y-12), (120 + filled_width, pos_y+2), 
                           colors[i], -1)
            
            # Draw label for the highest confidence class
            if self.current_label:
                cv2.putText(frame, f"Hasil: {self.current_label}", (10, 125), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    
    def process_image_file(self):
        """Proses file gambar tunggal"""
        try:
            print(f"Mencoba membaca gambar dari: {self.image_path}")
            
            # Baca gambar dari file dengan pengecekan error
            frame = cv2.imread(self.image_path)
            if frame is None:
                print(f"Error: Tidak dapat membaca gambar dari {self.image_path}")
                print("Periksa apakah file ada dan merupakan format gambar yang valid.")
                return
            
            print(f"Berhasil membaca gambar. Dimensi: {frame.shape}")
                
            # Tampilkan gambar asli dengan ukuran yang lebih terkontrol
            orig_frame = frame.copy()
            # Resize jika gambar terlalu besar
            h, w = orig_frame.shape[:2]
            if h > 800 or w > 1200:  # Fixed 'atau' to 'or'
                scale = min(800 / h, 1200 / w)
                orig_frame = cv2.resize(orig_frame, (int(w * scale), int(h * scale)))
                print(f"Gambar diresize untuk tampilan: {orig_frame.shape}")
            
            # Buat window dengan ukuran yang dapat disesuaikan
            cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Original Image", orig_frame)
            # Berikan waktu untuk OpenCV menampilkan window
            cv2.waitKey(200)  
            
            print("Pilih region untuk dianalisis...")
            print("Klik dan drag untuk memilih area, lalu tekan ENTER.")
            print("Tekan ESC untuk membatalkan pilihan.")
            
            # Force bring windows to front and focus
            cv2.setWindowProperty("Original Image", cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty("Original Image", cv2.WND_PROP_TOPMOST, 0)
            
            # Berikan waktu ekstra untuk memastikan window sudah tampil
            cv2.waitKey(500)  
            
            self.tracking_box = cv2.selectROI("Original Image", orig_frame, False)
            
            if sum(self.tracking_box) > 0:  # Jika kotak dipilih
                # Buat salinan untuk menampilkan hasil
                result_frame = frame.copy()
                
                # Tampilkan pesan "Processing..."
                message_frame = result_frame.copy()
                cv2.putText(message_frame, "Processing...", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Result", message_frame)
                cv2.waitKey(1)
                
                # Lakukan klasifikasi
                self.current_label, self.confidence, predictions = self.predict(frame, self.tracking_box)
                
                # Handle predictions
                if hasattr(predictions, 'tolist'):
                    self.prediction_scores = predictions.tolist()
                else:
                    self.prediction_scores = predictions
                
                # Gambar UI
                self.draw_ui(result_frame, self.tracking_box)
                
                # Tampilkan hasil
                cv2.imshow("Result", result_frame)
                
                print(f"\nHasil Klasifikasi: {self.current_label}")
                for i, label in enumerate(LABELS):
                    print(f"{label}: {self.prediction_scores[i]*100:.2f}%")
                
                # Simpan hasil ke file jika diperlukan
                save_option = input("\nSimpan hasil sebagai gambar? (y/n): ").lower()
                if save_option == 'y':
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_filename = f"result_{timestamp}.jpg"
                    cv2.imwrite(output_filename, result_frame)
                    print(f"Hasil disimpan ke {output_filename}")
                
                print("\nTekan tombol apa saja untuk keluar...")
                cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error saat memproses gambar: {e}")
            import traceback
            traceback.print_exc()
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def run_camera_mode(self):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_ANY)
            
            if not cap.isOpened():
                print("Trying DirectShow backend...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
            if not cap.isOpened():
                print("Error: Cannot open camera. Please check if camera is connected.")
                return
                    
            print("Camera opened successfully")
            print("Press 'q' to quit, 's' to select a new region")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab first frame")
                return
                
            cv2.imshow("Waste Tracking", frame)
            cv2.waitKey(1) 
            
            selection_mode = True
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame")
                    break
                
                display_frame = frame.copy()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    selection_mode = True
                    
                if selection_mode or self.tracking_box is None:
                    cv2.putText(display_frame, "Select an object to track", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("Waste Tracking", display_frame)
                    cv2.waitKey(30)
                    
                    print("Select region to analyze...")
                    self.tracking_box = cv2.selectROI("Waste Tracking", frame, False)
                    
                    if sum(self.tracking_box) > 0:
                        message_frame = display_frame.copy()
                        cv2.putText(message_frame, "Processing...", (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Waste Tracking", message_frame)
                        cv2.waitKey(1)
                        
                        self.current_label, self.confidence, predictions = self.predict(frame, self.tracking_box)
                        if hasattr(predictions, 'tolist'):
                            self.prediction_scores = predictions.tolist()
                        else:
                            self.prediction_scores = predictions
                        print(f"Selected object classified as: {self.current_label}")
                        selection_mode = False
                    else:
                        # If selection was canceled
                        selection_mode = True
                        continue
                
                # Regular tracking mode
                if self.tracking_box and sum(self.tracking_box) > 0:
                    # Update prediction periodically (not every frame to avoid lag)
                    if self.frame_count % 15 == 0:
                        self.current_label, self.confidence, predictions = self.predict(frame, self.tracking_box)
                        # Periksa tipe data predictions dan tangani dengan benar
                        if hasattr(predictions, 'tolist'):
                            self.prediction_scores = predictions.tolist()
                        else:
                            # Jika sudah berupa list, gunakan langsung
                            self.prediction_scores = predictions
                    
                    # Draw UI
                    self.draw_ui(display_frame, self.tracking_box)
                
                # Show the processed frame
                cv2.imshow("Waste Tracking", display_frame)
                self.frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        if self.input_source == "image" and self.image_path:
            self.process_image_file()
        else:
            self.run_camera_mode()

def display_help():
    print("\nTrack Waste - Aplikasi Klasifikasi Sampah")
    print("-----------------------------------------")
    print("Penggunaan:")
    print("  python track_waste.py                      : Interaktif memilih model dan sumber input")
    print("  python track_waste.py [path_model]         : Gunakan model tertentu")
    print("  python track_waste.py --camera             : Gunakan input kamera (default)")
    print("  python track_waste.py --image              : Gunakan input file gambar")
    print("  python track_waste.py --list               : Tampilkan semua model tersedia")
    print("  python track_waste.py --help               : Tampilkan bantuan ini")
    print("\nContoh kombinasi:")
    print("  python track_waste.py model/custom_model.h5 --image : Gunakan model custom dengan input gambar")
    print("")

def list_available_models(model_dir=MODEL_DIR):
    print("\nModel yang tersedia di direktori:")
    
    model_files = []
    for ext in ['*.h5', '*.keras', '*/saved_model.pb']:
        model_files.extend(glob.glob(os.path.join(model_dir, ext)))
    
    if not model_files:
        print(f"Tidak ada model ditemukan di {model_dir}")
        return
        
    for i, model_path in enumerate(model_files):
        print(f"  {i+1}. {model_path}")
    print("")

if __name__ == "__main__":
    try:
        model_path = None
        input_source = "camera"
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i].lower()
            
            if arg == "--help" or arg == "-h":
                display_help()
                sys.exit(0)
            elif arg == "--list" or arg == "-l":
                list_available_models()
                sys.exit(0)
            elif arg == "--camera" or arg == "-c":
                input_source = "camera"
            elif arg == "--image" or arg == "-i":
                input_source = "image"
            elif arg.startswith("--"):
                print(f"Unknown option: {arg}")
                display_help()
                sys.exit(1)
            else:
                model_path = sys.argv[i]
            
            i += 1
        
        if len(sys.argv) == 1:
            input_source = select_input_source()
        
        tracker = WasteTracker(model_path, input_source)
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C).")
        try:
            cv2.destroyAllWindows()
        except:
            pass
    except Exception as e:
        print(f"Program crashed: {e}")
        import traceback
        traceback.print_exc()
        try:
            cv2.destroyAllWindows()
        except:
            pass
