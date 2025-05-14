import tensorflow as tf
import tf2onnx
import onnx
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.layers import DepthwiseConv2D

# Define a custom function to handle the DepthwiseConv2D layer
def fix_depthwise_conv2d_config(config):
    # Remove the problematic 'groups' parameter if it exists
    if 'groups' in config:
        del config['groups']
    return config

# Create a custom DepthwiseConv2D class to handle loading
class CustomDepthwiseConv2D(DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config = fix_depthwise_conv2d_config(config)
        return cls(**config)

# Kompilasi model untuk menghindari warning
def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Loading model...")

try:
    # Load model dengan custom objects untuk menangani DepthwiseConv2D
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    model = load_model('model/keras_model.h5', custom_objects=custom_objects)
    
    # Kompilasi model untuk menghindari warning
    model = compile_model(model)
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.inputs[0].shape}")
    print(f"Model output shape: {model.outputs[0].shape}")
    
    # Tentukan input signature
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]
    
    # Set nama output
    model.output_names = ['output']
    
    print("Converting model to ONNX format...")
    # Konversi model ke ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=12)
    
    # Simpan model ONNX
    output_path = 'model/waste_classifier2.onnx'
    onnx.save(onnx_model, output_path)
    
    print(f"Model successfully converted and saved to {output_path}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nMencoba pendekatan alternatif...")
    
    try:
        # Pendekatan alternatif jika cara pertama gagal
        # Mendefinisikan sendiri model tanpa layer DepthwiseConv2D bermasalah
        
        # Pastikan model asli dimuat dengan mengabaikan error custom layer
        # Gunakan parameter custom_objects untuk mengganti definisi DepthwiseConv2D
        model = tf.keras.models.load_model('model/model.h5', compile=False, 
                                           custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        
        # Kompilasi model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Coba konversi lagi
        input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=12)
        onnx.save(onnx_model, 'model/waste_classifier.onnx')
        
        print("Model berhasil dikonversi dan disimpan sebagai 'model/waste_classifier.onnx'")
        
    except Exception as e2:
        print(f"Error pada pendekatan alternatif: {str(e2)}")
        print("\nMencoba pendekatan terakhir...")
        
        try:
            # Pendekatan terakhir: Simpan dan muat ulang model dalam format SavedModel
            model = tf.keras.models.load_model('model/model.h5', compile=False, 
                                              custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
            
            # Simpan sebagai SavedModel
            temp_saved_model_path = 'model/temp_saved_model'
            model.save(temp_saved_model_path, save_format='tf')
            
            # Konversi dari SavedModel
            import os
            if os.path.exists(temp_saved_model_path):
                onnx_model, _ = tf2onnx.convert.from_saved_model(
                    temp_saved_model_path,
                    output_path='model/waste_classifier.onnx'
                )
                print("Model berhasil dikonversi dari SavedModel!")
                
                # Hapus direktori temporary
                import shutil
                shutil.rmtree(temp_saved_model_path)
            else:
                print(f"Error: Direktori SavedModel tidak ditemukan di {temp_saved_model_path}")
                
        except Exception as e3:
            print(f"Semua pendekatan gagal. Error terakhir: {str(e3)}")
            print("\nSaran Troubleshooting:")
            print("1. Periksa apakah model benar-benar menggunakan DepthwiseConv2D")
            print("2. Coba konversi dengan model yang lebih sederhana")
            print("3. Jika menggunakan MobileNet, coba load model dengan weights=None terlebih dahulu")
            print("4. Downgrade TensorFlow dan tf2onnx ke versi yang kompatibel:")
            print("   - pip install tensorflow==2.8.0")
            print("   - pip install tf2onnx==1.9.0")
            print("   - pip install onnx==1.10.0")
            print("   - pip install numpy==1.21.0")