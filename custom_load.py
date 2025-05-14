import tensorflow as tf
import numpy as np
import os

def load_model_safely(model_path):
    """
    Load a Keras model safely, handling the DepthwiseConv2D 'groups' parameter issue.
    
    Args:
        model_path: Path to the model file (.h5) or directory (SavedModel)
        
    Returns:
        Loaded model or None if loading fails
    """
    print(f"Mencoba memuat model dari {model_path}")
    
    if not os.path.exists(model_path):
        print(f"File model tidak ditemukan: {model_path}")
        return None
    
    # Metode 1: Coba dengan custom_objects
    try:
        # Definisikan custom class untuk menangani DepthwiseConv2D dengan parameter 'groups'
        class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, groups=None, **kwargs):
                # Hapus parameter 'groups' yang menyebabkan error
                super(FixedDepthwiseConv2D, self).__init__(*args, **kwargs)
        
        # Load model dengan custom_objects
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
            compile=False
        )
        print("Model berhasil dimuat dengan custom_objects")
        return model
    except Exception as e:
        print(f"Metode 1 gagal: {e}")
    
    # Metode 2: Patch deserializer
    try:
        # Simpan deserializer asli
        original_deserialize = tf.keras.utils.deserialize_keras_object
        
        # Define patched version yang menghapus parameter 'groups'
        def patched_deserialize(config, **kwargs):
            if isinstance(config, dict) and 'class_name' in config and config['class_name'] == 'DepthwiseConv2D':
                if 'config' in config and 'groups' in config['config']:
                    config = config.copy()
                    config['config'] = config['config'].copy()
                    del config['config']['groups']
            return original_deserialize(config, **kwargs)
        
        # Apply patch
        tf.keras.utils.deserialize_keras_object = patched_deserialize
        
        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Restore original
        tf.keras.utils.deserialize_keras_object = original_deserialize
        
        print("Model berhasil dimuat dengan patched deserializer")
        return model
    except Exception as e:
        # Restore original in case of error
        tf.keras.utils.deserialize_keras_object = original_deserialize
        print(f"Metode 2 gagal: {e}")
    
    # Metode 3: Coba dengan SavedModel jika input adalah .h5
    if model_path.endswith('.h5'):
        try:
            # Convert to SavedModel first
            temp_dir = "temp_saved_model"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Gunakan TF 1.x API yang lebih toleran untuk membaca model
            with tf.Graph().as_default():
                with tf.compat.v1.Session() as sess:
                    tf.compat.v1.keras.backend.set_session(sess)
                    model = tf.keras.models.load_model(model_path, compile=False)
                    tf.compat.v1.saved_model.save(model, temp_dir)
            
            # Load dari SavedModel
            model = tf.keras.models.load_model(temp_dir)
            print("Model berhasil dimuat dengan konversi ke SavedModel")
            return model
        except Exception as e:
            print(f"Metode 3 gagal: {e}")
    
    print("Semua metode gagal, model tidak dapat dimuat")
    return None

# Test function - uncomment to test directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "model/keras_model.h5"  # Default path
    
    model = load_model_safely(model_path)
    
    if model is not None:
        print("\nModel berhasil dimuat!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Test prediction
        print("\nMencoba prediksi dengan input acak...")
        dummy_input = np.random.random((1,) + model.input_shape[1:])
        prediction = model.predict(dummy_input)
        print(f"Hasil prediksi: {prediction}")
