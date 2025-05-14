import tensorflow as tf
import tf2onnx
# Tidak bekerja

# Muat model .h5
model = tf.keras.models.load_model('model/model.h5')

# Konversi ke ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path='waste_classifier.onnx')

print("Model berhasil dikonversi ke ONNX!")