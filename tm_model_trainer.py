import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

# Config
IMG_SIZE = 512
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

def create_teachable_machine_model(num_classes):
    """
    Membuat model mirip dengan Teachable Machine
    menggunakan MobileNetV2 sebagai base model
    """
    # Load model dasar MobileNetV2 (model yang digunakan Teachable Machine)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze layer-layer pada base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Tambahkan layer klasifikasi kustom
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    # Layer output dengan jumlah kelas yang terdeteksi (Organik, Non_Organik, B3)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Kompilasi model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    data_dir = "dataset/train"
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model():
    # Periksa keberadaan folder dataset
    if not os.path.exists("dataset/train"):
        print("Error: Folder dataset tidak ditemukan!")
        print("Pastikan dataset ditempatkan di folder 'dataset/train'")
        print("dengan subfolder untuk setiap kelas (Organik, Non_Organik, B3)")
        return
    
    # Periksa keberadaan folder model
    if not os.path.exists("model"):
        os.makedirs("model")
    
    # Persiapan data
    train_generator, validation_generator = prepare_data()
    num_classes = len(train_generator.class_indices)
    
    # Tampilkan informasi kelas
    print(f"Melatih model untuk {num_classes} kelas:")
    for class_name, class_index in train_generator.class_indices.items():
        print(f" - {class_name}: {class_index}")
    
    # Buat model
    model = create_teachable_machine_model(num_classes)
    
    # Tampilkan ringkasan model
    print("\nRingkasan Model Teachable Machine:")
    model.summary()
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model/tm_waste_classifier_{IMG_SIZE}x{IMG_SIZE}_{timestamp}.h5"
    callbacks = [
        # Simpan model terbaik
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Early stopping untuk mencegah overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True
        ),
        # Mengurangi learning rate jika performa stagnan
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Latih model
    print(f"\nMemulai pelatihan model dengan input size {IMG_SIZE}x{IMG_SIZE}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi model
    print("\nEvaluasi model pada data validasi:")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Akurasi Validasi: {val_accuracy:.4f}")
    
    # Simpan model format TF
    final_model_path = f"model/tm_waste_classifier_{IMG_SIZE}x{IMG_SIZE}.h5"
    model.save(final_model_path)
    print(f"Model disimpan ke {final_model_path}")
    
    # Fine-tuning: unfreeze beberapa layer teratas dari base model
    print("\nMemulai fine-tuning model...")
    base_model = model.layers[0]  # MobileNetV2 base model
    # Unfreeze 30 layer teratas (dari total sekitar 155 layer)
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Compile ulang dengan learning rate yang lebih kecil
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Latih model lagi dengan fine-tuning
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=15,  # Lebih sedikit epoch untuk fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi setelah fine-tuning
    print("\nEvaluasi model setelah fine-tuning:")
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Akurasi Validasi Final: {val_accuracy:.4f}")
    
    # Simpan model final
    fine_tuned_model_path = f"model/tm_waste_classifier_{IMG_SIZE}x{IMG_SIZE}_finetuned.h5"
    model.save(fine_tuned_model_path)
    print(f"Model fine-tuned disimpan ke {fine_tuned_model_path}")
    
    # Gabungkan history dari kedua tahap training
    total_history = {}
    for key in history.history:
        total_history[key] = history.history[key] + history_fine.history[key]
    
    # Visualisasi hasil training
    plt.figure(figsize=(12, 5))
    # Plot akurasi
    plt.subplot(1, 2, 1)
    plt.plot(total_history['accuracy'])
    plt.plot(total_history['val_accuracy'])
    plt.title(f'Model Accuracy (size: {IMG_SIZE}x{IMG_SIZE})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(total_history['loss'])
    plt.plot(total_history['val_loss'])
    plt.title(f'Model Loss (size: {IMG_SIZE}x{IMG_SIZE})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'model/tm_training_history_{IMG_SIZE}x{IMG_SIZE}.png')
    plt.close()
    
    print(f"Visualisasi hasil training disimpan ke: model/tm_training_history_{IMG_SIZE}x{IMG_SIZE}.png")
    
    # Simpan label kelas
    import json
    class_indices = train_generator.class_indices
    # Balik key-value untuk mendapatkan index->nama_kelas
    class_mapping = {v: k for k, v in class_indices.items()}
    # Simpan ke file JSON
    with open(f'model/tm_class_mapping_{IMG_SIZE}x{IMG_SIZE}.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print("Pelatihan model berhasil selesai!")

if __name__ == "__main__":
    print("Teachable Machine Style Waste Classifier Trainer")
    print("="*50)
    print(f"Resolusi Input: {IMG_SIZE}x{IMG_SIZE} pixels")
    print(f"Jumlah Epoch: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*50)
    
    try:
        train_model()
    except Exception as e:
        print(f"Error terjadi selama pelatihan: {str(e)}")
        import traceback
        traceback.print_exc()
