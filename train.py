import sys
import os
import datetime

def check_imports():
    try:
        import tensorflow as tf
        print("TensorFlow installation found!")
        print(f"TensorFlow version: {tf.__version__}")

        try:
            tf.constant([1, 2, 3])
            print("TensorFlow is working correctly!")
        except Exception as e:
            print(f"TensorFlow is installed but not working properly: {e}")
            print("Try reinstalling with: pip install tensorflow==2.12.0")
            return False

        try:
            import numpy as np
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            import onnx
            import tf2onnx
            print("All other required packages are installed!")
            return True
        except ImportError as e:
            print(f"Error with other packages: {e}")
            return False

    except ImportError:
        print("TensorFlow is not installed!")
        print("Please run these commands:")
        print("python -m pip install --upgrade pip")
        print("pip install wheel setuptools")
        print("pip install tensorflow==2.12.0")
        return False

def check_dataset():
    data_dir = "dataset/train"
    categories = ["B3", "Non_Organik", "Organik" ]

    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found!")
        return False, 0

    found_categories = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            files = os.listdir(category_path)
            if len(files) > 0:
                found_categories.append(category)
                print(f"Found {len(files)} images in {category}")

    if len(found_categories) == 0:
        print("Error: No valid category folders found!")
        return False, 0

    print(f"Found {len(found_categories)} categories: {found_categories}")
    return True, len(found_categories)

if not os.path.exists("model"):
    os.makedirs("model")

if __name__ == "__main__":
    import_check = check_imports()
    dataset_check, num_classes = check_dataset()

    if not import_check or not dataset_check:
        sys.exit(1)

    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report

    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    BATCH_SIZE = 27
    EPOCHS = 10
    NUM_CLASSES = num_classes
    print(f"Training with {NUM_CLASSES} classes")

    data_dir = "dataset/train"

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=40,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    print("\nClass mapping:")
    for class_name, class_index in train_generator.class_indices.items():
        print(f"{class_name}: {class_index}")

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Test generator
    test_dir = "dataset/test"
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print("\nClass mapping (test):")
    for class_name, class_index in test_generator.class_indices.items():
        print(f"{class_name}: {class_index}")

    model = Sequential([
        Conv2D(32, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    print("\nEvaluating model performance on TEST set:")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc:.4f}")

    # validation_generator.reset()
    predictions = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    labels = list(validation_generator.class_indices.keys())

    correct_per_class = np.zeros(NUM_CLASSES)
    total_per_class = np.zeros(NUM_CLASSES)
    for true, pred in zip(y_true, y_pred):
        total_per_class[true] += 1
        if true == pred:
            correct_per_class[true] += 1

    per_class_accuracy = correct_per_class / total_per_class
    acc_df = pd.DataFrame({
        "Class": labels,
        "Accuracy": per_class_accuracy,
        "# Samples": total_per_class.astype(int)
    })
    print("\nPer-Class Accuracy:")
    print(acc_df)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    lrs = history.history.get("lr", [])
    if not lrs:
        lrs = [0.001] * len(history.history['loss'])

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'model/waste_classifier_{now}.h5'
    model.save(model_filename)

    fig, axs = plt.subplots(3, 2, figsize=(14, 14))

    axs[0, 0].plot(history.history['accuracy'], label='Train')
    axs[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axs[0, 0].set_title('Model Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()

    axs[0, 1].plot(history.history['loss'], label='Train')
    axs[0, 1].plot(history.history['val_loss'], label='Validation')
    axs[0, 1].set_title('Model Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    sns.heatmap(acc_df.set_index("Class")[["Accuracy"]], annot=True, cmap="Greens", fmt=".2f", ax=axs[1, 0])
    axs[1, 0].set_title("Per-Class Accuracy")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axs[1, 1])
    axs[1, 1].set_title("Confusion Matrix")
    axs[1, 1].set_xlabel("Predicted")
    axs[1, 1].set_ylabel("True")

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="YlGnBu", ax=axs[2, 0])
    axs[2, 0].set_title("Classification Report")

    axs[2, 1].plot(lrs)
    axs[2, 1].set_title("Learning Rate over Epochs")
    axs[2, 1].set_xlabel("Epoch")
    axs[2, 1].set_ylabel("Learning Rate")
    axs[2, 1].grid(True)

    plt.tight_layout()
    summary_filename = f"model/report/training_history_{now}.png"
    plt.savefig(summary_filename)
    plt.close()

    print("Training completed and model saved!")
    print(f"Model saved as {model_filename}")
    print(f"Training summary saved to {summary_filename}")

