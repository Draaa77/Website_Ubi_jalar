import os
import shutil
import random
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Config
DATASET_DIR = "C:\\xampp\\htdocs\\TA2025\\dataset_mentah"
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
SEED = 42

def split_dataset():
    if os.path.exists(BASE_DIR):
        print(f"Folder '{BASE_DIR}' sudah ada, lewati split dataset.")
        return

    print("Membagi dataset menjadi train dan val...")

    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print("Kelas ditemukan:", classes)

    for cls in classes:
        os.makedirs(os.path.join(TRAIN_DIR, cls))
        os.makedirs(os.path.join(VAL_DIR, cls))

        files = os.listdir(os.path.join(DATASET_DIR, cls))
        random.seed(SEED)
        random.shuffle(files)
        split_idx = int(len(files) * (1 - VALIDATION_SPLIT))

        train_files = files[:split_idx]
        val_files = files[split_idx:]

        for f in train_files:
            shutil.copy(os.path.join(DATASET_DIR, cls, f), os.path.join(TRAIN_DIR, cls, f))
        for f in val_files:
            shutil.copy(os.path.join(DATASET_DIR, cls, f), os.path.join(VAL_DIR, cls, f))

    print("Selesai membagi dataset.")

def create_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    split_dataset()

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=True,
                                       rotation_range=15,
                                       zoom_range=0.1)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(TRAIN_DIR,
                                                  target_size=IMAGE_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(VAL_DIR,
                                              target_size=IMAGE_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode='categorical')

    num_classes = len(train_gen.class_indices)
    print("Jumlah kelas:", num_classes)
    print("Class indices:", train_gen.class_indices)

    model = create_model(num_classes)
    model.summary()

    model.fit(train_gen,
              validation_data=val_gen,
              epochs=EPOCHS)

    model.save("model.h5")
    print("Model tersimpan ke model.h5")

    # Simpan mapping kelas (index -> class)
    idx2class = {v: k for k, v in train_gen.class_indices.items()}
    with open("idx2class.json", "w") as f:
        json.dump(idx2class, f)
    print("Mapping index ke kelas tersimpan ke idx2class.json")

if __name__ == "__main__":
    main()