import tensorflow as tf

# Load model Keras (.h5)
model = tf.keras.models.load_model("model.h5")

# Convert ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Simpan file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("model.tflite berhasil dibuat")
