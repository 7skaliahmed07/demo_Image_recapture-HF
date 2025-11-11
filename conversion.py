import tensorflow as tf
import os

print("🔄 Converting Keras model to TensorFlow Lite...")

# Load your original model
model = tf.keras.models.load_model('final_model.keras')
print("✅ Original model loaded")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# Convert
tflite_model = converter.convert()

# Save the converted model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TensorFlow Lite!")
print(f"📦 Original model: {os.path.getsize('final_model.keras') / (1024*1024):.1f} MB")
print(f"📦 TFLite model: {os.path.getsize('model.tflite') / (1024*1024):.1f} MB")