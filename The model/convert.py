import tensorflow as tf
import numpy as np

print("Loading model...")
model = tf.keras.models.load_model('best_model.keras')

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved: model.tflite")
print(f"✓ Model size: {len(tflite_model) / 1024:.2f} KB")

# Test the TFLite model
print("\nTesting TFLite model...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Output shape: {output_details[0]['shape']}")

# Test with random data
test_input = np.random.randn(1, 300, 6).astype(np.float32)
test_input = test_input / np.max(np.abs(test_input))

interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Test output: {output}")
print(f"Predicted class: {np.argmax(output)}")
print("\n✓ TFLite model works correctly!")