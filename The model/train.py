import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

print("Loading data...")
# Load data
data = np.load('my_data.npz')
X_train = data['X_train']  # shape: (samples, 300, 6)
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize ONCE, per sample 
print("\nNormalizing data...")
X_train = X_train / np.max(np.abs(X_train), axis=(1,2), keepdims=True)
X_test = X_test / np.max(np.abs(X_test), axis=(1,2), keepdims=True)

print("Normalization complete!")

# Build model 
print("\nBuilding model...")
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(300, 6)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks 
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=60,  
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.keras', 
    monitor='val_accuracy', 
    save_best_only=True,
    verbose=1
)

# Train 
print("\nTraining model...")
print("This may take a while with augmented data...\n")

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,  
    batch_size=64,  
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\n{'='*50}")
print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"{'='*50}")

# Save final model
model.save('my_activity_model.keras')
print("\n✓ Model saved as: my_activity_model.keras")

# Detailed predictions
print("\nDetailed test results:")
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

activities = ['sit', 'jum', 'walk', 'run']
print("\nPer-class accuracy:")
for i, activity in enumerate(activities):
    mask = true_classes == i
    if mask.sum() > 0:
        acc = (predicted_classes[mask] == i).sum() / mask.sum()
        total = mask.sum()
        correct = (predicted_classes[mask] == i).sum()
        print(f"  {activity}: {acc*100:.2f}% ({correct}/{total})")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print("          ", "  ".join(f"{a:>6}" for a in activities))
for i, activity in enumerate(activities):
    print(f"{activity:>6}:", "  ".join(f"{cm[i,j]:>6}" for j in range(4)))

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("\n✓ Training history saved as: training_history.png")
plt.show()

print(f"\n{'='*50}")
print("Training complete!")
print(f"Final test accuracy: {test_acc*100:.2f}%")
print(f"{'='*50}")