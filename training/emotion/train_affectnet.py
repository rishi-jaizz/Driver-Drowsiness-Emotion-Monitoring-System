import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Constants
RANDOM_STATE = 42
IMAGE_SHAPE = (48, 48)
INPUT_SHAPE = (48, 48, 1)
EPOCHS = 100
BATCH_SIZE = 32
NUM_CLASSES = 4

# Print versions
print(f"Keras version: {keras.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Paths
# Using absolute path resolution relative to this script location would be safer, 
# but sticking to notebook's relative path structure for now as requested.
# Assuming script is run from the directory it resides in.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../data/emotion_dataset")
train_path = os.path.join(DATA_DIR, "train")
val_path = os.path.join(DATA_DIR, "val")
test_path = os.path.join(DATA_DIR, "test")

print(f"Train path: {train_path}")
print(f"Val path: {val_path}")
print(f"Test path: {test_path}")

# Check directories
if not os.path.exists(train_path):
    print(f"Error: Train path does not exist: {train_path}")
    sys.exit(1)

# Data Generators
print("Creating Data Generators...")
datagen = ImageDataGenerator(
    rescale = 1./255, 
    rotation_range = 15, 
    width_shift_range = 10,
    height_shift_range = 10, 
    shear_range = 0.15,
    zoom_range = 0.15,
    horizontal_flip = True,
)

test_datagen = ImageDataGenerator(rescale = 1./255)

print("Loading Data...")
train_gen = datagen.flow_from_directory(
    train_path, 
    target_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE, 
    color_mode = 'grayscale',
    class_mode = 'categorical',
    shuffle = True
)

val_gen = datagen.flow_from_directory(
    val_path, 
    target_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE, 
    color_mode = 'grayscale',
    class_mode = 'categorical',
    shuffle = True
)

test_gen = test_datagen.flow_from_directory(
    test_path, 
    target_size = IMAGE_SHAPE,
    batch_size = BATCH_SIZE, 
    color_mode = 'grayscale',
    class_mode = 'categorical',
    shuffle = False,
)

classes = ['angry', 'happy', 'neutral', 'sad']

# Class Weights
print("Computing Class Weights...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")

# Model Definition
print("Defining Model...")
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.SpatialDropout2D(0.2),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.SpatialDropout2D(0.2),
    
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.SpatialDropout2D(0.2),
    
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# Callbacks
print("Setting up Callbacks...")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
# Save the best model during training
checkpoint = keras.callbacks.ModelCheckpoint('affectnet_model_best.keras', monitor='val_loss', save_best_only=True, verbose=1)
callbacks = [early_stopping, lr_scheduler, checkpoint]

# Training
print("Starting Training...")
history = model.fit(
    train_gen, 
    epochs=EPOCHS, 
    validation_data=val_gen, 
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# Plotting History
print("Plotting Training History...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved to training_history.png")

# Save Model
print("Saving Model...")
model.save('affectnet_model_final.keras')
model.save('affectnet_model_final.h5')
print("Models saved.")

# Evaluation
print("Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Predictions
print("Generating Predictions...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Confusion Matrix
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

print("Training Script Completed Successfully.")
