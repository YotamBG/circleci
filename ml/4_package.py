import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle

# Load preprocessed data
with open('training_data/train_images.pkl', 'rb') as f:
    train_images = pickle.load(f)

with open('training_data/train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

# Define a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, 
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, name='Dense')
])

# Compile the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Train the model briefly to initialize weights
model.fit(train_images, train_labels, epochs=1)

# Simplify the model export process
script_dir = os.path.dirname(os.path.abspath(__file__))
# Save in .keras native format:
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(repo_root, 'training_data')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'trained_model.keras')
model.save(output_path)
print(f"Model exported to '{output_path}'")
