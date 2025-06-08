import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Load preprocessed data
with np.load('data_prepared.npz') as data:
    train_images = data['train_images']
    train_labels = data['train_labels']

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
output_dir = os.path.join(script_dir, 'training_data/trained_model')
model.save(output_dir)
print(f"Model exported to '{output_dir}'.")
