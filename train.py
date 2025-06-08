import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os

# Load the data from the build step
with open('./training_data/train_images.pkl', 'rb') as f:
    train_images = pickle.load(f)

with open('./training_data/train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

with open('./training_data/test_images.pkl', 'rb') as f:
    test_images = pickle.load(f)

with open('./training_data/test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f)

# Define a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, 
                        strides=2, activation='relu', name='Conv1'),
    keras.layers.Flatten(),
    keras.layers.Dense(10, name='Dense')
])
model.summary()

# Compile and train the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

epochs = 5
model.fit(train_images, train_labels, epochs=epochs)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc}')

# Save the trained model
model.save(os.path.join('training_data/trained_model'))
print("Model training complete. Saved to 'training_data/trained_model'.")
