import tensorflow as tf
from tensorflow import keras
import pickle
import os

# Load and preprocess the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Save preprocessed data using pickle
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure the 'training_data' directory exists
training_data_dir = os.path.join(script_dir, 'training_data')
os.makedirs(training_data_dir, exist_ok=True)

# Add verbose logging
print("Loading and preprocessing the Fashion MNIST dataset...")
print(f"train_images.shape: {train_images.shape}, dtype: {train_images.dtype}")
print(f"test_images.shape: {test_images.shape}, dtype: {test_images.dtype}")

# Wrap file saving in try-except blocks
try:
    with open(os.path.join(training_data_dir, 'train_images.pkl'), 'wb') as f:
        pickle.dump(train_images, f)
    print("Saved train_images.pkl")

    with open(os.path.join(training_data_dir, 'train_labels.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
    print("Saved train_labels.pkl")

    with open(os.path.join(training_data_dir, 'test_images.pkl'), 'wb') as f:
        pickle.dump(test_images, f)
    print("Saved test_images.pkl")

    with open(os.path.join(training_data_dir, 'test_labels.pkl'), 'wb') as f:
        pickle.dump(test_labels, f)
    print("Saved test_labels.pkl")
except Exception as e:
    print(f"Error saving data: {e}")

print("Data preparation complete. Saved as pickle files.")
