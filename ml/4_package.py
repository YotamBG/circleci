#!/usr/bin/env python3
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow import keras

def load_data(root_dir: str):
    """Load preprocessed train images & labels from pickle files."""
    images_path = os.path.join(root_dir, "train_images.pkl")
    labels_path = os.path.join(root_dir, "train_labels.pkl")
    with open(images_path, "rb") as f:
        images = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    return images, labels

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Define a simple CNN model."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(8, 3, strides=2, activation="relu", name="Conv1"),
        keras.layers.Flatten(name="Flatten"),
        keras.layers.Dense(num_classes, name="Output")
    ])

def main():
    # Determine paths
    ml_dir = os.path.dirname(os.path.abspath(__file__))    # .../ml
    repo_root = os.path.dirname(ml_dir)                    # project root
    data_dir = os.path.join(repo_root, "training_data")
    os.makedirs(data_dir, exist_ok=True)

    # Load data
    print("Loading training data...")
    X_train, y_train = load_data(data_dir)
    print(f"  X_train shape: {X_train.shape}, dtype={X_train.dtype}")

    # Build, compile, and train model
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(set(y_train)))
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("Training model for 1 epoch (warm-up)...")
    model.fit(X_train, y_train, epochs=1, batch_size=32)

    # Export model
    output_path = os.path.join(data_dir, "trained_model.keras")
    print(f"Saving model to {output_path} …")
    model.save(output_path)
    print("Model export complete.")

if __name__ == "__main__":
    main()
