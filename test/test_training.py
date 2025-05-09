#!/usr/bin/env python3
"""
test_training.py

Test script to verify the model training pipeline using a single sample.
This helps ensure that data loading, preprocessing, and model training work correctly.
"""
import os
import sys

# Add parent directory to path to import from two_stream_aca_net.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import argparse
from two_stream_aca_net import (
    build_two_stream_aca_net,
    normalize_flow,
    augment_data,
    SEQ_LENGTH,
    IMG_SIZE
)

def pad_sequence(sequence, target_length):
    """Pad or truncate a sequence to match the target length."""
    if len(sequence) >= target_length:
        # Truncate to target length
        print(f"Truncating sequence from {len(sequence)} to {target_length} frames")
        return sequence[:target_length]
    else:
        # Pad by repeating the last frame
        padding_needed = target_length - len(sequence)
        print(f"Padding sequence from {len(sequence)} to {target_length} frames")
        padding = np.tile(sequence[-1:], (padding_needed, 1, 1, 1))
        return np.concatenate([sequence, padding], axis=0)

class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("\n=== Training Started ===")
        print("Model configuration:")
        print(f"Input shapes:")
        for i, input_shape in enumerate(self.model.input_shape):
            print(f"  Input {i}: {input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print("======================\n")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n=== Starting Epoch {epoch + 1} ===")
        print("Current learning rate:", float(self.model.optimizer.learning_rate))

    def on_batch_begin(self, batch, logs=None):
        if batch == 0:
            print("\nFirst batch shapes:")
            print(f"RGB data shape: {self.model.input[0].shape}")
            print(f"Flow data shape: {self.model.input[1].shape}")
            print(f"Labels shape: {self.model.output.shape}")

    def on_batch_end(self, batch, logs=None):
        if batch == 0:
            print("\nFirst batch metrics:")
            for metric, value in logs.items():
                print(f"{metric}: {value:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n=== Completed Epoch {epoch + 1} ===")
        print("Metrics:")
        for metric, value in logs.items():
            print(f"{metric}: {value:.4f}")
        print("======================\n")

class TestDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, rgb_data, flow_data, y, **kwargs):
        super().__init__(**kwargs)  # Properly initialize parent class
        self.rgb_data = rgb_data
        self.flow_data = flow_data
        self.y = y
        self._batch_count = 10  # Show 10 batches per epoch
        print("\nDataGenerator initialized with:")
        print(f"RGB data shape: {self.rgb_data.shape}")
        print(f"Flow data shape: {self.flow_data.shape}")
        print(f"Labels shape: {self.y.shape}")
        print(f"Number of batches per epoch: {self._batch_count}")
    
    def __len__(self):
        return self._batch_count
    
    def __getitem__(self, idx):
        if idx >= self._batch_count:
            raise IndexError(f"Index {idx} out of range (0 to {self._batch_count-1})")
        print(f"\rProcessing batch {idx + 1}/{self._batch_count}", end="", flush=True)
        return (self.rgb_data, self.flow_data), self.y
    
    def on_epoch_end(self):
        print("\nEpoch completed")
        print("Metrics for this epoch:")
        print(f"  - Processed {self._batch_count} batches")
        print(f"  - Input shapes: RGB {self.rgb_data.shape}, Flow {self.flow_data.shape}")
        print(f"  - Output shape: {self.y.shape}")

def test_single_sample_training(rgb_path, flow_path, output_dir):
    """Test training with a single sample."""
    print("\n=== Loading Sample Data ===")
    
    # Load data from NPZ files
    rgb_npz = np.load(rgb_path)
    flow_npz = np.load(flow_path)
    
    # Get the data array from the NPZ file
    rgb_data = rgb_npz['data'] if 'data' in rgb_npz else rgb_npz['arr_0']
    flow_data = flow_npz['data'] if 'data' in flow_npz else flow_npz['arr_0']
    
    print(f"\nOriginal data shapes:")
    print(f"RGB data shape: {rgb_data.shape}")
    print(f"Flow data shape: {flow_data.shape}")
    print(f"RGB data range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
    print(f"Flow data range: [{flow_data.min():.3f}, {flow_data.max():.3f}]")
    
    # Ensure both sequences are exactly SEQ_LENGTH frames
    rgb_data = pad_sequence(rgb_data, SEQ_LENGTH)
    flow_data = pad_sequence(flow_data, SEQ_LENGTH)
    
    print(f"\nAfter padding:")
    print(f"RGB data shape: {rgb_data.shape}")
    print(f"Flow data shape: {flow_data.shape}")
    
    # Normalize flow data
    flow_data = normalize_flow(flow_data)
    print(f"\nAfter flow normalization:")
    print(f"Flow data range: [{flow_data.min():.3f}, {flow_data.max():.3f}]")
    
    # Create a dummy dataset with a single sample
    # Add batch dimension
    rgb_data = np.expand_dims(rgb_data, axis=0)
    flow_data = np.expand_dims(flow_data, axis=0)
    
    print(f"\nFinal input shapes:")
    print(f"RGB batch shape: {rgb_data.shape}")
    print(f"Flow batch shape: {flow_data.shape}")
    
    # Create dummy label (assuming 8 classes)
    y = np.zeros((1, 8))
    y[0, 0] = 1  # Set first class as positive
    print(f"Label shape: {y.shape}")
    
    # Create data generator
    train_gen = TestDataGenerator(rgb_data, flow_data, y)
    
    # Build and compile model
    rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)
    
    print("\n=== Building Model ===")
    print(f"RGB input shape: {rgb_shape}")
    print(f"Flow input shape: {flow_shape}")
    
    model = build_two_stream_aca_net(rgb_shape, flow_shape)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile model with more verbose output
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("\n=== Starting Training ===")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Train for 2 epochs with debug callback and verbose output
    history = model.fit(train_gen,
                       epochs=2,
                       verbose=1,  # Changed to 1 for progress bar
                       callbacks=[
                           DebugCallback(),
                           tf.keras.callbacks.ProgbarLogger(),
                           tf.keras.callbacks.TensorBoard(log_dir=output_dir),
                           tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv')),
                           tf.keras.callbacks.ModelCheckpoint(
                               os.path.join(output_dir, 'best_model.keras'),
                               monitor='accuracy',
                               save_best_only=True,
                               mode='max',
                               verbose=1
                           )
                       ])
    
    print("\n=== Training Completed ===")
    print("Final metrics:")
    for metric, values in history.history.items():
        print(f"{metric}: {values[-1]:.4f}")
    
    # Save the final model and weights
    print("\n=== Saving Model ===")
    # Save full model
    model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(model_path)
    print(f"Full model saved to: {model_path}")
    
    # Save just the weights with correct extension
    weights_path = os.path.join(output_dir, 'model_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Test model training with a single sample")
    parser.add_argument('--rgb_path', type=str, required=True,
                      help='Path to a sample RGB npz file')
    parser.add_argument('--flow_path', type=str, required=True,
                      help='Path to the corresponding flow npz file')
    parser.add_argument('--output_dir', type=str, default='test_output',
                      help='Directory to save the test model')
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.rgb_path):
        print(f"Error: RGB file not found: {args.rgb_path}")
        return
    if not os.path.exists(args.flow_path):
        print(f"Error: Flow file not found: {args.flow_path}")
        return
    
    test_single_sample_training(args.rgb_path, args.flow_path, args.output_dir)

if __name__ == "__main__":
    main() 