#!/usr/bin/env python3
"""
test_training.py

Test script to verify the model training pipeline using a single sample.
This helps ensure that data loading, preprocessing, and model training work correctly.
Optimized for CPU-only training.
"""
import os
import sys
import glob
import random

# Add parent directory to path to import from two_stream_aca_net.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure TensorFlow for CPU-only training
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CPU-specific optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations for CPU
os.environ['TF_DISABLE_MKL'] = '0'  # Enable MKL for CPU
os.environ['TF_NUM_INTEROP_THREADS'] = '4'  # Number of threads for inter-op parallelism
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # Number of threads for intra-op parallelism

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure memory settings
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    def __init__(self, rgb_data, flow_data, y, validation_split=0.2, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.rgb_data = rgb_data
        self.flow_data = flow_data
        self.y = y
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Split data into training and validation
        split_idx = int(len(rgb_data) * (1 - validation_split))
        self.train_rgb = rgb_data[:split_idx]
        self.train_flow = flow_data[:split_idx]
        self.train_y = y[:split_idx]
        self.val_rgb = rgb_data[split_idx:]
        self.val_flow = flow_data[split_idx:]
        self.val_y = y[split_idx:]
        
        # Calculate number of batches
        self._batch_count = max(1, len(self.train_rgb) // self.batch_size)
        
        print("\nDataGenerator initialized with:")
        print(f"RGB data shape: {self.rgb_data.shape}")
        print(f"Flow data shape: {self.flow_data.shape}")
        print(f"Labels shape: {self.y.shape}")
        print(f"Training samples: {len(self.train_rgb)}")
        print(f"Validation samples: {len(self.val_rgb)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches per epoch: {self._batch_count}")
    
    def __len__(self):
        return self._batch_count
    
    def __getitem__(self, idx):
        if idx >= self._batch_count:
            raise IndexError(f"Index {idx} out of range (0 to {self._batch_count-1})")
        
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.train_rgb))
        
        batch_rgb = self.train_rgb[start_idx:end_idx]
        batch_flow = self.train_flow[start_idx:end_idx]
        batch_y = self.train_y[start_idx:end_idx]
        
        print(f"\rProcessing batch {idx + 1}/{self._batch_count}", end="", flush=True)
        return (batch_rgb, batch_flow), batch_y
    
    def get_validation_data(self):
        return (self.val_rgb, self.val_flow), self.val_y
    
    def on_epoch_end(self):
        print("\nEpoch completed")
        print("Metrics for this epoch:")
        print(f"  - Processed {self._batch_count} batches")
        print(f"  - Training samples: {len(self.train_rgb)}")
        print(f"  - Validation samples: {len(self.val_rgb)}")

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
    
    # Create data generator with validation split
    train_gen = TestDataGenerator(rgb_data, flow_data, y, validation_split=0.2)
    val_data = train_gen.get_validation_data()
    
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
    
    # Compile model with more metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'])
    
    print("\n=== Starting Training ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        DebugCallback(),
        tf.keras.callbacks.ProgbarLogger(),
        tf.keras.callbacks.TensorBoard(log_dir=output_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv')),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train for 5 epochs with validation
    history = model.fit(
        train_gen,
        validation_data=val_data,
        epochs=5,
        verbose=1,
        callbacks=callbacks
    )
    
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

def find_matching_samples(base_dir, num_samples=10):
    """Find matching RGB and flow sample pairs from the precomputed data."""
    # Get all RGB files
    rgb_files = []
    for action_dir in ['2p0', '2p1', '3p0', '3p1', 'ft0', 'ft1', 'mp0', 'mp1']:
        rgb_pattern = os.path.join(base_dir, action_dir, f"{action_dir}_*_x264_rgb.npz")
        rgb_files.extend(glob.glob(rgb_pattern))
    
    # Filter out files that don't have matching flow files
    valid_pairs = []
    for rgb_file in rgb_files:
        flow_file = rgb_file.replace('_rgb.npz', '_flow.npz')
        if os.path.exists(flow_file) and os.path.getsize(rgb_file) > 0 and os.path.getsize(flow_file) > 0:
            valid_pairs.append((rgb_file, flow_file))
    
    # Randomly select num_samples pairs
    if len(valid_pairs) < num_samples:
        print(f"Warning: Only found {len(valid_pairs)} valid pairs, using all available")
        selected_pairs = valid_pairs
    else:
        selected_pairs = random.sample(valid_pairs, num_samples)
    
    return selected_pairs

def load_and_preprocess_samples(sample_pairs):
    """Load and preprocess multiple samples."""
    rgb_samples = []
    flow_samples = []
    labels = []
    
    for i, (rgb_path, flow_path) in enumerate(sample_pairs):
        print(f"\nProcessing sample {i+1}/{len(sample_pairs)}")
        print(f"RGB: {os.path.basename(rgb_path)}")
        print(f"Flow: {os.path.basename(flow_path)}")
        
        # Load data
        rgb_npz = np.load(rgb_path)
        flow_npz = np.load(flow_path)
        
        # Get data arrays
        rgb_data = rgb_npz['data'] if 'data' in rgb_npz else rgb_npz['arr_0']
        flow_data = flow_npz['data'] if 'data' in flow_npz else flow_npz['arr_0']
        
        # Pad sequences
        rgb_data = pad_sequence(rgb_data, SEQ_LENGTH)
        flow_data = pad_sequence(flow_data, SEQ_LENGTH)
        
        # Normalize flow
        flow_data = normalize_flow(flow_data)
        
        rgb_samples.append(rgb_data)
        flow_samples.append(flow_data)
        
        # Create label based on action class (extract from filename)
        action_class = os.path.basename(rgb_path).split('_')[0]
        class_idx = {'2p0': 0, '2p1': 1, '3p0': 2, '3p1': 3, 
                    'ft0': 4, 'ft1': 5, 'mp0': 6, 'mp1': 7}[action_class]
        label = np.zeros(8)
        label[class_idx] = 1
        labels.append(label)
    
    # Stack samples into batches
    rgb_batch = np.stack(rgb_samples)
    flow_batch = np.stack(flow_samples)
    labels_batch = np.stack(labels)
    
    print(f"\nFinal dataset shapes:")
    print(f"RGB batch shape: {rgb_batch.shape}")
    print(f"Flow batch shape: {flow_batch.shape}")
    print(f"Labels shape: {labels_batch.shape}")
    
    return rgb_batch, flow_batch, labels_batch

def test_training(base_dir, output_dir, num_samples=5):
    """Test training with multiple random samples, optimized for CPU."""
    print("\n=== Finding Sample Pairs ===")
    sample_pairs = find_matching_samples(base_dir, num_samples)
    
    print("\n=== Loading and Preprocessing Samples ===")
    rgb_data, flow_data, y = load_and_preprocess_samples(sample_pairs)
    
    # Create data generator with smaller batch size for CPU
    train_gen = TestDataGenerator(rgb_data, flow_data, y, validation_split=0.2, batch_size=1)
    val_data = train_gen.get_validation_data()
    
    # Clear any existing models and memory
    tf.keras.backend.clear_session()
    
    # Build and compile model with CPU-optimized settings
    rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)
    
    print("\n=== Building Model ===")
    print(f"RGB input shape: {rgb_shape}")
    print(f"Flow input shape: {flow_shape}")
    
    model = build_two_stream_aca_net(rgb_shape, flow_shape)
    
    # Disable mixed precision for CPU training
    tf.keras.mixed_precision.set_global_policy('float32')
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile model with CPU-optimized settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 'categorical_accuracy', 'top_k_categorical_accuracy'],
                 jit_compile=False)  # Disable XLA compilation for CPU stability
    
    print("\n=== Starting Training ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define callbacks with adjusted patience for CPU training
    callbacks = [
        DebugCallback(),
        tf.keras.callbacks.ProgbarLogger(),
        tf.keras.callbacks.TensorBoard(log_dir=output_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv')),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,  # Increased patience for CPU training
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Increased patience for CPU training
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train with adjusted epochs for CPU
    history = model.fit(
        train_gen,
        validation_data=val_data,
        epochs=10,  # Increased epochs since CPU training is slower
        verbose=1,
        callbacks=callbacks
    )
    
    print("\n=== Training Completed ===")
    print("Final metrics:")
    for metric, values in history.history.items():
        print(f"{metric}: {values[-1]:.4f}")
    
    # Save the final model and weights
    print("\n=== Saving Model ===")
    model_path = os.path.join(output_dir, 'final_model.keras')
    model.save(model_path)
    print(f"Full model saved to: {model_path}")
    
    weights_path = os.path.join(output_dir, 'model_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Test model training with multiple samples (CPU-optimized)")
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Base directory containing precomputed data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save test results')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='Number of random samples to use for testing')
    parser.add_argument('--rgb_path', type=str,
                      help='Path to a single RGB sample (for backward compatibility)')
    parser.add_argument('--flow_path', type=str,
                      help='Path to a single flow sample (for backward compatibility)')
    
    args = parser.parse_args()
    
    if args.rgb_path and args.flow_path:
        # Use single sample mode for backward compatibility
        test_single_sample_training(args.rgb_path, args.flow_path, args.output_dir)
    else:
        # Use multiple samples mode
        test_training(args.base_dir, args.output_dir, args.num_samples)

if __name__ == '__main__':
    main() 