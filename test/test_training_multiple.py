#!/usr/bin/env python3
"""
test_training_multiple.py

Test script to verify the model training pipeline using 10 random samples.
This helps ensure that data loading, preprocessing, and model training work correctly
with multiple samples while maintaining the same pipeline as single sample testing.
"""
import os
import sys
import random
import glob
import time
from datetime import datetime

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

def normalize_rgb(rgb_data):
    """Normalize RGB data to [0, 1] range."""
    return rgb_data / 255.0

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

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.batch_start_time = None
        self.last_update_time = None
        self.gradient_start_time = None
    
    def on_train_begin(self, logs=None):
        print(f"\n=== Training Started at {datetime.now().strftime('%H:%M:%S')} ===")
        print("Model configuration:")
        print(f"Input shapes:")
        for i, input_shape in enumerate(self.model.input_shape):
            print(f"  Input {i}: {input_shape}")
        print(f"Output shape: {self.model.output_shape}")
        print("Model layers:")
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                print(f"  Layer {i}: {layer.name} - Input Layer")
            elif isinstance(layer, tf.keras.layers.TimeDistributed):
                print(f"  Layer {i}: {layer.name} - TimeDistributed Layer")
            else:
                try:
                    print(f"  Layer {i}: {layer.name} - {layer.output_shape}")
                except AttributeError:
                    print(f"  Layer {i}: {layer.name} - Shape not available")
        print("======================\n")
        self.last_update_time = time.time()
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"\n=== Starting Epoch {epoch + 1} at {datetime.now().strftime('%H:%M:%S')} ===")
        print(f"Current learning rate: {float(self.model.optimizer.learning_rate)}")
        self.last_update_time = time.time()
    
    def on_batch_begin(self, batch, logs=None):
        current_time = time.time()
        if self.last_update_time and (current_time - self.last_update_time) > 5:  # Reduced to 5 seconds
            if self.gradient_start_time:
                print(f"\nStill computing gradients... (taken {current_time - self.gradient_start_time:.1f}s so far)")
            else:
                print(f"\nStill processing batch {batch}... (taken {current_time - self.batch_start_time:.1f}s so far)")
            self.last_update_time = current_time
        
        self.batch_start_time = time.time()
        self.gradient_start_time = None
        if batch == 0:
            print("\nFirst batch details:")
            print(f"RGB data shape: {self.model.input[0].shape}")
            print(f"Flow data shape: {self.model.input[1].shape}")
            print(f"Labels shape: {self.model.output.shape}")
            print("Starting first batch processing...")
            self.last_update_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        batch_time = time.time() - self.batch_start_time
        print(f"\nBatch {batch + 1}/{self.params['steps']} completed in {batch_time:.1f}s")
        print(f"Metrics:")
        for metric, value in logs.items():
            print(f"  {metric}: {value:.4f}")
        self.last_update_time = time.time()
        self.gradient_start_time = None
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n=== Completed Epoch {epoch + 1} at {datetime.now().strftime('%H:%M:%S')} ===")
        print(f"Epoch duration: {epoch_time:.1f} seconds")
        print("Metrics:")
        for metric, value in logs.items():
            print(f"{metric}: {value:.4f}")
        print("==============================\n")
        self.last_update_time = time.time()

class MultiSampleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, rgb_data_list, flow_data_list, labels, batch_size=2, **kwargs):
        super().__init__(**kwargs)
        # Normalize RGB data
        self.rgb_data_list = [normalize_rgb(rgb) for rgb in rgb_data_list]
        self.flow_data_list = flow_data_list
        self.labels = labels
        self.batch_size = batch_size
        self.n_samples = len(rgb_data_list)
        self._batch_count = (self.n_samples + batch_size - 1) // batch_size
        
        # Verify data shapes and types
        print("\n=== Data Generator Initialization ===")
        print(f"Number of samples: {self.n_samples}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches per epoch: {self._batch_count}")
        
        # Check first sample shapes and data ranges
        print("\nFirst sample details:")
        print(f"RGB shape: {self.rgb_data_list[0].shape}")
        print(f"RGB range: [{np.min(self.rgb_data_list[0]):.3f}, {np.max(self.rgb_data_list[0]):.3f}]")
        print(f"Flow shape: {self.flow_data_list[0].shape}")
        print(f"Flow range: [{np.min(self.flow_data_list[0]):.3f}, {np.max(self.flow_data_list[0]):.3f}]")
        print(f"Label shape: {self.labels.shape}")
        print(f"Label distribution: {np.sum(self.labels, axis=0)}")
        print("=== Data Generator Ready ===\n")
    
    def __len__(self):
        return self._batch_count
    
    def __getitem__(self, idx):
        try:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.n_samples)
            
            # Get batch data
            batch_rgb = np.stack(self.rgb_data_list[start_idx:end_idx])
            batch_flow = np.stack(self.flow_data_list[start_idx:end_idx])
            batch_labels = self.labels[start_idx:end_idx]
            
            if idx == 0:  # Print details for first batch
                print("\nFirst batch details:")
                print(f"RGB batch shape: {batch_rgb.shape}")
                print(f"RGB batch range: [{np.min(batch_rgb):.3f}, {np.max(batch_rgb):.3f}]")
                print(f"Flow batch shape: {batch_flow.shape}")
                print(f"Flow batch range: [{np.min(batch_flow):.3f}, {np.max(batch_flow):.3f}]")
                print(f"Labels batch shape: {batch_labels.shape}")
                print(f"Labels in batch: {np.argmax(batch_labels, axis=1)}")
                print("First batch data loaded successfully")
            
            return (batch_rgb, batch_flow), batch_labels
        except Exception as e:
            print(f"\nError in batch {idx}: {str(e)}")
            print(f"Error occurred at: {datetime.now().strftime('%H:%M:%S')}")
            raise
    
    def on_epoch_end(self):
        print(f"\n=== Training Epoch Completed ===")
        print(f"Processed {self._batch_count} batches")
        print(f"Total samples in epoch: {self.n_samples}")
        print("==============================\n")

def get_random_samples(data_dir, num_samples=10):
    """Get random samples from the data directory."""
    # Define class mapping
    class_mapping = {
        '2p0': 0,  # Two-point miss
        '2p1': 1,  # Two-point make
        '3p0': 2,  # Three-point miss
        '3p1': 3,  # Three-point make
        'ft0': 4,  # Free throw miss
        'ft1': 5,  # Free throw make
        'mp0': 6,  # Mid-range miss
        'mp1': 7   # Mid-range make
    }
    
    # Get all class directories that match our mapping and have data
    class_dirs = []
    for d in os.listdir(data_dir):
        if (os.path.isdir(os.path.join(data_dir, d)) 
            and not d.startswith('.')
            and d in class_mapping):
            # Check if directory has any RGB files
            if glob.glob(os.path.join(data_dir, d, '*_rgb.npz')):
                class_dirs.append(d)
    
    # Create a new mapping with only the classes that have data
    present_classes = sorted(class_dirs)  # Sort to ensure consistent ordering
    new_class_mapping = {class_name: idx for idx, class_name in enumerate(present_classes)}
    
    print("\nFound classes with data:")
    for class_name, idx in new_class_mapping.items():
        print(f"  {class_name}: index {idx}")
    
    # Get all RGB and flow files
    rgb_files = []
    flow_files = []
    labels = []
    
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        class_idx = new_class_mapping[class_dir]
        
        # Get all RGB files in this class directory
        class_rgb_files = glob.glob(os.path.join(class_path, '*_rgb.npz'))
        
        for rgb_file in class_rgb_files:
            # Get corresponding flow file
            flow_file = rgb_file.replace('_rgb.npz', '_flow.npz')
            if os.path.exists(flow_file):
                rgb_files.append(rgb_file)
                flow_files.append(flow_file)
                labels.append(class_idx)
    
    print(f"\nFound {len(rgb_files)} valid sample pairs across {len(class_dirs)} classes")
    print("Class distribution:")
    for class_name, idx in new_class_mapping.items():
        count = labels.count(idx)
        if count > 0:
            print(f"  {class_name}: {count} samples")
    
    # Randomly select samples
    if len(rgb_files) < num_samples:
        print(f"Warning: Only {len(rgb_files)} samples available, using all of them")
        num_samples = len(rgb_files)
    
    indices = random.sample(range(len(rgb_files)), num_samples)
    selected_rgb = [rgb_files[i] for i in indices]
    selected_flow = [flow_files[i] for i in indices]
    selected_labels = [labels[i] for i in indices]
    
    # Print selected samples
    print("\nSelected samples:")
    for i, (rgb, flow, label) in enumerate(zip(selected_rgb, selected_flow, selected_labels)):
        class_name = [k for k, v in new_class_mapping.items() if v == label][0]
        print(f"Sample {i+1}:")
        print(f"  RGB: {os.path.basename(rgb)}")
        print(f"  Flow: {os.path.basename(flow)}")
        print(f"  Class: {class_name} (index: {label})")
    
    return selected_rgb, selected_flow, selected_labels, len(present_classes)

def test_multiple_samples_training(data_dir, output_dir, num_samples=10):
    """Test training with multiple random samples."""
    start_time = time.time()
    print(f"\n=== Starting Process at {datetime.now().strftime('%H:%M:%S')} ===")
    
    # Configure TensorFlow for better stability
    tf.config.optimizer.set_jit(False)  # Disable XLA compilation for stability
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "constant_folding": True,
        "shape_optimization": True,
        "remapping": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "loop_optimization": True,
        "function_optimization": True,
        "debug_stripper": True,
        "disable_model_pruning": False,
        "scoped_allocator_optimization": True,
        "pin_to_host_optimization": True,
        "implementation_selector": True,
        "auto_mixed_precision": False  # Disable mixed precision for stability
    })

    print("\n=== Loading Random Samples ===")
    
    # Get random samples
    rgb_files, flow_files, labels, num_classes = get_random_samples(data_dir, num_samples)
    
    # Load and process all samples
    rgb_data_list = []
    flow_data_list = []
    
    print("\n=== Processing Samples ===")
    for i, (rgb_file, flow_file) in enumerate(zip(rgb_files, flow_files)):
        print(f"\nProcessing sample {i+1}/{len(rgb_files)}")
        print(f"RGB file: {os.path.basename(rgb_file)}")
        print(f"Flow file: {os.path.basename(flow_file)}")
        
        # Load data
        rgb_npz = np.load(rgb_file)
        flow_npz = np.load(flow_file)
        
        rgb_data = rgb_npz['data'] if 'data' in rgb_npz else rgb_npz['arr_0']
        flow_data = flow_npz['data'] if 'data' in flow_npz else flow_npz['arr_0']
        
        print(f"Original shapes - RGB: {rgb_data.shape}, Flow: {flow_data.shape}")
        print(f"Original ranges - RGB: [{rgb_data.min():.3f}, {rgb_data.max():.3f}], Flow: [{flow_data.min():.3f}, {flow_data.max():.3f}]")
        
        # Process data
        rgb_data = pad_sequence(rgb_data, SEQ_LENGTH)
        flow_data = pad_sequence(flow_data, SEQ_LENGTH)
        flow_data = normalize_flow(flow_data)
        
        print(f"Processed shapes - RGB: {rgb_data.shape}, Flow: {flow_data.shape}")
        print(f"Processed ranges - RGB: [{rgb_data.min():.3f}, {rgb_data.max():.3f}], Flow: [{flow_data.min():.3f}, {flow_data.max():.3f}]")
        
        rgb_data_list.append(rgb_data)
        flow_data_list.append(flow_data)
    
    # Convert labels to one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=num_classes)
    print("\nLabel distribution after one-hot encoding:")
    print(np.sum(labels_one_hot, axis=0))
    
    print("\n=== Creating Data Generator ===")
    # Create data generator
    train_gen = MultiSampleDataGenerator(rgb_data_list, flow_data_list, labels_one_hot)
    
    # Build and compile model
    rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)
    
    print("\n=== Building Model ===")
    print(f"RGB input shape: {rgb_shape}")
    print(f"Flow input shape: {flow_shape}")
    print(f"Number of classes: {num_classes}")
    
    model = build_two_stream_aca_net(rgb_shape, flow_shape, num_classes=num_classes)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Compile model with more stable settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Disable mixed precision
    tf.keras.mixed_precision.set_global_policy('float32')
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False  # Disable XLA compilation
    )
    
    print("\n=== Starting Model Training ===")
    print(f"Training start time: {datetime.now().strftime('%H:%M:%S')}")
    print("Using float32 precision for better stability")
    print("XLA compilation disabled for better compatibility")
    
    # Test a single forward pass before training
    print("\nTesting single forward pass...")
    test_batch = next(iter(train_gen))
    try:
        print("Running test forward pass...")
        test_output = model(test_batch[0], training=False)
        print("Test forward pass successful")
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output range: [{tf.reduce_min(test_output):.3f}, {tf.reduce_max(test_output):.3f}]")
    except Exception as e:
        print(f"Error in test forward pass: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        print("\nTrying to continue with training...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Train with more stable settings
        print("\nStarting actual training...")
        print("Training will run for 2 epochs")
        print("First batch might take longer due to initialization")
        
        # Add early stopping to prevent hanging
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Add reduce LR on plateau to help with training stability
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
        
        history = model.fit(
            train_gen,
            epochs=2,
            verbose=1,  # Use TensorFlow's progress bar
            callbacks=[
                TrainingProgressCallback(),
                early_stopping,
                reduce_lr,
                tf.keras.callbacks.TensorBoard(log_dir=output_dir),
                tf.keras.callbacks.CSVLogger(os.path.join(output_dir, 'training_log.csv')),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(output_dir, 'best_model.keras'),
                    monitor='accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                )
            ]
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model state...")
        model.save(os.path.join(output_dir, 'interrupted_model.keras'))
        raise
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print(f"Error occurred at: {datetime.now().strftime('%H:%M:%S')}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        print("\nAttempting to save model state...")
        try:
            model.save(os.path.join(output_dir, 'error_model.keras'))
            print("Model state saved despite error")
        except:
            print("Could not save model state")
        raise
    
    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Total process time: {total_time:.2f} seconds")
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
    parser = argparse.ArgumentParser(description="Test model training with multiple random samples")
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the processed data (with rgb and flow subdirectories)')
    parser.add_argument('--output_dir', type=str, default='test_output',
                      help='Directory to save the test model')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of random samples to use for training')
    args = parser.parse_args()
    
    # Verify input directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    test_multiple_samples_training(args.data_dir, args.output_dir, args.num_samples)

if __name__ == "__main__":
    main() 