#!/usr/bin/env python3
"""
two_stream_aca_net.py

Implementation of Two-Stream ACA-Net for basketball action recognition:
- Stream 1: RGB frames with ACA-Net (modified ResNet50 with LSTA and TSCI modules)
- Stream 2: Optical Flow with ACA-Net (same architecture, different weights)
- Fusion: Concatenation of features followed by FC layers
"""
import os
import sys
import numpy as np
import tensorflow as tf
import cv2 # Keep for frame extraction
import random # For augmentations
import argparse # <-- ADDED IMPORT
from multiprocessing import cpu_count # <-- ADDED IMPORT
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    GlobalAveragePooling2D, Concatenate, Reshape, Multiply,
    LayerNormalization, Permute, Attention, Add, TimeDistributed,
    InputSpec, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, to_categorical # Added Sequence
from sklearn.model_selection import train_test_split # For splitting paths
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix # For eval
from tqdm import tqdm # Keep for evaluation progress

# Configuration
SEQ_LENGTH = 50
IMG_SIZE = (120, 160) # height, width for Keras layers
IMG_SIZE_CV = (160, 120) # width, height for cv2.resize
NUM_CLASSES = 8
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
FLOW_DIR = '/kaggle/input/flow-rgb-data-basketball-51-50/flow_data/flow_data' # Updated Flow path
RGB_DIR = '/kaggle/input/flow-rgb-data-basketball-51-50/rgb_data/rgb_data' # Updated RGB path

# Helper: Ensure correct input shape for TimeDistributed ResNet
class CorrectShapeLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Reshape from (batch, time, h, w, c) to (batch*time, h, w, c)
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        h, w, c = IMG_SIZE[0], IMG_SIZE[1], tf.shape(inputs)[-1] # Use defined IMG_SIZE
        reshaped_inputs = tf.reshape(inputs, (batch_size * seq_len, h, w, c))
        return reshaped_inputs

    def compute_output_shape(self, input_shape):
        # input_shape = (batch, time, h, w, c)
        h, w, c = IMG_SIZE[0], IMG_SIZE[1], input_shape[-1]
        return (input_shape[0] * input_shape[1], h, w, c)

def LSTA_module(x, name_prefix="rgb"):
    """
    Long Short-Term Attention (LSTA) module for temporal attention
    
    Args:
        x: Input tensor, shape (batch, seq_len, features)
        name_prefix: Prefix for naming layers
        
    Returns:
        Tensor with temporal attention applied, shape (batch, seq_len, features)
    """
    batch_size, seq_len, features = tf.keras.backend.int_shape(x)
    if features is None: # Handle dynamic feature dimension after GAP
        features = tf.shape(x)[-1]

    # Temporal attention: Treat sequence steps as tokens
    # Use MultiHeadAttention for better performance
    q = Dense(features, name=f"{name_prefix}_lsta_q_dense")(x)
    k = Dense(features, name=f"{name_prefix}_lsta_k_dense")(x)
    v = Dense(features, name=f"{name_prefix}_lsta_v_dense")(x)

    # Add LayerNorm before attention
    q = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lsta_q_norm")(q)
    k = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lsta_k_norm")(k)
    v = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lsta_v_norm")(v)

    # Multi-head attention - assumes features is divisible by num_heads
    num_heads = 8 # Example, could be tuned
    # Ensure features is statically known or handle dynamically
    if isinstance(features, tf.Tensor):
         # Cannot directly compute head_dim if features is a Tensor
         # Option 1: Assume a fixed feature size (e.g., 2048 from ResNet)
         # Option 2: Pass feature size as argument or use simpler attention
         print("Warning: LSTA features is a Tensor, using simplified Attention.")
         attn_output = Attention(name=f"{name_prefix}_lsta_attention")([q, k, v]) # Simplified fallback
    elif features % num_heads != 0:
         print(f"Warning: LSTA features ({features}) not divisible by num_heads ({num_heads}). Using simplified Attention.")
         attn_output = Attention(name=f"{name_prefix}_lsta_attention")([q, k, v]) # Simplified fallback
    else:
        head_dim = features // num_heads
        attn_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_dim, name=f"{name_prefix}_lsta_mha"
        )
        attn_output = attn_layer(query=q, key=k, value=v)
    
    # Projection and residual connection
    attn_output = Dense(features, name=f"{name_prefix}_lsta_proj_dense")(attn_output)
    attn_output = Dropout(0.1, name=f"{name_prefix}_lsta_dropout")(attn_output) # Add dropout

    output = Add(name=f"{name_prefix}_lsta_add")([x, attn_output])
    output = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_lsta_out_norm")(output)
    
    return output

def TSCI_module(x, name_prefix="rgb"):
    """
    Temporal-Spatial-Channel Interaction (TSCI) module - Simplified version
    Applied *after* Global Average Pooling in this structure.
    Focuses on channel interactions across time.
    
    Args:
        x: Input tensor, shape (batch, seq_len, features)
        name_prefix: Prefix for naming layers
        
    Returns:
        Enhanced feature maps, shape (batch, seq_len, features)
    """
    batch_size, seq_len, features = tf.keras.backend.int_shape(x)
    if features is None:
         features = tf.shape(x)[-1]

    # Channel Attention across sequence
    # Average over time dimension
    channel_pool_temporal = tf.keras.layers.GlobalAveragePooling1D(name=f"{name_prefix}_tsci_temporal_avg")(x) # Use Keras layer

    # MLP for channel attention weights
    # Ensure features is statically known or handle dynamically
    if isinstance(features, tf.Tensor):
        # Cannot determine intermediate Dense size directly from Tensor shape
        # Using fixed intermediate size or alternative approach
        intermediate_dim = 256 # Example fixed size
        print(f"Warning: TSCI features is a Tensor, using fixed intermediate dim {intermediate_dim}.")
    else:
        intermediate_dim = features // 16

    channel_attn = Dense(intermediate_dim, activation='relu', name=f"{name_prefix}_tsci_channel_fc1")(channel_pool_temporal)
    channel_attn = Dense(features, activation='sigmoid', name=f"{name_prefix}_tsci_channel_fc2")(channel_attn)
    channel_attn = Reshape((1, features), name=f"{name_prefix}_tsci_channel_reshape")(channel_attn) # Reshape to (batch, 1, features)
    
    # Apply channel attention (broadcast over sequence length)
    x_channel = Multiply(name=f"{name_prefix}_tsci_channel_multiply")([x, channel_attn])
    
    # Simpler combination for now, focusing on channel enhancement
    # Residual connection might be better here
    output = Add(name=f"{name_prefix}_tsci_add")([x, x_channel])
    output = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_tsci_out_norm")(output)
    
    return output

def build_aca_net_stream(input_shape, name_prefix="rgb"):
    """
    Build a single ACA-Net stream using ResNet50 backbone.
    
    Args:
        input_shape: Shape of input tensor (seq_len, height, width, channels)
        name_prefix: Prefix for naming layers (e.g., "rgb" or "flow")
        
    Returns:
        Model representing a single stream feature extractor.
    """
    seq_len, h, w, c = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape, name=f"{name_prefix}_input")
    
    # Base Model: ResNet50 pre-trained on ImageNet
    # include_top=False to get features before the final classification layer
    # Ensure input_shape uses H, W for Keras
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(h, w, 3), pooling=None)

    # Handle Flow Input (2 channels) vs RGB (3 channels)
    if name_prefix == "flow" and c == 2:
        # Need to adapt the first layer of ResNet50 for 2 channels
        flow_input_tensor = Input(shape=(h, w, c), name="flow_input_tensor")
        flow_conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', name=f"{name_prefix}_conv1_custom")(flow_input_tensor)
        first_layer_model = Model(inputs=flow_input_tensor, outputs=flow_conv1, name=f"{name_prefix}_first_layer")
        x = TimeDistributed(first_layer_model, name=f"{name_prefix}_td_first_layer")(inputs)
    
        # Apply the rest of the ResNet layers (skipping the original conv1)
        # Find the layer after conv1 + bn + relu
        relu_layer_name = 'conv1_relu' # Standard ResNet50 layer name
        if relu_layer_name not in [l.name for l in base_model.layers]:
             print(f"Warning: Layer '{relu_layer_name}' not found in base model. Trying 'conv1_conv'.")
             # Fallback or inspect base_model.summary() if this fails
             relu_layer_name = 'conv1_conv' # Might vary

        resnet_layers_after_conv1 = Model(inputs=base_model.get_layer(relu_layer_name).output,
                                           outputs=base_model.output,
                                           name=f"{name_prefix}_resnet_body")
        # Set layers to trainable (fine-tuning the body)
        # for layer in resnet_layers_after_conv1.layers: layer.trainable = True
        x = TimeDistributed(resnet_layers_after_conv1, name=f"{name_prefix}_td_resnet_body")(x)

    else: # RGB Stream (c=3) - Use standard ResNet50
        if c != 3:
            raise ValueError(f"RGB stream expects 3 channels, but got {c}")
        rgb_base = ResNet50(include_top=False, weights='imagenet', input_shape=(h, w, 3), pooling='avg')
        # Fine-tuning: Set layers to trainable (or freeze some base layers)
        # for layer in rgb_base.layers: layer.trainable = True
        # rgb_base.trainable = True # Or control layer by layer

        x = TimeDistributed(rgb_base, name=f"{name_prefix}_td_resnet")(inputs)

    # If pooling wasn't done in base_model (e.g., for flow), apply it now
    if name_prefix == "flow":
         x = TimeDistributed(GlobalAveragePooling2D(), name=f"{name_prefix}_td_gap")(x)
    
    # Feature dimension is now (batch, seq_len, 2048) for both streams
    x = LSTA_module(x, name_prefix=name_prefix)
    x = TSCI_module(x, name_prefix=name_prefix)
    final_features = GlobalAveragePooling1D(name=f"{name_prefix}_global_avg_pool_time")(x)
    model = Model(inputs=inputs, outputs=final_features, name=f"{name_prefix}_stream")
    return model

def build_two_stream_aca_net(rgb_shape, flow_shape, num_classes=NUM_CLASSES):
    """
    Build the complete Two-Stream ACA-Net using ResNet50 backbones.
    
    Args:
        rgb_shape: Shape of RGB input (seq_len, height, width, 3)
        flow_shape: Shape of optical flow input (seq_len, height, width, 2)
        num_classes: Number of output classes
        
    Returns:
        Complete Two-Stream ACA-Net model
    """
    # Build RGB stream
    rgb_stream = build_aca_net_stream(rgb_shape, name_prefix="rgb")
    rgb_input = rgb_stream.input
    rgb_features = rgb_stream.output # Shape: (batch, 2048)
    
    # Build Flow stream
    flow_stream = build_aca_net_stream(flow_shape, name_prefix="flow")
    flow_input = flow_stream.input
    flow_features = flow_stream.output # Shape: (batch, 2048)
    
    # Concatenate features from both streams (2048 + 2048 = 4096)
    concat_features = Concatenate(name="fusion_concat")([rgb_features, flow_features]) # Shape: (batch, 4096)
    
    # Fully connected layers for classification as specified in the document
    x = Dense(1024, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), name="fusion_fc1")(concat_features)
    x = Dropout(0.5, name="fusion_dropout")(x) # Added dropout as specified
    outputs = Dense(num_classes, activation='softmax', name="fusion_output")(x) # Use num_classes
    
    # Create the final model with two inputs
    model = Model(inputs=[rgb_input, flow_input], outputs=outputs, name="two_stream_aca_net")
    
    return model

def normalize_flow(flow_sequence):
    """Normalize flow vectors. Flow values can vary widely.
       Simple approach: Clip and scale to [0, 1].
       More robust: Standardize (subtract mean, divide by std dev) per channel.
       Let's use clipping & scaling for now based on user plan.
    """
    # Clip extreme values (e.g., based on expected max displacement)
    # These bounds might need tuning based on dataset observation
    min_bound, max_bound = -20, 20
    flow_sequence = np.clip(flow_sequence, min_bound, max_bound)
    # Scale to [0, 1]
    normalized_flow = (flow_sequence - min_bound) / (max_bound - min_bound)
    return normalized_flow.astype(np.float32)

def augment_data(rgb_seq, flow_seq, img_h, img_w, augment=True):
    """Apply synchronized augmentations to RGB and Flow sequences."""
    if not augment:
        return rgb_seq, flow_seq

    # 1. Random Horizontal Flip
    if random.random() < 0.5:
        rgb_seq = rgb_seq[:, :, ::-1, :]  # Flip width axis
        flow_seq = flow_seq[:, :, ::-1, :]  # Flip width axis
        flow_seq[..., 0] *= -1             # Invert horizontal flow component

    # 2. Random Crop (example: crop to 85-95% of original size)
    # This requires careful implementation to ensure crop is identical
    # For simplicity, let's use resize augmentation (less common but simpler)
    # Or skip crop if implementation is complex
    # Example: Simple center crop (not random)
    # crop_h, crop_w = int(img_h * 0.9), int(img_w * 0.9)
    # start_h = (img_h - crop_h) // 2
    # start_w = (img_w - crop_w) // 2
    # rgb_seq = rgb_seq[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
    # flow_seq = flow_seq[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
    # Need resize back to original IMG_SIZE after crop

    # 3. Color Jitter (on RGB only)
    if random.random() < 0.5:
        if not isinstance(rgb_seq, tf.Tensor):
            rgb_seq_tensor = tf.convert_to_tensor(rgb_seq, dtype=tf.float32)
        else:
            rgb_seq_tensor = rgb_seq
        rgb_seq_tensor = tf.image.random_brightness(rgb_seq_tensor, max_delta=0.2)
        rgb_seq_tensor = tf.image.random_contrast(rgb_seq_tensor, lower=0.8, upper=1.2)
        rgb_seq_tensor = tf.image.random_saturation(rgb_seq_tensor, lower=0.8, upper=1.2)
        rgb_seq_tensor = tf.clip_by_value(rgb_seq_tensor, 0.0, 1.0)
        rgb_seq = rgb_seq_tensor.numpy()

    return rgb_seq, flow_seq

class DataGenerator(Sequence):
    """Generates batches of pre-computed RGB and Flow data."""
    def __init__(self, video_paths, labels, batch_size, num_classes,
                 seq_length, img_size, flow_dir, rgb_dir,
                 dataset_base_path, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.img_h, self.img_w = img_size
        self.flow_dir = flow_dir
        self.rgb_dir = rgb_dir
        self.dataset_base_path = dataset_base_path
        self.augment = augment
        self.indices = np.arange(len(self.video_paths))
        print(f"DataGenerator initialized with {len(self.video_paths)} videos")
        print(f"Flow directory: {self.flow_dir}")
        print(f"RGB directory: {self.rgb_dir}")
        if self.augment:
            self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data, skipping samples with missing RGB or Flow."""
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_video_paths = [self.video_paths[k] for k in batch_indices]
        batch_labels = [self.labels[k] for k in batch_indices]

        valid_X_rgb = []
        valid_X_flow = []
        valid_y = []
        skipped_count = 0

        for video_path, label in zip(batch_video_paths, batch_labels):
            try:
                # 1. Construct expected paths for BOTH precomputed files
                relative_path = os.path.relpath(video_path, self.dataset_base_path)
                base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                flow_filename = os.path.join(self.flow_dir,
                                             os.path.dirname(relative_path),
                                             f"{base_filename}_flow.npy")
                rgb_filename = os.path.join(self.rgb_dir,
                                            os.path.dirname(relative_path),
                                            f"{base_filename}_rgb.npy")

                # Debug: Print paths for first batch only
                if index == 0 and len(valid_X_rgb) == 0:
                    print(f"\nDebug - First sample in batch:")
                    print(f"Video path: {video_path}")
                    print(f"Flow file: {flow_filename}")
                    print(f"RGB file: {rgb_filename}")

                # 2. Check if BOTH files exist
                if not os.path.exists(flow_filename) or not os.path.exists(rgb_filename):
                    skipped_count += 1
                    if index == 0:  # Only print for first batch
                        print(f"Skipping sample - Missing files for {base_filename}")
                        if not os.path.exists(flow_filename):
                            print(f"Missing flow file: {flow_filename}")
                        if not os.path.exists(rgb_filename):
                            print(f"Missing RGB file: {rgb_filename}")
                    continue

                # 3. Load pre-computed RGB
                rgb_sequence = np.load(rgb_filename)
                flow_sequence = np.load(flow_filename)

                # Debug shapes for first batch
                if index == 0 and len(valid_X_rgb) == 0:
                    print(f"Loaded shapes - RGB: {rgb_sequence.shape}, Flow: {flow_sequence.shape}")

                # Check shapes
                expected_rgb_shape = (self.seq_length, self.img_h, self.img_w, 3)
                expected_flow_shape = (self.seq_length, self.img_h, self.img_w, 2)
                if rgb_sequence.shape != expected_rgb_shape or flow_sequence.shape != expected_flow_shape:
                    print(f"Warning: Shape mismatch for {base_filename}. Skipping.")
                    print(f"  RGB shape: {rgb_sequence.shape}, Expected: {expected_rgb_shape}")
                    print(f"  Flow shape: {flow_sequence.shape}, Expected: {expected_flow_shape}")
                    skipped_count += 1
                    continue

                # Process and append valid sample
                flow_sequence = normalize_flow(flow_sequence)
                rgb_sequence, flow_sequence = augment_data(rgb_sequence, flow_sequence,
                                                         self.img_h, self.img_w,
                                                         self.augment)

                valid_X_rgb.append(rgb_sequence)
                valid_X_flow.append(flow_sequence)
                valid_y.append(to_categorical(label, num_classes=self.num_classes))

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                skipped_count += 1
                continue
                
        # Print batch statistics for first batch
        if index == 0:
            print(f"\nBatch {index} statistics:")
            print(f"Total samples in batch: {len(batch_video_paths)}")
            print(f"Valid samples: {len(valid_X_rgb)}")
            print(f"Skipped samples: {skipped_count}")

        if not valid_y:
            rgb_shape = (0, self.seq_length, self.img_h, self.img_w, 3)
            flow_shape = (0, self.seq_length, self.img_h, self.img_w, 2)
            label_shape = (0, self.num_classes)
            return ((np.zeros(rgb_shape, dtype=np.float32), np.zeros(flow_shape, dtype=np.float32)), 
                    np.zeros(label_shape, dtype=np.float32))

        batch_X_rgb = np.array(valid_X_rgb, dtype=np.float32)
        batch_X_flow = np.array(valid_X_flow, dtype=np.float32)
        batch_y = np.array(valid_y, dtype=np.float32)

        return ((batch_X_rgb, batch_X_flow), batch_y)

    def on_epoch_end(self):
        """Shuffle indices after each epoch only for training."""
        if self.augment: # Only shuffle if it's the training generator
            np.random.shuffle(self.indices)

    @property
    def element_spec(self):
        # Use None for batch dimension to allow variable batch sizes
        rgb_spec = tf.TensorSpec(shape=(None, self.seq_length, self.img_h, self.img_w, 3), dtype=tf.float32)
        flow_spec = tf.TensorSpec(shape=(None, self.seq_length, self.img_h, self.img_w, 2), dtype=tf.float32)
        label_spec = tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32)
        # Return as tuple: ((feature_specs), label_spec)
        return ((rgb_spec, flow_spec), label_spec)

def train_two_stream_model(train_paths, train_labels, val_paths, val_labels,
                             flow_dir, rgb_dir, dataset_base_path, epochs=30):
    """Train the Two-Stream ACA-Net using data generators."""
    print(f"\n=== Training Configuration ===")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Flow directory: {flow_dir}")
    print(f"RGB directory: {rgb_dir}")
    print(f"Dataset base path: {dataset_base_path}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Image size: {IMG_SIZE}")
    print("=============================\n")

    # Verify directories exist
    if not os.path.exists(flow_dir):
        raise ValueError(f"Flow directory does not exist: {flow_dir}")
    if not os.path.exists(rgb_dir):
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")

    # Verify sample files exist
    sample_flow = os.path.join(flow_dir, "2p0", "2p0_v168_012503_x264_flow.npy")
    sample_rgb = os.path.join(rgb_dir, "2p0", "2p0_v168_012503_x264_rgb.npy")
    
    print("\nVerifying sample files:")
    print(f"Sample flow file exists: {os.path.exists(sample_flow)}")
    print(f"Sample RGB file exists: {os.path.exists(sample_rgb)}")
    
    if not os.path.exists(sample_flow) or not os.path.exists(sample_rgb):
        print("\nWARNING: Sample files not found. Please verify:")
        print("1. The precomputed data was properly uploaded to Kaggle")
        print("2. The directory structure matches the original dataset")
        print("3. The file naming convention is correct (_flow.npy and _rgb.npy)")
        print("\nExpected structure:")
        print(f"{flow_dir}/")
        print("  ├── 2p0/")
        print("  │   └── 2p0_v168_012503_x264_flow.npy")
        print("  └── ...")
        print(f"{rgb_dir}/")
        print("  ├── 2p0/")
        print("  │   └── 2p0_v168_012503_x264_rgb.npy")
        print("  └── ...")

    # Create data generators
    train_generator = DataGenerator(
        video_paths=train_paths, labels=train_labels, batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        flow_dir=flow_dir, rgb_dir=rgb_dir, dataset_base_path=dataset_base_path,
        augment=True
    )
    val_generator = DataGenerator(
        video_paths=val_paths, labels=val_labels, batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        flow_dir=flow_dir, rgb_dir=rgb_dir, dataset_base_path=dataset_base_path,
        augment=False
    )

    # Build and compile model
    rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)
    model = build_two_stream_aca_net(rgb_shape, flow_shape, NUM_CLASSES)
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    print("\n=== Model Architecture ===")
    model.summary()
    print("=========================\n")
    
    # Calculate steps
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    print(f"\n=== Training Parameters ===")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Total epochs: {epochs}")
    print("===========================\n")

    # Callbacks with debug prints
    class DebugCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nStarting epoch {epoch + 1}/{epochs}")
            print("Loading training data...")
            
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nCompleted epoch {epoch + 1}/{epochs}")
            print(f"Training loss: {logs.get('loss'):.4f}")
            print(f"Training accuracy: {logs.get('accuracy'):.4f}")
            print(f"Validation loss: {logs.get('val_loss'):.4f}")
            print(f"Validation accuracy: {logs.get('val_accuracy'):.4f}")

    checkpoint = ModelCheckpoint(
        "two_stream_aca_net_best.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Add debug callback
    debug_callback = DebugCallback()
    
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, reduce_lr, debug_callback],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, test_paths, test_labels, flow_dir, rgb_dir, dataset_base_path): # Added rgb_dir
    """
    Evaluate the trained model using a data generator for the test set.
    """
    print(f"Evaluating on {len(test_paths)} test samples...")

    test_generator = DataGenerator(
        video_paths=test_paths, labels=test_labels, batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        flow_dir=flow_dir, rgb_dir=rgb_dir, dataset_base_path=dataset_base_path, # Pass rgb_dir
        augment=False # No augmentation for testing
    )

    # Predict using the generator
    # Use tqdm for progress bar
    y_pred_prob = model.predict(test_generator,
                              steps=len(test_generator),
                              verbose=1
                              )

    # Ensure we only evaluate on the actual number of test samples
    num_test_samples = len(test_paths)
    y_pred_prob = y_pred_prob[:num_test_samples]
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Prepare true labels (convert from integers if needed)
    y_true = np.array(test_labels)
    y_true_cat = to_categorical(y_true, num_classes=NUM_CLASSES)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Try to calculate AUC
    try:
        # Use one-hot encoded true labels for AUC
        metrics['auc'] = roc_auc_score(y_true_cat, y_pred_prob, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Could not calculate AUC: {e}", file=sys.stderr)
        metrics['auc'] = 'N/A'
    
    return metrics

def get_video_paths_and_labels(dataset_path):
    """Collect all .mp4 video paths and numeric labels based on subfolder names."""
    print(f"Reading data from: {dataset_path}")
    label_dirs = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])
    if not label_dirs:
         raise ValueError(f"No subdirectories found in {dataset_path}. Check dataset structure.")

    label2id = {label: idx for idx, label in enumerate(label_dirs)}
    id2label = {idx: label for label, idx in label2id.items()}

    video_paths, labels = [], []
    for label in label_dirs:
        folder = os.path.join(dataset_path, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.mp4'):
                video_paths.append(os.path.join(folder, fname))
                labels.append(label2id[label])

    if not video_paths:
         raise ValueError(f"No .mp4 files found in subdirectories of {dataset_path}.")

    print(f"Found {len(video_paths)} videos across {len(label_dirs)} classes.")
    return video_paths, labels, label2id

def main(args):
    """Main function modified to use generators and precomputed flow."""
    print("Two-Stream ACA-Net Training Pipeline")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Flow Data Path: {args.flow_path}")
    print(f"RGB Data Path: {args.rgb_path}") # Added print for RGB path
    print(f"Epochs: {args.epochs}")

    # Check if dataset and flow directories exist
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset directory not found at: {args.dataset_path}", file=sys.stderr)
        return
    if not os.path.isdir(args.flow_path):
        print(f"Error: Flow data directory not found at: {args.flow_path}", file=sys.stderr)
        print("Please run precompute_flow.py first.")
        return
    if not os.path.isdir(args.rgb_path): # Check for RGB path
        print(f"Error: Precomputed RGB data directory not found at: {args.rgb_path}", file=sys.stderr)
        print("Please run precompute_rgb.py first or check the path.")
        return
    
    # Get video paths and integer labels
    try:
        video_paths, labels, label2id = get_video_paths_and_labels(args.dataset_path)
    except ValueError as e:
         print(f"Error reading dataset: {e}", file=sys.stderr)
         return

    # --- Added: Select 50% of the dataset ---
    num_total_videos = len(video_paths)
    indices = list(range(num_total_videos))
    SEED = 42 # Use the same fixed seed as in precompute_flow.py
    random.seed(SEED) # Set the random seed
    random.shuffle(indices) # Shuffle the indices (now deterministic)
    sample_indices = indices[:int(num_total_videos * 0.50)]

    video_paths_subset = [video_paths[i] for i in sample_indices]
    labels_subset = [labels[i] for i in sample_indices]
    print(f"Using a 50% subset: {len(video_paths_subset)} videos out of {num_total_videos} total for train/val/test split.")
    # --- End Added Section ---

    # Train/Validation/Test split on the *subset* of paths and labels
    print("Splitting data subset into training, validation, and test sets...")
    # First split into train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        video_paths_subset, labels_subset, test_size=0.2, stratify=labels_subset, random_state=42 # Use subset
    )
    # Then split train+val into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.2, # 20% of original train_val -> 16% of total subset
        stratify=train_val_labels, random_state=42
    )
    print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}, Test samples: {len(test_paths)}")
    
    # Train model using generators, passing rgb_dir
    print("Starting model training...")
    model, history = train_two_stream_model(train_paths, train_labels, val_paths, val_labels,
                                          flow_dir=args.flow_path,
                                          rgb_dir=args.rgb_path, # Pass RGB path
                                          dataset_base_path=args.dataset_path,
                                          epochs=args.epochs)
    
    # Evaluate model using generator, passing rgb_dir
    print("Starting model evaluation on test set...")
    best_model_path = "two_stream_aca_net_best.keras"
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for evaluation.")
        try:
             eval_model = tf.keras.models.load_model(best_model_path)
        except Exception as e: 
             print(f"Warning: Could not load best model...", file=sys.stderr)
             eval_model = model # Fallback
    else:
        print("Warning: Best model checkpoint not found...", file=sys.stderr)
        eval_model = model # Fallback
        
    metrics = evaluate_model(eval_model, test_paths, test_labels,
                             flow_dir=args.flow_path,
                             rgb_dir=args.rgb_path, # Pass RGB path
                             dataset_base_path=args.dataset_path)
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"AUC:       {metrics['auc']}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save the final (best) model - already saved by ModelCheckpoint
    print(f"\nBest model saved during training to: {best_model_path}")

    # Optional: Plot training history
    # Implement plot_history function if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Stream ACA-Net on Basketball-51.")
    parser.add_argument('--dataset_path', type=str, default='/kaggle/input/basketball-51/Basketball-51',
                        help='Path to the root directory of the Basketball-51 video dataset.')
    parser.add_argument('--flow_path', type=str, default=FLOW_DIR,
                        help='Path to the directory containing precomputed flow .npy files.')
    parser.add_argument('--rgb_path', type=str, default=RGB_DIR,
                        help='Path to the directory containing precomputed RGB .npy files.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs.')
    # Add other arguments if needed (e.g., batch size, learning rate)

    args = parser.parse_args()
    main(args) 