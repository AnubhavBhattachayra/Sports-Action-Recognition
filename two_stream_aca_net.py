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
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    GlobalAveragePooling2D, Concatenate, Reshape, Multiply,
    LayerNormalization, Permute, Attention, Add, TimeDistributed,
    InputSpec
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
# FLOW_DIR = './flow_data' # Old default path for precomputed flow
FLOW_DIR = '/kaggle/input/flow-data-50-basketball-51/flow_data' # New Kaggle dataset default path

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
    channel_pool_temporal = tf.reduce_mean(x, axis=1) # Shape: (batch, features)

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

def extract_rgb_frames(video_path, seq_length, img_size_cv):
    """Extracts, resizes, samples/pads RGB frames ONLY."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}", file=sys.stderr)
        # Return black frames as fallback
        return np.zeros((seq_length, img_size_cv[1], img_size_cv[0], 3), dtype=np.float32)

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size_cv)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        print(f"Warning: No frames extracted from {video_path}", file=sys.stderr)
        return np.zeros((seq_length, img_size_cv[1], img_size_cv[0], 3), dtype=np.float32)

    # Sample or pad frames
    if len(all_frames) >= seq_length:
        idxs = np.linspace(0, len(all_frames) - 1, seq_length).astype(int)
        frames = [all_frames[i] for i in idxs]
    else:
        frames = all_frames + [all_frames[-1]] * (seq_length - len(all_frames))

    # Normalize RGB frames to [0, 1]
    rgb_sequence = np.array(frames, dtype=np.float32) / 255.0
    return rgb_sequence

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
        rgb_seq = tf.image.random_brightness(rgb_seq, max_delta=0.2)
        rgb_seq = tf.image.random_contrast(rgb_seq, lower=0.8, upper=1.2)
        rgb_seq = tf.image.random_saturation(rgb_seq, lower=0.8, upper=1.2)
        # Clip values to ensure they remain in [0, 1]
        rgb_seq = tf.clip_by_value(rgb_seq, 0.0, 1.0)
        # Convert back to numpy if needed downstream, depends on generator structure
        # rgb_seq = rgb_seq.numpy()

    return rgb_seq, flow_seq

class DataGenerator(Sequence):
    """Generates batches of RGB and pre-computed Flow data."""
    def __init__(self, video_paths, labels, batch_size, num_classes,
                 seq_length, img_size, img_size_cv, flow_dir,
                 dataset_base_path, augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.img_h, self.img_w = img_size # Keras format (h, w)
        self.img_size_cv = img_size_cv # OpenCV format (w, h)
        self.flow_dir = flow_dir
        self.dataset_base_path = dataset_base_path
        self.augment = augment
        self.indices = np.arange(len(self.video_paths))
        self.on_epoch_end() # Shuffle indices initially

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Get video paths and labels for this batch
        batch_video_paths = [self.video_paths[k] for k in batch_indices]
        batch_labels = [self.labels[k] for k in batch_indices]

        # Initialize batch arrays
        # Use Keras image format (h, w)
        batch_X_rgb = np.zeros((self.batch_size, self.seq_length, self.img_h, self.img_w, 3), dtype=np.float32)
        batch_X_flow = np.zeros((self.batch_size, self.seq_length, self.img_h, self.img_w, 2), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)

        # Generate data for each item in the batch
        for i, video_path in enumerate(batch_video_paths):
            try:
                # 1. Extract RGB frames
                rgb_sequence = extract_rgb_frames(video_path, self.seq_length, self.img_size_cv)

                # 2. Load pre-computed flow
                relative_path = os.path.relpath(video_path, self.dataset_base_path)
                base_filename = os.path.splitext(os.path.basename(relative_path))[0]
                flow_filename = os.path.join(self.flow_dir,
                                             os.path.dirname(relative_path),
                                             f"{base_filename}_flow.npy")

                if os.path.exists(flow_filename):
                    flow_sequence = np.load(flow_filename)
                    # Ensure flow shape matches expected (might be saved with cv size)
                    if flow_sequence.shape[1:3] != (self.img_h, self.img_w):
                         # Example resize - might need adjustment based on how flow was saved
                         flow_sequence_resized = np.zeros((self.seq_length, self.img_h, self.img_w, 2), dtype=np.float32)
                         for frame_idx in range(self.seq_length):
                              # Use INTER_LINEAR for flow
                              flow_sequence_resized[frame_idx] = cv2.resize(flow_sequence[frame_idx],
                                                                           (self.img_w, self.img_h),
                                                                           interpolation=cv2.INTER_LINEAR)
                         flow_sequence = flow_sequence_resized

                else:
                    print(f"Warning: Flow file not found: {flow_filename}. Using zeros.", file=sys.stderr)
                    flow_sequence = np.zeros((self.seq_length, self.img_h, self.img_w, 2), dtype=np.float32)

                # 3. Normalize Flow
                flow_sequence = normalize_flow(flow_sequence)

                # 4. Data Augmentation (synchronized)
                rgb_sequence, flow_sequence = augment_data(rgb_sequence, flow_sequence,
                                                         self.img_h, self.img_w,
                                                         self.augment)

                # 5. Assign to batch arrays
                batch_X_rgb[i,] = rgb_sequence
                batch_X_flow[i,] = flow_sequence
                batch_y[i,] = to_categorical(batch_labels[i], num_classes=self.num_classes)

            except Exception as e:
                print(f"Error generating data for {video_path}: {e}", file=sys.stderr)
                # Keep zero arrays for this sample

        return [batch_X_rgb, batch_X_flow], batch_y

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        np.random.shuffle(self.indices)

def train_two_stream_model(train_paths, train_labels, val_paths, val_labels,
                             flow_dir, dataset_base_path, epochs=30):
    """
    Train the Two-Stream ACA-Net using data generators.
    """
    print(f"Training with {len(train_paths)} samples, validating with {len(val_paths)} samples.")

    # Create data generators
    train_generator = DataGenerator(
        video_paths=train_paths, labels=train_labels, batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        img_size_cv=IMG_SIZE_CV, flow_dir=flow_dir, dataset_base_path=dataset_base_path,
        augment=True # Enable augmentation for training
    )
    val_generator = DataGenerator(
        video_paths=val_paths, labels=val_labels, batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        img_size_cv=IMG_SIZE_CV, flow_dir=flow_dir, dataset_base_path=dataset_base_path,
        augment=False # No augmentation for validation
    )

    # Define model input shapes
    rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
    flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)

    # Build the model
    model = build_two_stream_aca_net(rgb_shape, flow_shape, NUM_CLASSES)

    # Compile the model (Optimizer and Loss from Step 6)
    optimizer = Adam(learning_rate=0.0001)
    loss = 'categorical_crossentropy'
    # Add F1 Score metric if needed (requires definition like in original main.py)
    metrics = ['accuracy'] # Add F1ScoreMetric(...) if available

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Model Compiled.")
    model.summary()

    # Define callbacks (Ensure F1 score monitoring if used)
    checkpoint_path = "two_stream_aca_net_best.h5" # Changed name
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy', # Or val_f1_score if metric is added
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False # Save entire model
    )
    early_stopping = EarlyStopping(
        monitor='val_accuracy', # Or val_f1_score
        patience=10, # Increased patience slightly
        mode='max',
        verbose=1,
        restore_best_weights=True # Restore best weights at the end
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5, # Reduce LR if val_loss plateaus for 5 epochs
        min_lr=1e-6,
        verbose=1
    )

    # Calculate steps per epoch
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Train the model using the generator
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        workers=max(1, cpu_count() // 2), # Use multiple workers for generator
        use_multiprocessing=True
    )

    # Load best weights if not restored by EarlyStopping
    # model.load_weights(checkpoint_path) # Optional: Explicitly load best if needed

    return model, history

def evaluate_model(model, test_paths, test_labels, flow_dir, dataset_base_path):
    """
    Evaluate the trained model using a data generator for the test set.
    """
    print(f"Evaluating on {len(test_paths)} test samples...")

    test_generator = DataGenerator(
        video_paths=test_paths, labels=test_labels, batch_size=BATCH_SIZE, # Use same batch size
        num_classes=NUM_CLASSES, seq_length=SEQ_LENGTH, img_size=IMG_SIZE,
        img_size_cv=IMG_SIZE_CV, flow_dir=flow_dir, dataset_base_path=dataset_base_path,
        augment=False # No augmentation for testing
    )

    # Predict using the generator
    # Use tqdm for progress bar
    y_pred_prob = model.predict(test_generator,
                              steps=len(test_generator),
                              workers=max(1, cpu_count() // 2),
                              use_multiprocessing=True,
                              verbose=1)

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
        metrics['auc'] = roc_auc_score(y_true_cat, y_pred_prob, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
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
    print(f"Epochs: {args.epochs}")

    # Check if dataset and flow directories exist
    if not os.path.isdir(args.dataset_path):
        print(f"Error: Dataset directory not found at: {args.dataset_path}", file=sys.stderr)
        return
    if not os.path.isdir(args.flow_path):
        print(f"Error: Flow data directory not found at: {args.flow_path}", file=sys.stderr)
        print("Please run precompute_flow.py first.")
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

    # Train model using generators
    print("Starting model training...")
    model, history = train_two_stream_model(train_paths, train_labels, val_paths, val_labels,
                                          flow_dir=args.flow_path,
                                          dataset_base_path=args.dataset_path,
                                          epochs=args.epochs)

    # Evaluate model using generator
    print("Starting model evaluation on test set...")
    metrics = evaluate_model(model, test_paths, test_labels,
                             flow_dir=args.flow_path,
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
    print(f"\nBest model saved during training to: two_stream_aca_net_best.h5")

    # Optional: Plot training history
    # Implement plot_history function if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Stream ACA-Net on Basketball-51.")
    parser.add_argument('--dataset_path', type=str, default='/kaggle/input/basketball-51/Basketball-51',
                        help='Path to the root directory of the Basketball-51 video dataset.')
    parser.add_argument('--flow_path', type=str, default=FLOW_DIR, # Use the updated FLOW_DIR constant
                        help='Path to the directory containing precomputed flow .npy files.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs.')
    # Add other arguments if needed (e.g., batch size, learning rate)

    args = parser.parse_args()
    main(args) 