#!/usr/bin/env python3
"""
main.py

Consolidated script for Basketball-51 action recognition training pipeline.
Run this file to execute the complete training process on your local machine.
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization, Add, GlobalAveragePooling2D, Attention, Layer, MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Reshape, Embedding, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# Define F1 Score metric for multi-class classification
class F1ScoreMetric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', num_classes=8, **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', initializer='zeros', shape=(num_classes,))
        self.false_positives = self.add_weight(name='fp', initializer='zeros', shape=(num_classes,))
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', shape=(num_classes,))
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Convert to one-hot for per-class metrics
        y_true_one_hot = tf.one_hot(y_true, self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred, self.num_classes)
        
        # Calculate true positives, false positives, false negatives
        true_pos = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        false_pos = tf.reduce_sum(y_pred_one_hot * (1 - y_true_one_hot), axis=0)
        false_neg = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)
        
        # Update running variables
        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)
        self.false_negatives.assign_add(false_neg)
        
    def result(self):
        # Calculate F1 score
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        
        # Average F1 score across all classes (macro averaging)
        return tf.reduce_mean(f1)
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

# Reduce TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")

# Enable mixed precision training for better memory efficiency
set_global_policy('mixed_float16')
print("Mixed precision training enabled")

# Configuration
DATASET_PATH = r'/kaggle/input/basketball-51/Basketball-51'
SEQ_LENGTH = 30            # number of frames per clip
IMG_SIZE = (96, 96)        # reduced from 112x112 to 96x96
BATCH_SIZE = 4             # reduced from 8 to 4
PATCH_SIZE = 16            # for ViT
EMBED_DIM = 128            # embedding dimension for ViT
NUM_HEADS = 4              # number of attention heads
NUM_TRANSFORMER_LAYERS = 4  # number of transformer blocks

def print_section(title):
    """Print a formatted section title for better console readability."""
    line = "-" * 80
    print(f"\n{line}\n{title}\n{line}")

def get_video_paths_and_labels(dataset_path):
    """Collect all .mp4 video paths and numeric labels based on subfolder names."""
    label_dirs = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])
    label2id = {label: idx for idx, label in enumerate(label_dirs)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    video_paths, labels = [], []
    for label in label_dirs:
        folder = os.path.join(dataset_path, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.mp4'):
                video_paths.append(os.path.join(folder, fname))
                labels.append(label2id[label])
                
    # Create a summary of the dataset
    class_counts = {}
    for label in labels:
        class_name = id2label[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("Dataset summary:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} videos")
        
    return video_paths, labels, label2id

def extract_frames(video_path, seq_length, img_size):
    """Read a video, sample or pad to `seq_length`, resize frames, and normalize."""
    frames = []
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
        cap.release()
        
        if len(all_frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Sample or pad frames to match the required sequence length
        if len(all_frames) >= seq_length:
            # Sample frames evenly
            idxs = np.linspace(0, len(all_frames) - 1, seq_length).astype(int)
            frames = [all_frames[i] for i in idxs]
        else:
            # Pad with the last frame
            frames = all_frames + [all_frames[-1]] * (seq_length - len(all_frames))
        
        frames = np.array(frames, dtype='float32') / 255.0
        
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        # Return a blank sequence if there's an error
        frames = np.zeros((seq_length, img_size[1], img_size[0], 3), dtype='float32')
        
    return frames

def compute_optical_flow_sequence(frames):
    """Given an array of RGB frames, compute Farneback optical flow for each consecutive pair."""
    try:
        import cv2
        flows = []
        prev_gray = cv2.cvtColor((frames[0] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        
        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor((frames[i] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                               None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
            prev_gray = curr_gray
            
        # Duplicate first flow to match sequence length
        flows.insert(0, flows[0])
        flows = np.stack(flows, axis=0)
        flows = flows.astype('float32')
        
    except Exception as e:
        print(f"Error computing optical flow: {str(e)}")
        # Return zero flows if there's an error
        flows = np.zeros((len(frames), frames.shape[1], frames.shape[2], 2), dtype='float32')
        
    return flows

def prepare_data(paths, labels, seq_length, img_size, num_classes):
    """Extract RGB and flow sequences for each video, return fused X and one-hot y."""
    X_rgb, X_flow, y = [], [], []
    
    for vp, lbl in tqdm(zip(paths, labels), total=len(paths), desc='Preparing data'):
        try:
            frames = extract_frames(vp, seq_length, img_size)
            flow = compute_optical_flow_sequence(frames)
            X_rgb.append(frames)
            X_flow.append(flow)
            y.append(lbl)
        except Exception as e:
            print(f"Error processing {vp}: {str(e)}")
    
    X_rgb = np.array(X_rgb)
    X_flow = np.array(X_flow)
    
    # Fuse modalities along channel axis: (B, T, H, W, 3+2)
    X = np.concatenate([X_rgb, X_flow], axis=-1)
    y = to_categorical(y, num_classes)
    
    return X, y

def create_data_generator(paths, labels, seq_length, img_size, num_classes, batch_size):
    """Create a generator that yields batches of data to reduce memory usage.
    
    This generator processes videos in small batches instead of loading all into memory at once.
    """
    num_samples = len(paths)
    indices = np.arange(num_samples)
    
    # Convert labels to one-hot encoding
    y_categorical = to_categorical(labels, num_classes)
    
    while True:
        # Shuffle indices each epoch
        np.random.shuffle(indices)
        
        # Process in batches
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Initialize batch arrays
            batch_X_rgb = []
            batch_X_flow = []
            batch_y = []
            
            # Process each video in the batch
            for i in batch_indices:
                try:
                    frames = extract_frames(paths[i], seq_length, img_size)
                    flow = compute_optical_flow_sequence(frames)
                    batch_X_rgb.append(frames)
                    batch_X_flow.append(flow)
                    batch_y.append(y_categorical[i])
                except Exception as e:
                    print(f"Error processing {paths[i]}: {str(e)}")
                    # Skip this sample
                    continue
            
            if len(batch_X_rgb) == 0:
                # Skip empty batches
                continue
                
            # Convert to numpy arrays
            batch_X_rgb = np.array(batch_X_rgb)
            batch_X_flow = np.array(batch_X_flow)
            
            # Fuse modalities along channel axis: (B, T, H, W, 3+2)
            batch_X = np.concatenate([batch_X_rgb, batch_X_flow], axis=-1)
            batch_y = np.array(batch_y)
            
            yield batch_X, batch_y

def process_test_batch(paths, labels, seq_length, img_size, num_classes, batch_size=8):
    """Process test data in small batches to avoid memory issues."""
    num_samples = len(paths)
    X_batches = []
    y_batches = []
    
    # Convert labels to one-hot encoding
    y_categorical = to_categorical(labels, num_classes)
    
    # Process in batches
    for start_idx in tqdm(range(0, num_samples, batch_size), desc="Processing test data"):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_paths = paths[start_idx:end_idx]
        batch_labels = y_categorical[start_idx:end_idx]
        
        # Initialize batch arrays
        batch_X_rgb = []
        batch_X_flow = []
        batch_y = []
        
        # Process each video in the batch
        for i, (vp, lbl) in enumerate(zip(batch_paths, batch_labels)):
            try:
                frames = extract_frames(vp, seq_length, img_size)
                flow = compute_optical_flow_sequence(frames)
                batch_X_rgb.append(frames)
                batch_X_flow.append(flow)
                batch_y.append(lbl)
            except Exception as e:
                print(f"Error processing {vp}: {str(e)}")
                # Skip this sample
                continue
        
        if len(batch_X_rgb) == 0:
            # Skip empty batches
            continue
            
        # Convert to numpy arrays
        batch_X_rgb = np.array(batch_X_rgb)
        batch_X_flow = np.array(batch_X_flow)
        
        # Fuse modalities along channel axis: (B, T, H, W, 3+2)
        batch_X = np.concatenate([batch_X_rgb, batch_X_flow], axis=-1)
        batch_y = np.array(batch_y)
        
        X_batches.append(batch_X)
        y_batches.append(batch_y)
    
    # Concatenate all batches (this is smaller than the original dataset)
    X_test = np.concatenate(X_batches, axis=0) if X_batches else np.array([])
    y_test = np.concatenate(y_batches, axis=0) if y_batches else np.array([])
    
    return X_test, y_test

class PatchEncoder(Layer):
    """
    Breaks image into patches, linearly embeds each patch, adds positional embeddings.
    Used in the Vision Transformer architecture.
    """
    def __init__(self, num_patches, embedding_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.projection = Dense(embedding_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=embedding_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded = self.projection(patch)
        encoded = embedded + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'embedding_dim': self.embedding_dim,
        })
        return config

def mlp(x, hidden_units, dropout_rate):
    """MLP block for Transformer."""
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def transformer_encoder_block(inputs, num_heads, embed_dim, ff_dim, dropout_rate=0.1):
    """Transformer encoder block with multi-head attention."""
    # Normalization and Multi-head Attention
    attention_output = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads
    )(attention_output, attention_output)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = Add()([inputs, attention_output])
    
    # Feed Forward Network
    ffn_output = LayerNormalization(epsilon=1e-6)(out1)
    ffn_output = mlp(ffn_output, [ff_dim, embed_dim], dropout_rate)
    return Add()([out1, ffn_output])

def cross_attention_block(queries, keys_values, num_heads, embed_dim, dropout_rate=0.1):
    """Cross-attention block to fuse information from two modalities."""
    # Normalization and Cross-Attention
    norm_q = LayerNormalization(epsilon=1e-6)(queries)
    norm_kv = LayerNormalization(epsilon=1e-6)(keys_values)
    
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads
    )(query=norm_q, key=norm_kv, value=norm_kv)
    
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = Add()([queries, attention_output])
    
    # Feed Forward Network
    ffn_output = LayerNormalization(epsilon=1e-6)(out1)
    ffn_output = mlp(ffn_output, [embed_dim * 2, embed_dim], dropout_rate)
    return Add()([out1, ffn_output])

class ChannelSlicer(Layer):
    """Custom layer to slice input tensor along the channel dimension."""
    def __init__(self, start, size, **kwargs):
        super(ChannelSlicer, self).__init__(**kwargs)
        self.start = start
        self.size = size
        
    def call(self, inputs):
        return inputs[:, :, :, :, self.start:self.start+self.size]
    
    def get_config(self):
        config = super(ChannelSlicer, self).get_config()
        config.update({
            'start': self.start,
            'size': self.size
        })
        return config

class PatchExtractor(Layer):
    """Custom layer to extract patches from images."""
    def __init__(self, patch_size, **kwargs):
        super(PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size
        
    def call(self, inputs):
        # inputs shape: [batch_size, height, width, channels]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return patches
    
    def get_config(self):
        config = super(PatchExtractor, self).get_config()
        config.update({
            'patch_size': self.patch_size
        })
        return config

class FrameSelector(Layer):
    """Custom layer to select a specific frame from a sequence."""
    def __init__(self, frame_index, **kwargs):
        super(FrameSelector, self).__init__(**kwargs)
        self.frame_index = frame_index
        
    def call(self, inputs):
        # inputs shape: [batch_size, seq_len, height, width, channels]
        return inputs[:, self.frame_index]
    
    def get_config(self):
        config = super(FrameSelector, self).get_config()
        config.update({
            'frame_index': self.frame_index
        })
        return config

def build_model(input_shape, num_classes, class_weights=None):
    """
    Build a dual-stream Vision Transformer (ViT) model with cross-modal fusion 
    for basketball action recognition using RGB and optical flow modalities.
    """
    # Input layer for RGB+Flow (5 channels)
    inp = Input(shape=input_shape)
    
    # Separate RGB and optical flow channels
    rgb_frames = ChannelSlicer(0, 3)(inp)
    flow_frames = ChannelSlicer(3, 2)(inp)
    
    # Calculate number of patches
    input_h, input_w = input_shape[1], input_shape[2]
    num_patches = (input_h // PATCH_SIZE) * (input_w // PATCH_SIZE)
    patch_dim = PATCH_SIZE * PATCH_SIZE
    
    # Process each frame in the sequence
    sequence_length = input_shape[0]
    
    # RGB Stream - Extract patches from each frame
    rgb_patches_list = []
    flow_patches_list = []
    
    for i in range(sequence_length):
        # Extract frame i from RGB sequence
        rgb_frame = FrameSelector(i)(rgb_frames)
        
        # Extract patches
        rgb_patch = PatchExtractor(PATCH_SIZE)(rgb_frame)
        # Reshape to [batch_size, num_patches, patch_dim * 3]
        rgb_patch_flat = Reshape((-1, patch_dim * 3))(rgb_patch)
        rgb_patches_list.append(rgb_patch_flat)
        
        # Extract frame i from flow sequence
        flow_frame = FrameSelector(i)(flow_frames)
        
        # Extract patches
        flow_patch = PatchExtractor(PATCH_SIZE)(flow_frame)
        # Reshape to [batch_size, num_patches, patch_dim * 2]
        flow_patch_flat = Reshape((-1, patch_dim * 2))(flow_patch)
        flow_patches_list.append(flow_patch_flat)
    
    # Concatenate all frames
    rgb_patches = Concatenate(axis=1)(rgb_patches_list)
    flow_patches = Concatenate(axis=1)(flow_patches_list)
    
    # Add batch normalization for more stable training
    rgb_patches = BatchNormalization()(rgb_patches)
    flow_patches = BatchNormalization()(flow_patches)
    
    # Encode patches
    rgb_encoded_patches = PatchEncoder(num_patches * sequence_length, EMBED_DIM)(rgb_patches)
    flow_encoded_patches = PatchEncoder(num_patches * sequence_length, EMBED_DIM)(flow_patches)
    
    # Apply transformer encoder blocks to each modality
    rgb_x = rgb_encoded_patches
    for _ in range(NUM_TRANSFORMER_LAYERS // 2):
        rgb_x = transformer_encoder_block(rgb_x, NUM_HEADS, EMBED_DIM, EMBED_DIM * 2)
    
    flow_x = flow_encoded_patches
    for _ in range(NUM_TRANSFORMER_LAYERS // 2):
        flow_x = transformer_encoder_block(flow_x, NUM_HEADS, EMBED_DIM, EMBED_DIM * 2)
    
    # Cross-modal fusion (RGB as query, flow as key/value)
    fused_features = cross_attention_block(rgb_x, flow_x, NUM_HEADS, EMBED_DIM)
    
    # Apply more transformer layers to fused features
    for _ in range(NUM_TRANSFORMER_LAYERS // 2):
        fused_features = transformer_encoder_block(fused_features, NUM_HEADS, EMBED_DIM, EMBED_DIM * 2)
    
    # Global pooling
    pooled_output = GlobalAveragePooling1D()(fused_features)
    
    # Fixed dimension approach for LSTM - ensure pooled_output has proper dimensionality
    # Using Dense to map to fixed-size representation for LSTM input
    x = Dense(256, activation='gelu')(pooled_output)
    x = BatchNormalization()(x)
    
    # Reshape for LSTM [batch_size, seq_length=1, features]
    x = Reshape((1, 256))(x)
    
    # Temporal modeling with LSTM
    x = LSTM(256, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = LSTM(256)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Classification head
    x = Dense(512, activation='gelu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    
    # Create F1 score metric
    f1_metric = F1ScoreMetric(num_classes=num_classes)
    
    # Create a custom weighted loss function if class weights are provided
    if class_weights is not None:
        # Convert class weights dict to tensor with shape matching the number of classes
        class_weight_tensor = tf.constant([class_weights.get(i, 1.0) for i in range(num_classes)])
        
        # Define a weighted categorical crossentropy loss function
        def weighted_categorical_crossentropy(y_true, y_pred):
            # Calculate standard categorical crossentropy
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # Get the class indices
            class_indices = tf.argmax(y_true, axis=-1)
            
            # Get the weights for each sample based on its class
            weights = tf.gather(class_weight_tensor, class_indices)
            
            # Apply the weights to the losses
            return tf.reduce_mean(cce * weights)
        
        loss_fn = weighted_categorical_crossentropy
    else:
        loss_fn = 'categorical_crossentropy'
    
    # Use fixed learning rate instead of cosine decay to ensure model learns
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Fixed learning rate
        loss=loss_fn,
        metrics=['accuracy', f1_metric]
    )
    
    return model

def visualize_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Train F1 Score')
    plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig('training_history.png')
    print("Training history plot saved to 'training_history.png'")
    
    # Show plot
    plt.show()

def train_model(epochs=40, test_size=0.2, random_state=42, checkpoint_dir='models', dataset_fraction=0.3):
    """Run the complete training pipeline.
    
    Args:
        epochs: Number of training epochs (default: 40)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        checkpoint_dir: Directory to save model checkpoints
        dataset_fraction: Fraction of the full dataset to use (default: 0.3)
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"basketball51_training_log_{timestamp}.txt"
    
    print_section("BASKETBALL-51 ACTION RECOGNITION TRAINING")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Dataset Fraction: {dataset_fraction*100:.0f}%")
    print(f"Epochs: {epochs}")
    print(f"Log File: {log_file}")
    
    # Verify dataset path
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path does not exist: {DATASET_PATH}")
        return
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    try:
        # Get video paths and labels
        print_section("DATASET PREPARATION")
        video_paths, labels, label2id = get_video_paths_and_labels(DATASET_PATH)
        num_classes = len(label2id)
        id2label = {idx: label for label, idx in label2id.items()}
        
        print(f"Found {len(video_paths)} videos across {num_classes} classes")
        
        # Sample only a fraction of the dataset to reduce memory usage
        if dataset_fraction < 1.0:
            # Use stratified sampling to maintain class distribution
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-dataset_fraction, random_state=random_state)
            indices = list(range(len(video_paths)))
            
            for train_idx, _ in sss.split(indices, labels):
                sampled_indices = train_idx
            
            # Get the sampled paths and labels
            video_paths = [video_paths[i] for i in sampled_indices]
            labels = [labels[i] for i in sampled_indices]
            
            print(f"Sampled {len(video_paths)} videos ({dataset_fraction*100:.0f}% of dataset)")
            
            # Count classes after sampling
            class_counts = {}
            for label in labels:
                class_name = id2label[label]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1
            
            print("Sampled dataset class distribution:")
            for class_name, count in class_counts.items():
                print(f"  - {class_name}: {count} videos")
        
        # Split data into train and test sets
        train_p, test_p, train_l, test_l = train_test_split(
            video_paths, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        
        print(f"Training set: {len(train_p)} videos")
        print(f"Test set: {len(test_p)} videos")
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_p) // BATCH_SIZE
        validation_steps = max(1, len(test_p) // BATCH_SIZE)
        
        # Compute class weights to handle imbalance
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print("\nCalculated class weights to handle imbalance:")
        for class_id, weight in class_weight_dict.items():
            print(f"  - Class {id2label[class_id]}: {weight:.4f}")
        print("(Will be applied through weighted loss function)")
        
        print_section("DATA PROCESSING")
        print("Using memory-efficient data generator for training data")
        
        # Create data generators instead of loading all data at once
        train_generator = create_data_generator(
            train_p, train_l, SEQ_LENGTH, IMG_SIZE, num_classes, BATCH_SIZE
        )
        
        # For the test set, we'll process it in batches but keep it in memory
        # since it's smaller and we need it for final evaluation
        print("Processing test data in batches...")
        X_test, y_test = process_test_batch(
            test_p, test_l, SEQ_LENGTH, IMG_SIZE, num_classes
        )
        
        print(f"Test data shape: {X_test.shape}")
        
        # Build model
        print_section("MODEL ARCHITECTURE")
        input_shape = (SEQ_LENGTH, IMG_SIZE[1], IMG_SIZE[0], 5)  # 3 (RGB) + 2 (flow)
        model = build_model(input_shape, num_classes, class_weight_dict)
        model.summary()
        
        # Callbacks with increased patience
        callbacks = [
            ModelCheckpoint(
                os.path.join(checkpoint_dir, f'best_model_{timestamp}.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,  # Increased patience to avoid early stopping
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Gentler reduction
                patience=5,   # More patience before reducing
                min_lr=1e-6
            )
        ]
        
        # Train model
        print_section("TRAINING")
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        # Use fit with generator (without class_weight parameter)
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callbacks
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Evaluate model
        print_section("EVALUATION")
        print("Evaluating model on test set...")
        loss, acc, f1 = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f'final_model_{timestamp}.keras')
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Create a results dataframe
        results = {
            'Class': list(id2label.values()),
            'Class_ID': list(id2label.keys())
        }
        results_df = pd.DataFrame(results)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Epoch': list(range(1, len(history.history['accuracy']) + 1)),
            'Train_Accuracy': history.history['accuracy'],
            'Val_Accuracy': history.history['val_accuracy'],
            'Train_Loss': history.history['loss'],
            'Val_Loss': history.history['val_loss'],
            'Train_F1': history.history['f1_score'],
            'Val_F1': history.history['val_f1_score']
        })
        
        metrics_file = f'training_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Training metrics saved to {metrics_file}")
        
        # Visualize training progress
        print_section("VISUALIZATION")
        visualize_training_history(history)
        
        # Save training summary to log file
        with open(log_file, 'w') as f:
            f.write(f"BASKETBALL-51 ACTION RECOGNITION TRAINING SUMMARY\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {DATASET_PATH}\n")
            f.write(f"Videos: {len(video_paths)}\n")
            f.write(f"Classes: {num_classes}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch Size: {BATCH_SIZE}\n")
            f.write(f"Sequence Length: {SEQ_LENGTH}\n")
            f.write(f"Image Size: {IMG_SIZE}\n")
            f.write(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
            
            f.write("RESULTS\n")
            f.write(f"Final Test Loss: {loss:.4f}\n")
            f.write(f"Final Test Accuracy: {acc:.4f}\n")
            f.write(f"Final Test F1 Score: {f1:.4f}\n\n")
            
            f.write("CLASS DISTRIBUTION\n")
            for class_name, count in class_counts.items():
                f.write(f"  - {class_name}: {count} videos\n")
            
            f.write("\nFINAL EPOCH METRICS\n")
            final_epoch = metrics_df.iloc[-1]
            f.write(f"Training Accuracy: {final_epoch['Train_Accuracy']:.4f}\n")
            f.write(f"Validation Accuracy: {final_epoch['Val_Accuracy']:.4f}\n")
            f.write(f"Training Loss: {final_epoch['Train_Loss']:.4f}\n")
            f.write(f"Validation Loss: {final_epoch['Val_Loss']:.4f}\n")
            f.write(f"Training F1 Score: {final_epoch['Train_F1']:.4f}\n")
            f.write(f"Validation F1 Score: {final_epoch['Val_F1']:.4f}\n")
        
        print(f"Training summary saved to {log_file}")
        print_section("TRAINING COMPLETE")
        
        return model, history, results_df
        
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_on_video(model_path, video_path=None):
    """Test a trained model on a specific video or the first available video."""
    print_section("VIDEO PREDICTION TEST")
    
    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
    # If no video specified, use first one found
    if video_path is None:
        video_paths, labels, label2id = get_video_paths_and_labels(DATASET_PATH)
        if not video_paths:
            print("No videos found in dataset at:", DATASET_PATH)
            return False
        video_path = video_paths[0]
        true_label = labels[0]
        id2label = {idx: lbl for lbl, idx in label2id.items()}
        print(f"Selected video: {video_path}")
        print(f"True label: {id2label[true_label]}")
    else:
        print(f"Using video: {video_path}")
        true_label = None
    
    # Process the video
    frames = extract_frames(video_path, SEQ_LENGTH, IMG_SIZE)
    flow = compute_optical_flow_sequence(frames)
    
    print("Frames shape:", frames.shape)
    print("Flow shape:  ", flow.shape)
    
    # Combine and add batch dimension
    X = np.concatenate([frames, flow], axis=-1)
    X = X[np.newaxis, ...]
    
    # Run prediction
    preds = model.predict(X, verbose=1)
    pred_idx = int(np.argmax(preds[0]))
    
    # Get class names
    _, _, label2id = get_video_paths_and_labels(DATASET_PATH)
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    
    # Display prediction
    print(f"Predicted class: {id2label[pred_idx]} (confidence: {preds[0][pred_idx]:.4f})")
    
    if true_label is not None:
        is_correct = pred_idx == true_label
        print(f"Correct prediction: {is_correct}")
    
    return True

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Basketball-51 Action Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help='Mode: train model, test model, or both (default: train)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of training epochs (default: 40)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--checkpoint_dir', type=str, default='models',
                        help='Directory to save model checkpoints (default: models)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for testing (required for test mode)')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to video file for testing (optional)')
    parser.add_argument('--dataset_fraction', type=float, default=0.3,
                        help='Fraction of the dataset to use (default: 0.3)')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'both':
        model, history, _ = train_model(
            epochs=args.epochs,
            test_size=args.test_size,
            checkpoint_dir=args.checkpoint_dir,
            dataset_fraction=args.dataset_fraction
        )
        
        # For 'both' mode, use the trained model for testing
        if args.mode == 'both' and model is not None:
            # In 'both' mode, we use the final model we just trained
            model_path = os.path.join(args.checkpoint_dir, f'final_model_{timestamp}.keras')
            test_on_video(model_path, args.video_path)
    
    elif args.mode == 'test':
        if args.model_path is None:
            print("ERROR: Model path must be specified for test mode")
            print("Use --model_path to specify the path to a saved model")
            return
        
        test_on_video(args.model_path, args.video_path)

if __name__ == "__main__":
    main() 