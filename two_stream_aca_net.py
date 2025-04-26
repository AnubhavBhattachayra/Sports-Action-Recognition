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
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    GlobalAveragePooling2D, Concatenate, Reshape, Multiply,
    LayerNormalization, Permute, Attention, Add, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Configuration
SEQ_LENGTH = 50  # 50 consecutive frames (instead of 30)
IMG_SIZE = (120, 160)  # width, height as specified in the document (instead of 112, 112)
NUM_CLASSES = 8  # 8 action classes in Basketball-51 (2p0, 2p1, 3p0, 3p1, etc.)
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4

def LSTA_module(x, name_prefix="rgb"):
    """
    Long Short-Term Attention (LSTA) module for temporal attention
    
    Args:
        x: Input tensor, shape (batch, seq_len, height, width, channels)
        name_prefix: Prefix for naming layers (for two separate streams)
        
    Returns:
        Tensor with same shape but with temporal attention applied
    """
    # Reshape to handle temporal dimension
    batch_size, seq_len, h, w, c = tf.keras.backend.int_shape(x)
    
    # Reshape to (batch*seq_len, h, w, c) to apply shared CNN operations
    x_reshaped = Reshape((batch_size * seq_len, h, w, c))(x)
    
    # Apply temporal attention
    # First project to query, key, value spaces
    q = Conv2D(c // 8, kernel_size=1, name=f"{name_prefix}_lsta_q")(x_reshaped)
    k = Conv2D(c // 8, kernel_size=1, name=f"{name_prefix}_lsta_k")(x_reshaped)
    v = Conv2D(c // 2, kernel_size=1, name=f"{name_prefix}_lsta_v")(x_reshaped)
    
    # Reshape back to include temporal dimension
    q = Reshape((batch_size, seq_len, h * w, c // 8))(q)
    k = Reshape((batch_size, seq_len, h * w, c // 8))(k)
    v = Reshape((batch_size, seq_len, h * w, c // 2))(v)
    
    # Apply attention mechanism
    attn_output = Attention(name=f"{name_prefix}_lsta_attention")([q, k, v])
    
    # Reshape to match original dimensions
    attn_output = Reshape((batch_size, seq_len, h, w, c // 2))(attn_output)
    
    # Project back to original channel dimension
    proj = Conv2D(c, kernel_size=1, name=f"{name_prefix}_lsta_proj")(
        Reshape((batch_size * seq_len, h, w, c // 2))(attn_output)
    )
    proj = Reshape((batch_size, seq_len, h, w, c))(proj)
    
    # Residual connection
    output = Add(name=f"{name_prefix}_lsta_residual")([x, proj])
    
    return output

def TSCI_module(x, name_prefix="rgb"):
    """
    Temporal-Spatial-Channel Interaction (TSCI) module
    
    Args:
        x: Input tensor, shape (batch, seq_len, height, width, channels)
        name_prefix: Prefix for naming layers (for two separate streams)
        
    Returns:
        Enhanced feature maps
    """
    # Get shape
    batch_size, seq_len, h, w, c = tf.keras.backend.int_shape(x)
    
    # Spatial attention
    # Average pooling across channels
    spatial_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    spatial_attn = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid',
                         name=f"{name_prefix}_tsci_spatial_attn")(
        Reshape((batch_size * seq_len, h, w, 1))(spatial_pool)
    )
    spatial_attn = Reshape((batch_size, seq_len, h, w, 1))(spatial_attn)
    
    # Apply spatial attention
    x_spatial = Multiply(name=f"{name_prefix}_tsci_spatial_multiply")([x, spatial_attn])
    
    # Channel attention
    # Global average pooling across spatial dimensions
    channel_pool = GlobalAveragePooling2D()(
        Reshape((batch_size * seq_len, h, w, c))(x)
    )
    channel_pool = Reshape((batch_size, seq_len, c))(channel_pool)
    
    # MLP for channel attention
    channel_attn = Dense(c // 16, activation='relu', name=f"{name_prefix}_tsci_channel_fc1")(
        Reshape((batch_size * seq_len, c))(channel_pool)
    )
    channel_attn = Dense(c, activation='sigmoid', name=f"{name_prefix}_tsci_channel_fc2")(channel_attn)
    channel_attn = Reshape((batch_size, seq_len, 1, 1, c))(channel_attn)
    
    # Apply channel attention
    x_channel = Multiply(name=f"{name_prefix}_tsci_channel_multiply")([x, channel_attn])
    
    # Combine spatial and channel attention
    output = Add(name=f"{name_prefix}_tsci_combine")([x_spatial, x_channel])
    
    return output

def ACA_Block(x, filters, name_prefix="rgb"):
    """
    Attentive Context Aware (ACA) Block built on top of ResNet block
    
    Args:
        x: Input tensor
        filters: Number of filters
        name_prefix: Prefix for naming layers (for two separate streams)
        
    Returns:
        Output tensor
    """
    # Apply 1x1 conv to match filter dimensions if needed
    input_channels = tf.keras.backend.int_shape(x)[-1]
    if input_channels != filters:
        shortcut = Conv2D(filters, kernel_size=1, name=f"{name_prefix}_aca_shortcut")(x)
    else:
        shortcut = x
        
    # Standard ResNet operations
    x = Conv2D(filters, kernel_size=3, padding='same', activation='relu',
              kernel_regularizer=l2(WEIGHT_DECAY),
              name=f"{name_prefix}_aca_conv1")(x)
    x = LayerNormalization(name=f"{name_prefix}_aca_ln1")(x)
    x = Conv2D(filters, kernel_size=3, padding='same',
              kernel_regularizer=l2(WEIGHT_DECAY),
              name=f"{name_prefix}_aca_conv2")(x)
    
    # Add shortcut connection
    x = Add(name=f"{name_prefix}_aca_add")([x, shortcut])
    x = tf.keras.activations.relu(x)
    
    return x

def build_aca_net_stream(input_shape, name_prefix="rgb"):
    """
    Build a single ACA-Net stream (either RGB or Flow)
    
    Args:
        input_shape: Shape of input tensor (seq_len, height, width, channels)
        name_prefix: Prefix for naming layers (for two separate streams)
        
    Returns:
        Model representing a single stream of the Two-Stream ACA-Net
    """
    seq_len, h, w, c = input_shape
    
    # Input layer
    inputs = Input(shape=input_shape, name=f"{name_prefix}_input")
    
    # Modify first conv layer of ResNet50 to handle different input channels
    if name_prefix == "flow" and c == 2:
        # For optical flow input (2 channels)
        x = Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu',
                  name=f"{name_prefix}_first_conv")(inputs)
    else:
        # For RGB input (3 channels) or Flow input (if 3 channels)
        # Use TimeDistributed to apply the same ResNet across all frames
        x = TimeDistributed(Conv2D(64, kernel_size=7, strides=2, padding='same', 
                           activation='relu', name=f"{name_prefix}_first_conv"))(inputs)
    
    # Continue with ResNet50 blocks but add ACA-Blocks in between
    # (simplified version - in a full implementation, we'd modify ResNet50 directly)
    
    # Block 1
    x = TimeDistributed(MaxPooling2D(pool_size=3, strides=2, padding='same'))(x)
    x = TimeDistributed(ACA_Block(x, 64, name_prefix=f"{name_prefix}_b1"))(x)
    
    # Block 2
    x = TimeDistributed(ACA_Block(x, 128, name_prefix=f"{name_prefix}_b2"))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2))(x)
    
    # Block 3
    x = TimeDistributed(ACA_Block(x, 256, name_prefix=f"{name_prefix}_b3"))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2))(x)
    
    # Block 4
    x = TimeDistributed(ACA_Block(x, 512, name_prefix=f"{name_prefix}_b4"))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=2, strides=2))(x)
    
    # Block 5
    x = TimeDistributed(ACA_Block(x, 1024, name_prefix=f"{name_prefix}_b5"))(x)
    x = TimeDistributed(ACA_Block(x, 2048, name_prefix=f"{name_prefix}_b6"))(x)
    
    # Apply LSTA module for temporal attention
    x = LSTA_module(x, name_prefix=name_prefix)
    
    # Apply TSCI module for spatial-channel interaction
    x = TSCI_module(x, name_prefix=name_prefix)
    
    # Global average pooling
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # Obtain a 2048-dimensional feature vector (as specified in the document)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    model = Model(inputs=inputs, outputs=x, name=f"{name_prefix}_stream")
    
    return model

def build_two_stream_aca_net(rgb_shape, flow_shape):
    """
    Build the complete Two-Stream ACA-Net
    
    Args:
        rgb_shape: Shape of RGB input (seq_len, height, width, 3)
        flow_shape: Shape of optical flow input (seq_len, height, width, 2)
        
    Returns:
        Complete Two-Stream ACA-Net model
    """
    # Build RGB stream
    rgb_stream = build_aca_net_stream(rgb_shape, name_prefix="rgb")
    rgb_input = rgb_stream.input
    rgb_features = rgb_stream.output
    
    # Build Flow stream
    flow_stream = build_aca_net_stream(flow_shape, name_prefix="flow")
    flow_input = flow_stream.input
    flow_features = flow_stream.output
    
    # Concatenate features from both streams (2048 + 2048 = 4096)
    concat_features = Concatenate(name="fusion_concat")([rgb_features, flow_features])
    
    # Fully connected layers for classification as specified in the document
    x = Dense(1024, activation='relu', kernel_regularizer=l2(WEIGHT_DECAY), name="fusion_fc1")(concat_features)
    x = Dropout(0.5, name="fusion_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name="prediction")(x)
    
    # Create the two-stream model
    model = Model(inputs=[rgb_input, flow_input], outputs=outputs, name="two_stream_aca_net")
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_rgb_flow_data(video_paths, labels):
    """
    Preprocess the data by extracting RGB frames and optical flow maps
    
    Args:
        video_paths: List of paths to video files
        labels: List of corresponding labels
        
    Returns:
        X_rgb: RGB frames
        X_flow: Optical flow maps
        y: One-hot encoded labels
    """
    from tensorflow.keras.utils import to_categorical
    import cv2
    from tqdm import tqdm
    
    X_rgb, X_flow, y = [], [], []
    
    for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Preprocessing videos"):
        try:
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if len(frames) == 0:
                print(f"Warning: No frames extracted from {video_path}")
                continue
                
            # Sample or pad frames to match SEQ_LENGTH
            if len(frames) >= SEQ_LENGTH:
                # Sample frames evenly
                indices = np.linspace(0, len(frames) - 1, SEQ_LENGTH).astype(int)
                rgb_frames = [frames[i] for i in indices]
            else:
                # Pad with the last frame
                rgb_frames = frames + [frames[-1]] * (SEQ_LENGTH - len(frames))
                
            # Convert to numpy array and normalize
            rgb_frames = np.array(rgb_frames, dtype='float32') / 255.0
            
            # Compute optical flow
            flow_frames = []
            prev_gray = cv2.cvtColor((rgb_frames[0] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
            
            for i in range(1, len(rgb_frames)):
                curr_gray = cv2.cvtColor((rgb_frames[i] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
                # Use TV-L1 algorithm for better flow estimation
                # If not available, fallback to Farneback
                try:
                    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
                    flow = optical_flow.calc(prev_gray, curr_gray, None)
                except:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                                     None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_frames.append(flow)
                prev_gray = curr_gray
                
            # Duplicate first flow to match sequence length
            flow_frames.insert(0, flow_frames[0])
            flow_frames = np.array(flow_frames, dtype='float32')
            
            # Data augmentation would go here (random crop, flip, etc.)
            # For simplicity, omitted in this implementation
            
            X_rgb.append(rgb_frames)
            X_flow.append(flow_frames)
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    # Convert to numpy arrays
    X_rgb = np.array(X_rgb)
    X_flow = np.array(X_flow)
    y = to_categorical(y, NUM_CLASSES)
    
    return X_rgb, X_flow, y

def train_two_stream_aca_net(X_rgb, X_flow, y, epochs=30, validation_split=0.2):
    """
    Train the Two-Stream ACA-Net
    
    Args:
        X_rgb: RGB input data
        X_flow: Optical flow input data
        y: Labels
        epochs: Number of training epochs
        validation_split: Fraction of data to use for validation
        
    Returns:
        Trained model and training history
    """
    # Split data into train and validation sets
    # For simplicity, using indexing instead of sklearn's train_test_split
    num_samples = len(X_rgb)
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * validation_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    X_rgb_train, X_rgb_val = X_rgb[train_idx], X_rgb[val_idx]
    X_flow_train, X_flow_val = X_flow[train_idx], X_flow[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Define model input shapes
    rgb_shape = X_rgb.shape[1:]   # (seq_len, height, width, 3)
    flow_shape = X_flow.shape[1:] # (seq_len, height, width, 2)
    
    # Build the model
    model = build_two_stream_aca_net(rgb_shape, flow_shape)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        [X_rgb_train, X_flow_train], y_train,
        validation_data=([X_rgb_val, X_flow_val], y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return model, history

def evaluate_model(model, X_rgb_test, X_flow_test, y_test):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_rgb_test: RGB test data
        X_flow_test: Optical flow test data
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    # Get predictions
    y_pred_prob = model.predict([X_rgb_test, X_flow_test])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Try to calculate AUC (requires scikit-learn >= 0.24)
    try:
        metrics['auc'] = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    except:
        metrics['auc'] = 'N/A (requires scikit-learn >= 0.24)'
    
    return metrics

def main():
    """Main function to run the Two-Stream ACA-Net pipeline"""
    print("Two-Stream ACA-Net for Basketball Action Recognition")
    
    # Define dataset path (adjust as needed)
    DATASET_PATH = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        return
    
    # Get video paths and labels
    from basketball51_action_recognition import get_video_paths_and_labels
    video_paths, labels, label2id = get_video_paths_and_labels(DATASET_PATH)
    print(f"Found {len(video_paths)} videos across {len(label2id)} classes: {label2id}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_rgb, X_flow, y = preprocess_rgb_flow_data(video_paths, labels)
    print(f"RGB data shape: {X_rgb.shape}")
    print(f"Flow data shape: {X_flow.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_rgb_train, X_rgb_test, X_flow_train, X_flow_test, y_train, y_test = train_test_split(
        X_rgb, X_flow, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
    )
    
    # Train model
    print("Training model...")
    model, history = train_two_stream_aca_net(X_rgb_train, X_flow_train, y_train, epochs=30)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_rgb_test, X_flow_test, y_test)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save model
    model.save("two_stream_aca_net_model.h5")
    print("Model saved as 'two_stream_aca_net_model.h5'")

if __name__ == "__main__":
    main() 