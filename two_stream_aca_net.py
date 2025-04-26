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
    LayerNormalization, Permute, Attention, Add, TimeDistributed,
    InputSpec
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
    head_dim = features // num_heads
    if head_dim * num_heads != features: # Ensure divisibility
        # Fallback or adjustment needed if features not divisible
        # For now, let's assume it is, or adjust Dense output dims
         print(f"Warning: LSTA features ({features}) not divisible by num_heads ({num_heads}). Adjusting.")
         # Example adjust: use simpler attention or adjust head_dim
         attn_output = Attention(name=f"{name_prefix}_lsta_attention")([q, k, v]) # Simplified fallback
    else:
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
    channel_attn = Dense(features // 16, activation='relu', name=f"{name_prefix}_tsci_channel_fc1")(channel_pool_temporal)
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
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(h, w, 3), pooling=None) # pooling=None initially

    # Handle Flow Input (2 channels) vs RGB (3 channels)
    if name_prefix == "flow" and c == 2:
        # Need to adapt the first layer of ResNet50 for 2 channels
        # Create a new input layer for the flow stream with the correct shape
        flow_input_tensor = Input(shape=(h, w, c), name="flow_input_tensor")
        # Create a new first conv layer matching ResNet's conv1 but with 2 input channels
        # Keep strides, padding, etc., the same as ResNet50's conv1
        flow_conv1 = Conv2D(64, kernel_size=7, strides=2, padding='same', name=f"{name_prefix}_conv1_custom")(flow_input_tensor)
        # Create a model segment for the modified first layer
        first_layer_model = Model(inputs=flow_input_tensor, outputs=flow_conv1, name=f"{name_prefix}_first_layer")

        # Apply the modified first layer using TimeDistributed
        x = TimeDistributed(first_layer_model, name=f"{name_prefix}_td_first_layer")(inputs)

        # Freeze original ResNet conv1 weights if loading imagenet weights? No, train from scratch or adapt.
        # For simplicity here, let's allow this first custom layer to train.

        # Apply the rest of the ResNet layers (skipping the original conv1) using TimeDistributed
        # Need to carefully connect output of flow_conv1 to input of ResNet's layer after conv1
        # Get ResNet layers after conv1
        resnet_layers_after_conv1 = Model(inputs=base_model.get_layer('conv1_relu').output, # Example layer after conv1
                                           outputs=base_model.output, # Output before global pooling
                                           name=f"{name_prefix}_resnet_body")
        x = TimeDistributed(resnet_layers_after_conv1, name=f"{name_prefix}_td_resnet_body")(x)

    else: # RGB Stream (c=3) - Use standard ResNet50
        # Wrap the standard ResNet50 in TimeDistributed
        # Important: ResNet50 expects 3 channels. Flow stream might need different handling if c != 3.
        if c != 3:
            raise ValueError(f"RGB stream expects 3 channels, but got {c}")

        # Option 1: Modify ResNet50 first layer (as above, but less common for RGB)
        # Option 2: Use standard ResNet50 (requires input shape (h, w, 3))
        # We assume standard RGB input for this branch.
        rgb_base = ResNet50(include_top=False, weights='imagenet', input_shape=(h, w, 3), pooling='avg') # Use avg pooling here
        # Set layers to non-trainable if using pre-trained weights and not fine-tuning all
        # for layer in rgb_base.layers[:-4]: # Example: freeze all but last few
        #     layer.trainable = False

        # Apply TimeDistributed ResNet
        x = TimeDistributed(rgb_base, name=f"{name_prefix}_td_resnet")(inputs)
        # Output shape here is (batch, seq_len, 2048) because pooling='avg'

    # If pooling wasn't done in base_model (e.g., for flow), apply it now
    if name_prefix == "flow": # Assuming flow stream didn't use pooling='avg' in base
         # Output of resnet_layers_after_conv1 is feature map, need GAP
         # Reshape needed before applying GAP within TimeDistributed context
         # Current shape: (batch, seq_len, feat_h, feat_w, feat_c)
         # Desired input for LSTA/TSCI: (batch, seq_len, features)
         x = TimeDistributed(GlobalAveragePooling2D(), name=f"{name_prefix}_td_gap")(x)
         # Output shape: (batch, seq_len, 2048)

    # Feature dimension is now (batch, seq_len, 2048) for both streams

    # Apply LSTA module for temporal attention
    x = LSTA_module(x, name_prefix=name_prefix)
    # Output shape: (batch, seq_len, 2048)

    # Apply TSCI module for spatial-channel interaction (adapted for temporal features)
    x = TSCI_module(x, name_prefix=name_prefix)
    # Output shape: (batch, seq_len, 2048)

    # Final Pooling across time dimension to get one vector per sequence
    # Use GlobalAveragePooling1D across the sequence length
    final_features = GlobalAveragePooling1D(name=f"{name_prefix}_global_avg_pool_time")(x)
    # Output shape: (batch, 2048)

    # Create the stream model
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
    model = build_two_stream_aca_net(rgb_shape, flow_shape, NUM_CLASSES)
    
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
    DATASET_PATH = '/kaggle/input/basketball-51/Basketball-51' # Updated path
    
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