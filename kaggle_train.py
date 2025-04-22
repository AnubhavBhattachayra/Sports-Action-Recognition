#!/usr/bin/env python3
"""
kaggle_train.py

Basketball-51 action recognition training script optimized for Kaggle environment.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Configuration
SEQ_LENGTH = 30          # number of frames per clip
IMG_SIZE = (112, 112)    # width, height
BATCH_SIZE = 8

# Handle Kaggle vs local environment
if os.path.exists('/kaggle/input'):
    # For Kaggle: Update this path to match your dataset location in Kaggle
    DATASET_PATH = '/kaggle/input/basketball-51/Basketball-51'
    IS_KAGGLE = True
else:
    # For local environment
    DATASET_PATH = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
    IS_KAGGLE = False

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
        # Use OpenCV to read frames from the video file
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

def build_model(input_shape, num_classes):
    """Define a CNN-LSTM model for spatio-temporal classification."""
    inp = Input(shape=input_shape)
    
    # First convolutional block
    x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(inp)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    
    # Second convolutional block
    x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    
    # Third convolutional block (additional capacity)
    x = TimeDistributed(Conv2D(128, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    
    # Flatten and feed to LSTM
    x = TimeDistributed(Flatten())(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256)(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    out = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualize_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main(epochs=20, test_size=0.2, random_state=42, checkpoint_dir='models'):
    print("Basketball-51 Action Recognition Training")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Epochs: {epochs}")
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Get video paths and labels
    video_paths, labels, label2id = get_video_paths_and_labels(DATASET_PATH)
    num_classes = len(label2id)
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"Found {len(video_paths)} videos across {num_classes} classes")
    
    # Split data into train and test sets
    train_p, test_p, train_l, test_l = train_test_split(
        video_paths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    print(f"Training set: {len(train_p)} videos")
    print(f"Test set: {len(test_p)} videos")
    
    # Prepare data
    print("Preparing training data...")
    X_train, y_train = prepare_data(train_p, train_l, SEQ_LENGTH, IMG_SIZE, num_classes)
    print("Preparing test data...")
    X_test, y_test = prepare_data(test_p, test_l, SEQ_LENGTH, IMG_SIZE, num_classes)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Build model
    input_shape = (SEQ_LENGTH, IMG_SIZE[1], IMG_SIZE[0], 5)  # 3 (RGB) + 2 (flow)
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("Evaluating model...")
    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save final model
    model.save(os.path.join(checkpoint_dir, 'final_model.h5'))
    
    # Visualize training progress
    visualize_training_history(history)
    
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
        'Val_Loss': history.history['val_loss']
    })
    metrics_df.to_csv('training_metrics.csv', index=False)
    
    print("Training complete!")
    print(f"Model saved to {os.path.join(checkpoint_dir, 'final_model.h5')}")
    print(f"Training metrics saved to training_metrics.csv")
    
    return model, history, results_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Basketball-51 action recognition model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='models', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    if IS_KAGGLE:
        # In Kaggle, just run with the provided arguments
        model, history, results_df = main(
            epochs=args.epochs,
            test_size=args.test_size,
            random_state=args.random_state,
            checkpoint_dir=args.checkpoint_dir
        )
    else:
        # In local environment, use the command line arguments
        main(
            epochs=args.epochs,
            test_size=args.test_size,
            random_state=args.random_state,
            checkpoint_dir=args.checkpoint_dir
        ) 