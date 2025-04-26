#!/usr/bin/env python3
"""
run_two_stream_model.py

Script to train and test the Two-Stream ACA-Net model for basketball action recognition.
"""
import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf

# Reduce TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Two-Stream ACA-Net for Basketball Action Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'],
                        help='Mode: train, test, or visualize')
    parser.add_argument('--data_path', type=str, 
                        default=r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51',
                        help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='two_stream_aca_net_model.h5',
                        help='Path to save or load model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--test_video', type=str, default=None, 
                        help='Path to a specific video file for testing')
    parser.add_argument('--visualize_type', type=str, default='tsne', 
                        choices=['tsne', 'confusion', 'attention'],
                        help='Type of visualization to generate')
    args = parser.parse_args()
    
    # Import required modules
    from two_stream_aca_net import (
        SEQ_LENGTH, IMG_SIZE, NUM_CLASSES, BATCH_SIZE,
        build_two_stream_aca_net, preprocess_rgb_flow_data,
        train_two_stream_aca_net, evaluate_model
    )
    
    # Check dataset path
    if not os.path.exists(args.data_path):
        print(f"Dataset not found at: {args.data_path}")
        return
    
    # Import necessary functions from the existing codebase
    try:
        from basketball51_action_recognition import get_video_paths_and_labels
    except ImportError:
        print("Error importing basketball51_action_recognition.py")
        return
    
    # Get video paths and labels
    video_paths, labels, label2id = get_video_paths_and_labels(args.data_path)
    id2label = {idx: label for label, idx in label2id.items()}
    print(f"Found {len(video_paths)} videos across {len(label2id)} classes: {label2id}")
    
    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"two_stream_model_{args.mode}_{timestamp}.log"
    
    if args.mode == 'train':
        print(f"Training Two-Stream ACA-Net model (log: {log_filename})")
        
        # Preprocess data
        print("Preprocessing data...")
        X_rgb, X_flow, y = preprocess_rgb_flow_data(video_paths, labels)
        print(f"RGB data shape: {X_rgb.shape}")
        print(f"Flow data shape: {X_flow.shape}")
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_rgb_train, X_rgb_test, X_flow_train, X_flow_test, y_train, y_test = train_test_split(
            X_rgb, X_flow, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
        )
        
        # Train model
        print(f"Training model with {args.epochs} epochs and batch size {args.batch_size}...")
        model, history = train_two_stream_aca_net(
            X_rgb_train, X_flow_train, y_train, 
            epochs=args.epochs
        )
        
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
        
        # Save model
        model.save(args.model_path)
        print(f"Model saved as '{args.model_path}'")
        
        # Save training history plot
        import matplotlib.pyplot as plt
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
        plt.savefig(f'training_history_{timestamp}.png')
        print(f"Training history plot saved as 'training_history_{timestamp}.png'")
        
    elif args.mode == 'test':
        print(f"Testing Two-Stream ACA-Net model (log: {log_filename})")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        # Load the model
        model = tf.keras.models.load_model(args.model_path)
        print(f"Loaded model from: {args.model_path}")
        
        if args.test_video:
            # Test on a specific video
            if not os.path.exists(args.test_video):
                print(f"Test video not found: {args.test_video}")
                return
            
            print(f"Testing on video: {args.test_video}")
            
            # Extract frames and compute optical flow
            import cv2
            
            # Extract frames from the video
            cap = cv2.VideoCapture(args.test_video)
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
                print(f"No frames extracted from {args.test_video}")
                return
                
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
            
            # Add batch dimension
            X_rgb = rgb_frames[np.newaxis, ...]
            X_flow = flow_frames[np.newaxis, ...]
            
            # Make prediction
            y_pred = model.predict([X_rgb, X_flow])
            pred_idx = np.argmax(y_pred[0])
            
            # Print prediction
            print(f"Predicted class: {id2label[pred_idx]} (score={y_pred[0][pred_idx]:.4f})")
            
            # Display top 3 predictions with scores
            top_indices = np.argsort(y_pred[0])[-3:][::-1]
            print("\nTop 3 predictions:")
            for i, idx in enumerate(top_indices):
                print(f"{i+1}. {id2label[idx]}: {y_pred[0][idx]:.4f}")
                
        else:
            # Test on the entire test set
            print("Testing on the entire test set...")
            
            # Preprocess data
            print("Preprocessing data...")
            X_rgb, X_flow, y = preprocess_rgb_flow_data(video_paths, labels)
            
            # Train/test split
            from sklearn.model_selection import train_test_split
            _, X_rgb_test, _, X_flow_test, _, y_test = train_test_split(
                X_rgb, X_flow, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
            )
            
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
            
            # Generate confusion matrix plot
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            y_pred = model.predict([X_rgb_test, X_flow_test])
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[id2label[i] for i in range(len(id2label))],
                        yticklabels=[id2label[i] for i in range(len(id2label))])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_{timestamp}.png')
            print(f"Confusion matrix saved as 'confusion_matrix_{timestamp}.png'")
            
    elif args.mode == 'visualize':
        print(f"Visualizing Two-Stream ACA-Net model (log: {log_filename})")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return
        
        # Load the model
        model = tf.keras.models.load_model(args.model_path)
        print(f"Loaded model from: {args.model_path}")
        
        # Preprocess data for visualization
        print("Preprocessing data...")
        X_rgb, X_flow, y = preprocess_rgb_flow_data(video_paths, labels)
        y_classes = np.argmax(y, axis=1)
        
        # Create a feature extractor model (without the final classification layer)
        rgb_input = model.inputs[0]
        flow_input = model.inputs[1]
        
        # Get the layer before the final dense layer
        feature_output = model.get_layer('fusion_fc1').output
        feature_model = tf.keras.models.Model(inputs=[rgb_input, flow_input], outputs=feature_output)
        
        # Extract features
        features = feature_model.predict([X_rgb, X_flow])
        
        if args.visualize_type == 'tsne':
            # t-SNE visualization
            print("Generating t-SNE visualization...")
            
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(features)
            
            # Plot
            plt.figure(figsize=(12, 10))
            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#800000', '#008000']
            
            # Create a scatter plot
            for i, label in enumerate(np.unique(y_classes)):
                indices = np.where(y_classes == label)[0]
                plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                           c=colors[i % len(colors)], label=id2label[label], alpha=0.7)
            
            plt.title('t-SNE Visualization of Features')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'tsne_visualization_{timestamp}.png')
            print(f"t-SNE visualization saved as 'tsne_visualization_{timestamp}.png'")
            
        elif args.visualize_type == 'confusion':
            # Confusion matrix visualization
            print("Generating confusion matrix visualization...")
            
            # Split the data
            from sklearn.model_selection import train_test_split
            _, X_rgb_test, _, X_flow_test, _, y_test = train_test_split(
                X_rgb, X_flow, y, test_size=0.2, stratify=y_classes, random_state=42
            )
            
            # Make predictions
            y_pred = model.predict([X_rgb_test, X_flow_test])
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Create confusion matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=[id2label[i] for i in range(len(id2label))],
                       yticklabels=[id2label[i] for i in range(len(id2label))])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Normalized Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_norm_{timestamp}.png')
            print(f"Normalized confusion matrix saved as 'confusion_matrix_norm_{timestamp}.png'")
            
        elif args.visualize_type == 'attention':
            # Attention maps visualization
            print("Attention maps visualization not yet implemented.")
            print("This would require model architecture modification to extract attention weights.")
            
    else:
        print(f"Invalid mode: {args.mode}")
        return

if __name__ == "__main__":
    main() 