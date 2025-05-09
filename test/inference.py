#!/usr/bin/env python3
"""
inference.py

Script for running inference on the Two-Stream ACA-Net model.
Supports both single sample inference and batch inference.
Includes visualization capabilities for predictions.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import argparse
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import from two_stream_aca_net.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from two_stream_aca_net import (
    build_two_stream_aca_net,
    normalize_flow,
    SEQ_LENGTH,
    IMG_SIZE,
    NUM_CLASSES
)

def load_sample(rgb_path, flow_path):
    """Load and preprocess a single sample for inference."""
    print("\n=== Loading Sample ===")
    
    # Load data from NPZ files
    rgb_npz = np.load(rgb_path)
    flow_npz = np.load(flow_path)
    
    # Get the data array from the NPZ file
    rgb_data = rgb_npz['data'] if 'data' in rgb_npz else rgb_npz['arr_0']
    flow_data = flow_npz['data'] if 'data' in flow_npz else flow_npz['arr_0']
    
    print(f"Original shapes:")
    print(f"RGB data shape: {rgb_data.shape}")
    print(f"Flow data shape: {flow_data.shape}")
    
    # Ensure both sequences are exactly SEQ_LENGTH frames
    if len(rgb_data) > SEQ_LENGTH:
        rgb_data = rgb_data[:SEQ_LENGTH]
    elif len(rgb_data) < SEQ_LENGTH:
        padding = np.tile(rgb_data[-1:], (SEQ_LENGTH - len(rgb_data), 1, 1, 1))
        rgb_data = np.concatenate([rgb_data, padding], axis=0)
    
    if len(flow_data) > SEQ_LENGTH:
        flow_data = flow_data[:SEQ_LENGTH]
    elif len(flow_data) < SEQ_LENGTH:
        padding = np.tile(flow_data[-1:], (SEQ_LENGTH - len(flow_data), 1, 1, 1))
        flow_data = np.concatenate([flow_data, padding], axis=0)
    
    # Normalize flow data
    flow_data = normalize_flow(flow_data)
    
    # Add batch dimension
    rgb_data = np.expand_dims(rgb_data, axis=0)
    flow_data = np.expand_dims(flow_data, axis=0)
    
    print(f"\nFinal input shapes:")
    print(f"RGB batch shape: {rgb_data.shape}")
    print(f"Flow batch shape: {flow_data.shape}")
    
    return rgb_data, flow_data

def visualize_prediction(rgb_data, flow_data, predictions, output_path=None):
    """Visualize the prediction results with RGB and flow frames."""
    # Get the first frame from the sequence
    rgb_frame = rgb_data[0, 0]  # First frame of first batch
    flow_frame = flow_data[0, 0]  # First frame of first batch
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot RGB frame
    ax1.imshow(rgb_frame)
    ax1.set_title('RGB Frame')
    ax1.axis('off')
    
    # Plot flow frame (magnitude)
    flow_magnitude = np.sqrt(flow_frame[..., 0]**2 + flow_frame[..., 1]**2)
    flow_plot = ax2.imshow(flow_magnitude, cmap='jet')
    ax2.set_title('Optical Flow Magnitude')
    ax2.axis('off')
    plt.colorbar(flow_plot, ax=ax2)
    
    # Add prediction results as text
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]
    plt.suptitle(f'Prediction: Class {pred_class} (Confidence: {confidence:.4f})')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def run_inference(model_path, rgb_path, flow_path, true_label=None, visualize=True):
    """Run inference on a single sample."""
    try:
        # Load and preprocess the sample
        rgb_data, flow_data = load_sample(rgb_path, flow_path)
        
        # Load the model
        print("\n=== Loading Model ===")
        rgb_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
        flow_shape = (SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 2)
        model = build_two_stream_aca_net(rgb_shape, flow_shape)
        model.load_weights(model_path)
        
        # Make prediction
        print("\n=== Making Prediction ===")
        predictions = model.predict([rgb_data, flow_data])
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        print("\n=== Results ===")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        if true_label is not None:
            print(f"True label: {true_label}")
            print(f"Correct: {predicted_class == true_label}")
        
        # Print top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 predictions:")
        for idx in top_3_idx:
            print(f"Class {idx}: {predictions[0][idx]:.4f}")
        
        # Visualize if requested
        if visualize:
            output_path = os.path.join(os.path.dirname(model_path), 'prediction_visualization.png')
            visualize_prediction(rgb_data, flow_data, predictions, output_path)
        
        return predicted_class, confidence, predictions
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Run inference with Two-Stream ACA-Net model")
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model weights')
    parser.add_argument('--rgb_path', type=str, required=True,
                      help='Path to the RGB data file')
    parser.add_argument('--flow_path', type=str, required=True,
                      help='Path to the flow data file')
    parser.add_argument('--true_label', type=int,
                      help='True label for the sample (optional)')
    parser.add_argument('--no_visualize', action='store_true',
                      help='Disable visualization')
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    if not os.path.exists(args.rgb_path):
        print(f"Error: RGB file not found: {args.rgb_path}")
        return
    if not os.path.exists(args.flow_path):
        print(f"Error: Flow file not found: {args.flow_path}")
        return
    
    # Run inference
    run_inference(
        args.model_path,
        args.rgb_path,
        args.flow_path,
        args.true_label,
        not args.no_visualize
    )

if __name__ == "__main__":
    main() 