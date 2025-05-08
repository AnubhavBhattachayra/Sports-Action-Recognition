#!/usr/bin/env python3
"""
precompute_combined.py

Combined script to precompute both RGB frames and optical flow for the Basketball-51 dataset.
Processes both modalities in parallel for each video to ensure consistency.
"""
import os
import sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

# Configuration
SEQ_LENGTH = 50
IMG_SIZE = (120, 160)  # height, width for Keras layers
IMG_SIZE_CV = (160, 120)  # width, height for cv2.resize

def extract_frames(video_path):
    """Extract frames from video and compute optical flow."""
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame sampling
        if total_frames < SEQ_LENGTH:
            print(f"Warning: Video {video_path} has only {total_frames} frames, less than required {SEQ_LENGTH}")
            return None, None

        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames-1, SEQ_LENGTH, dtype=int)
        
        # Initialize arrays
        rgb_frames = []
        prev_frame = None
        flow_frames = []
        
        # Process frames
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in frame_indices:
                # Resize frame
                frame = cv2.resize(frame, IMG_SIZE_CV)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(frame)
                
                # Compute flow if we have a previous frame
                if prev_frame is not None:
                    # Convert to grayscale for flow computation
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Compute optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    flow_frames.append(flow)
                
                prev_frame = frame.copy()
            
        cap.release()
        
        # Convert to numpy arrays
        rgb_sequence = np.array(rgb_frames, dtype=np.uint8)  # Store as uint8
        flow_sequence = np.array(flow_frames, dtype=np.float32)
        
        # Normalize flow
        flow_sequence = normalize_flow(flow_sequence)
        
        return rgb_sequence, flow_sequence
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None, None

def normalize_flow(flow_sequence):
    """Normalize flow vectors to [0, 1] range."""
    min_bound, max_bound = -20, 20
    flow_sequence = np.clip(flow_sequence, min_bound, max_bound)
    normalized_flow = (flow_sequence - min_bound) / (max_bound - min_bound)
    return normalized_flow.astype(np.float32)

def process_video(args):
    """Process a single video and save both RGB and flow data."""
    video_path, output_base, dataset_path = args
    try:
        # Extract frames and compute flow
        rgb_sequence, flow_sequence = extract_frames(video_path)
        
        if rgb_sequence is None or flow_sequence is None:
            return False
            
        # Create output paths
        relative_path = os.path.relpath(video_path, dataset_path)
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        output_dir = os.path.join(output_base, os.path.dirname(relative_path))
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save RGB data with compression
        rgb_output_path = os.path.join(output_dir, f"{base_filename}_rgb.npz")
        np.savez_compressed(rgb_output_path, data=rgb_sequence)
        
        # Save flow data with compression
        flow_output_path = os.path.join(output_dir, f"{base_filename}_flow.npz")
        np.savez_compressed(flow_output_path, data=flow_sequence)
        
        return True
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Precompute RGB and Flow data for Basketball-51 dataset")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the Basketball-51 dataset directory')
    parser.add_argument('--output_path', type=str, default='./precomputed_data',
                        help='Base path for output directories')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),
                        help='Number of worker processes')
    args = parser.parse_args()
    
    # Create output directories
    rgb_output_dir = os.path.join(args.output_path, 'rgb_data')
    flow_output_dir = os.path.join(args.output_path, 'flow_data')
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(flow_output_dir, exist_ok=True)
    
    # Get all video paths
    video_paths = []
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    if not video_paths:
        print(f"No .mp4 files found in {args.dataset_path}")
        return
    
    print(f"Found {len(video_paths)} videos")
    print(f"Using {args.workers} worker processes")
    
    # Process videos in parallel
    with Pool(args.workers) as pool:
        # Create arguments for each video
        process_args = [(path, args.output_path, args.dataset_path) for path in video_paths]
        
        # Process videos with progress bar
        results = list(tqdm(
            pool.imap(process_video, process_args),
            total=len(video_paths),
            desc="Processing videos"
        ))
    
    # Print summary
    successful = sum(results)
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {successful}/{len(video_paths)} videos")
    print(f"RGB data saved to: {rgb_output_dir}")
    print(f"Flow data saved to: {flow_output_dir}")

if __name__ == "__main__":
    main() 