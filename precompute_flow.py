#!/usr/bin/env python3
"""
precompute_flow.py

Calculates optical flow (using Farneback for speed) for all videos
in the Basketball-51 dataset and saves them as NumPy arrays.
Uses multiprocessing for faster computation.

Run this script *once* before training the main model.
"""
import os
import sys
import cv2 # OpenCV is needed
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# --- Configuration (should match training script) ---
SEQ_LENGTH = 50
IMG_SIZE = (160, 120) # width, height for OpenCV resize function
# Note: Training script uses (height, width) for Keras layers (120, 160)
# Ensure consistency in interpretation or adjust resize accordingly.
# Using (width, height) for cv2.resize here.

# --- Helper Functions ---

def extract_frames(video_path, seq_length, img_size_cv):
    """Extracts, resizes, samples/pads frames similar to training setup."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}", file=sys.stderr)
        return None

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use img_size_cv (width, height) for OpenCV
        frame = cv2.resize(frame, img_size_cv)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Keep RGB for consistency if needed later
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        print(f"Warning: No frames extracted from {video_path}", file=sys.stderr)
        return None

    # Sample or pad frames
    if len(all_frames) >= seq_length:
        idxs = np.linspace(0, len(all_frames) - 1, seq_length).astype(int)
        frames = [all_frames[i] for i in idxs]
    else:
        frames = all_frames + [all_frames[-1]] * (seq_length - len(all_frames))

    # Return as list of RGB frames (uint8) before flow calculation
    return frames # List of numpy arrays

def compute_and_save_flow(video_path, dataset_base_path, output_base_path, seq_length, img_size_cv):
    """Processes a single video: extracts frames, computes flow, saves flow."""
    try:
        # Extract RGB frames (as uint8)
        rgb_frames_list = extract_frames(video_path, seq_length, img_size_cv)
        if rgb_frames_list is None:
            return video_path, False # Indicate failure

        # Convert first frame to grayscale for flow calculation start
        prev_gray = cv2.cvtColor(rgb_frames_list[0], cv2.COLOR_RGB2GRAY)

        flow_frames = []
        # Calculate Farneback flow for subsequent frames
        for i in range(1, len(rgb_frames_list)):
            curr_rgb = rgb_frames_list[i]
            curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                               None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_frames.append(flow)
            prev_gray = curr_gray # Update previous frame

        # Handle the sequence length: Insert flow for the first frame
        # Common practice: duplicate the first calculated flow map
        if flow_frames:
             flow_frames.insert(0, flow_frames[0])
        else: # If only one frame extracted somehow
             # Create a zero flow map
             zero_flow = np.zeros((img_size_cv[1], img_size_cv[0], 2), dtype=np.float32)
             flow_frames = [zero_flow] * seq_length


        # Pad if fewer than seq_length flow maps were generated (shouldn't happen with insert)
        while len(flow_frames) < seq_length:
             flow_frames.append(flow_frames[-1])

        flow_sequence = np.array(flow_frames, dtype=np.float32) # Shape: (seq_length, H, W, 2)

        # Determine output path
        relative_path = os.path.relpath(video_path, dataset_base_path)
        output_dir = os.path.join(output_base_path, os.path.dirname(relative_path))
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        output_filename = os.path.join(output_dir, f"{base_filename}_flow.npy")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the flow sequence
        np.save(output_filename, flow_sequence)
        return video_path, True # Indicate success

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}", file=sys.stderr)
        return video_path, False # Indicate failure


def get_video_paths(dataset_path):
    """Finds all .mp4 video file paths within the dataset directory."""
    video_paths = []
    print(f"Searching for videos in: {dataset_path}")
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}", file=sys.stderr)
        return []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    print(f"Found {len(video_paths)} video files.")
    return video_paths

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Precompute Optical Flow for Basketball-51 dataset.")
    parser.add_argument('--dataset_path', type=str, default='/kaggle/input/basketball-51/Basketball-51',
                        help='Path to the root directory of the Basketball-51 dataset.')
    parser.add_argument('--output_path', type=str, default='./flow_data',
                        help='Path to the directory where flow .npy files will be saved.')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),
                        help='Number of worker processes for parallel computation.')
    args = parser.parse_args()

    print(f"Using {args.workers} worker processes.")
    print(f"Output directory: {args.output_path}")

    video_paths = get_video_paths(args.dataset_path)
    if not video_paths:
        return # Exit if no videos found

    # Create the main output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Prepare partial function for multiprocessing map
    process_func = partial(compute_and_save_flow,
                           dataset_base_path=args.dataset_path,
                           output_base_path=args.output_path,
                           seq_length=SEQ_LENGTH,
                           img_size_cv=IMG_SIZE) # Pass cv2 compatible size

    # Use multiprocessing Pool
    processed_count = 0
    failed_videos = []
    with Pool(processes=args.workers) as pool:
        # Use tqdm for progress bar
        with tqdm(total=len(video_paths), desc="Computing Flow") as pbar:
            # Use imap_unordered for potentially better performance and progress updates
            for video_path, success in pool.imap_unordered(process_func, video_paths):
                if success:
                    processed_count += 1
                else:
                    failed_videos.append(video_path)
                pbar.update(1) # Update progress bar for each completed video

    print(f"\nFinished processing.")
    print(f"Successfully processed and saved flow for {processed_count}/{len(video_paths)} videos.")
    if failed_videos:
        print(f"Failed to process {len(failed_videos)} videos:")
        # for vid in failed_videos:
        #     print(f"  - {vid}") # Can be very verbose

if __name__ == "__main__":
    main() 