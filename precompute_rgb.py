#!/usr/bin/env python3
"""
precompute_rgb.py

Extracts, processes (resize, sample/pad, normalize), and saves RGB frame
sequences from the Basketball-51 dataset as NumPy arrays.
Uses multiprocessing for faster computation.

Run this script *once* before training the main model, after setting up the dataset.
"""
import os
import sys
import cv2
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import random
from tqdm import tqdm

# --- Configuration (should match training script) ---
SEQ_LENGTH = 50
IMG_SIZE = (120, 160) # height, width for Keras/output array
IMG_SIZE_CV = (160, 120) # width, height for OpenCV resize function
SEED = 42 # Use the same seed as other scripts for subset consistency

# --- Helper Functions ---

def extract_and_process_rgb(video_path, seq_length, img_size_cv, img_size_keras):
    """Extracts, resizes, samples/pads, and normalizes RGB frames."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}", file=sys.stderr)
        return None # Indicate failure

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize using OpenCV dimensions (width, height)
        frame = cv2.resize(frame, img_size_cv)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # Convert to NumPy array and normalize
    # Output shape should be (seq_length, height, width, channels) using Keras dimensions
    rgb_sequence = np.array(frames, dtype=np.float32) / 255.0

    # Verify shape matches Keras expectation (height, width)
    expected_shape = (seq_length, img_size_keras[0], img_size_keras[1], 3)
    if rgb_sequence.shape != expected_shape:
        print(f"Warning: Shape mismatch for {video_path}. Got {rgb_sequence.shape}, expected {expected_shape}", file=sys.stderr)
        # Attempt to reshape or handle error, for now return None
        return None

    return rgb_sequence # Shape: (seq_length, H, W, 3), float32, normalized

def process_and_save_rgb(video_path, dataset_base_path, output_base_path, seq_length, img_size_cv, img_size_keras):
    """Processes a single video: extracts/processes RGB frames and saves them."""
    try:
        # Extract and process RGB frames
        rgb_sequence = extract_and_process_rgb(video_path, seq_length, img_size_cv, img_size_keras)

        if rgb_sequence is None:
            return video_path, False # Indicate failure

        # Determine output path
        relative_path = os.path.relpath(video_path, dataset_base_path)
        output_dir = os.path.join(output_base_path, os.path.dirname(relative_path))
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        # Consistent naming convention
        output_filename = os.path.join(output_dir, f"{base_filename}_rgb.npy")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the RGB sequence
        np.save(output_filename, rgb_sequence)
        return video_path, True # Indicate success

    except Exception as e:
        print(f"Error processing {video_path} for RGB: {str(e)}", file=sys.stderr)
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
    parser = argparse.ArgumentParser(description="Precompute RGB frame sequences for Basketball-51 dataset.")
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51', # Adjust default for local if needed
                        help='Path to the root directory of the Basketball-51 video dataset.')
    parser.add_argument('--output_path', type=str, default='./rgb_data', # Default output dir for RGB
                        help='Path to the directory where RGB .npy files will be saved.')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),
                        help='Number of worker processes for parallel computation.')
    parser.add_argument('--subset_fraction', type=float, default=0.5,
                        help='Fraction of the dataset to process (e.g., 0.5 for 50%).')
    args = parser.parse_args()

    print(f"Using {args.workers} worker processes.")
    print(f"Output directory for RGB data: {args.output_path}")

    video_paths = get_video_paths(args.dataset_path)
    if not video_paths:
        return # Exit if no videos found

    # --- Select Subset (Consistent Selection) ---
    random.seed(SEED)
    num_videos_to_use = int(len(video_paths) * args.subset_fraction)
    random.shuffle(video_paths) # Shuffle deterministically
    video_paths_subset = video_paths[:num_videos_to_use]
    print(f"Using a {args.subset_fraction*100:.0f}% subset: {len(video_paths_subset)} videos out of {len(video_paths)} total.")
    # --- End Subset Selection ---

    # Create the main output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Prepare partial function for multiprocessing map
    process_func = partial(process_and_save_rgb,
                           dataset_base_path=args.dataset_path,
                           output_base_path=args.output_path,
                           seq_length=SEQ_LENGTH,
                           img_size_cv=IMG_SIZE_CV, # Pass cv2 size
                           img_size_keras=IMG_SIZE) # Pass Keras size

    # Use multiprocessing Pool
    processed_count = 0
    failed_videos = []
    with Pool(processes=args.workers) as pool:
        with tqdm(total=len(video_paths_subset), desc="Processing RGB Frames") as pbar:
            for video_path, success in pool.imap_unordered(process_func, video_paths_subset):
                if success:
                    processed_count += 1
                else:
                    failed_videos.append(video_path)
                pbar.update(1)

    print(f"\nFinished processing RGB frames.")
    print(f"Successfully processed and saved RGB for {processed_count}/{len(video_paths_subset)} videos.")
    if failed_videos:
        print(f"Failed to process RGB for {len(failed_videos)} videos:")
        # for vid in failed_videos: print(f"  - {vid}") # Verbose output

if __name__ == "__main__":
    main() 