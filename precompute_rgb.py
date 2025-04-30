#!/usr/bin/env python3
"""
precompute_rgb.py

Extracts RGB frames for all videos in the (subset of) Basketball-51 dataset,
processes them (resize, sample/pad, normalize), and saves them as NumPy arrays.
Uses multiprocessing for faster computation.

Run this script *once* after deciding on the subset and before training.
"""
import os
import sys
import cv2 # OpenCV is needed
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import random # For subset selection

# --- Configuration (should match training script) ---
SEQ_LENGTH = 50
IMG_SIZE = (120, 160) # height, width for Keras layers
IMG_SIZE_CV = (160, 120) # width, height for OpenCV resize function
SEED = 42 # Use the same seed as other scripts

# --- Helper Functions (Copied/adapted from two_stream_aca_net.py) ---

def extract_and_save_rgb(video_path, dataset_base_path, output_base_path, seq_length, img_size_cv):
    """Extracts, processes, and saves RGB frames for a single video."""
    try:
        # 1. Extract frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}", file=sys.stderr)
            return video_path, False # Indicate failure

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
            return video_path, False # Indicate failure

        # 2. Sample or pad frames
        processed_frames = []
        if len(all_frames) >= seq_length:
            idxs = np.linspace(0, len(all_frames) - 1, seq_length).astype(int)
            processed_frames = [all_frames[i] for i in idxs]
        else:
            processed_frames = all_frames + [all_frames[-1]] * (seq_length - len(all_frames))

        # 3. Normalize RGB frames to [0, 1] and convert to float32
        rgb_sequence = np.array(processed_frames, dtype=np.float32) / 255.0
        # Expected shape: (seq_length, H, W, 3) where H,W match IMG_SIZE
        if rgb_sequence.shape[1:3] != (IMG_SIZE[0], IMG_SIZE[1]):
             # This shouldn't happen if resize used IMG_SIZE_CV correctly, but double-check
             print(f"Warning: Frame dimensions mismatch {rgb_sequence.shape[1:3]} vs {IMG_SIZE} for {video_path}", file=sys.stderr)
             # Attempt resize again if needed? Or skip?
             # For now, proceed, but investigate if warning appears.

        # 4. Determine output path
        relative_path = os.path.relpath(video_path, dataset_base_path)
        output_dir = os.path.join(output_base_path, os.path.dirname(relative_path))
        base_filename = os.path.splitext(os.path.basename(relative_path))[0]
        output_filename = os.path.join(output_dir, f"{base_filename}_rgb.npy") # Suffix changed to _rgb

        # 5. Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 6. Save the RGB sequence
        np.save(output_filename, rgb_sequence)
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
    video_paths.sort() # <-- ADDED: Sort the paths for consistency
    return video_paths

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Precompute RGB frame sequences for Basketball-51 dataset.")
    # Default path adjusted for local use example
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51',
                        help='Path to the root directory of the Basketball-51 dataset.')
    parser.add_argument('--output_path', type=str, default='./rgb_data', # Default output dir for RGB
                        help='Path to the directory where RGB .npy files will be saved.')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() // 2),
                        help='Number of worker processes for parallel computation.')
    args = parser.parse_args()

    print(f"Using {args.workers} worker processes.")
    print(f"Output directory: {args.output_path}")

    video_paths = get_video_paths(args.dataset_path)
    if not video_paths:
        return # Exit if no videos found

    video_paths.sort() # <-- ADDED: Sort the paths for consistency

    # --- Select 50% of the videos (Consistent Selection) ---
    random.seed(SEED) # Set the random seed
    num_videos_to_use = int(len(video_paths) * 0.50)
    random.shuffle(video_paths) # Shuffle the list in place (deterministic)
    video_paths_subset = video_paths[:num_videos_to_use]
    print(f"Using a 50% subset: {len(video_paths_subset)} videos out of {len(video_paths)} total.")
    # --- End Subset Selection ---

    # Create the main output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Prepare partial function for multiprocessing map
    process_func = partial(extract_and_save_rgb,
                           dataset_base_path=args.dataset_path,
                           output_base_path=args.output_path,
                           seq_length=SEQ_LENGTH,
                           img_size_cv=IMG_SIZE_CV) # Pass cv2 compatible size

    # Use multiprocessing Pool
    processed_count = 0
    failed_videos = []
    with Pool(processes=args.workers) as pool:
        # Use tqdm for progress bar
        with tqdm(total=len(video_paths_subset), desc="Extracting RGB Frames") as pbar:
            # Use imap_unordered for potentially better performance and progress updates
            for video_path, success in pool.imap_unordered(process_func, video_paths_subset):
                if success:
                    processed_count += 1
                else:
                    failed_videos.append(video_path)
                pbar.update(1) # Update progress bar for each completed video

    print(f"\nFinished processing.")
    print(f"Successfully processed and saved RGB for {processed_count}/{len(video_paths_subset)} videos.")
    if failed_videos:
        print(f"Failed to process {len(failed_videos)} videos:")
        # for vid in failed_videos:
        #     print(f"  - {vid}") # Can be very verbose

if __name__ == "__main__":
    main() 