#!/usr/bin/env python3
"""
create_video_subset.py

Creates a 20% subset of the Basketball-51 video dataset that matches
the precomputed RGB and Flow data subset. This ensures we have the
corresponding video files for our precomputed data.
"""
import os
import sys
import shutil
from tqdm import tqdm

def get_subset_video_names(precomputed_dir):
    """Get list of video names from the precomputed subset."""
    video_names = set()
    
    print(f"Scanning precomputed subset directory: {precomputed_dir}")
    for root, dirs, files in os.walk(precomputed_dir):
        for file in files:
            if file.endswith('_rgb.npz'):
                # Extract video name from RGB file name
                video_name = file.replace('_rgb.npz', '.mp4')
                video_names.add(video_name)
    
    print(f"Found {len(video_names)} videos in precomputed subset")
    return video_names

def create_video_subset(basketball51_dir, output_dir, subset_video_names):
    """Create subset of video files matching precomputed data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating video subset in: {output_dir}")
    print(f"Looking for {len(subset_video_names)} videos")
    
    # Track statistics
    found_count = 0
    missing_count = 0
    
    # Copy matching videos
    for root, dirs, files in os.walk(basketball51_dir):
        for file in files:
            if file in subset_video_names:
                # Get relative path to maintain directory structure
                rel_path = os.path.relpath(root, basketball51_dir)
                src_path = os.path.join(root, file)
                dst_dir = os.path.join(output_dir, rel_path)
                dst_path = os.path.join(dst_dir, file)
                
                # Create destination directory
                os.makedirs(dst_dir, exist_ok=True)
                
                # Copy video file
                try:
                    shutil.copy2(src_path, dst_path)
                    found_count += 1
                    print(f"\rCopied {found_count} videos", end="", flush=True)
                except Exception as e:
                    print(f"\nError copying {file}: {str(e)}")
                    missing_count += 1
    
    print(f"\n\nSubset creation completed!")
    print(f"Videos found and copied: {found_count}")
    print(f"Videos missing: {missing_count}")
    print(f"Output directory: {output_dir}")
    
    # Print directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def main():
    # Source directory containing full Basketball-51 dataset
    basketball51_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
    
    # Directory containing precomputed subset
    precomputed_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\precomputed_data_subset'
    
    # Output directory for video subset
    output_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51_subset'
    
    # Remove existing subset directory if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing subset directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Get list of videos from precomputed subset
    subset_video_names = get_subset_video_names(precomputed_dir)
    
    if not subset_video_names:
        print("No videos found in precomputed subset!")
        return
    
    # Create video subset
    create_video_subset(basketball51_dir, output_dir, subset_video_names)

if __name__ == "__main__":
    main() 