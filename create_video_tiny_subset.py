#!/usr/bin/env python3
"""
create_video_tiny_subset.py

Creates a 0.5% subset of the Basketball-51 video dataset that matches
the precomputed tiny RGB and Flow data subset. This ensures we have the
corresponding video files for our precomputed tiny data.
"""
import os
import sys
import shutil
from tqdm import tqdm

def get_subset_video_names(precomputed_dir):
    """Get list of video names from the precomputed tiny subset."""
    video_names = set()
    
    print(f"Scanning precomputed tiny subset directory: {precomputed_dir}")
    for root, dirs, files in os.walk(precomputed_dir):
        for file in files:
            if file.endswith('_rgb.npz'):
                # Extract video name from RGB file name
                video_name = file.replace('_rgb.npz', '.mp4')
                video_names.add(video_name)
    
    print(f"Found {len(video_names)} videos in precomputed tiny subset")
    return video_names

def create_video_subset(basketball51_dir, output_dir, subset_video_names):
    """Create subset of video files matching precomputed tiny data."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating video tiny subset in: {output_dir}")
    print(f"Looking for {len(subset_video_names)} videos")
    
    # Track statistics
    found_count = 0
    missing_count = 0
    missing_videos = []
    
    # Copy matching videos with progress bar
    with tqdm(total=len(subset_video_names), desc="Copying videos") as pbar:
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
                        pbar.update(1)
                    except Exception as e:
                        print(f"\nError copying {file}: {str(e)}")
                        missing_count += 1
                        missing_videos.append(file)
    
    print(f"\n\nTiny subset creation completed!")
    print(f"Videos found and copied: {found_count}")
    print(f"Videos missing: {missing_count}")
    if missing_count > 0:
        print("Missing videos:")
        for v in missing_videos:
            print(f"  - {v}")
    print(f"Output directory: {output_dir}")
    
    # Calculate percentage of full dataset
    total_videos = 0
    for root, dirs, files in os.walk(basketball51_dir):
        for file in files:
            if file.endswith('.mp4'):
                total_videos += 1
    
    if total_videos > 0:
        percentage = (found_count / total_videos) * 100
        print(f"\nSubset percentage: {percentage:.2f}% of original dataset ({found_count}/{total_videos})")
    
    # Print directory structure
    print("\nDirectory structure:")
    class_counts = {}
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        if level == 1:  # Class level
            class_name = os.path.basename(root)
            class_counts[class_name] = len([f for f in files if f.endswith('.mp4')])
    
    # Print class distribution
    if class_counts:
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} videos")

def main():
    # Source directory containing full Basketball-51 dataset
    basketball51_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
    
    # Directory containing precomputed tiny subset
    precomputed_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\precomputed_data_tiny'
    
    # Output directory for video tiny subset
    output_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51_tiny'
    
    # Remove existing subset directory if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing tiny subset directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Get list of videos from precomputed tiny subset
    subset_video_names = get_subset_video_names(precomputed_dir)
    
    if not subset_video_names:
        print("No videos found in precomputed tiny subset!")
        return
    
    # Create video tiny subset
    create_video_subset(basketball51_dir, output_dir, subset_video_names)

if __name__ == "__main__":
    main() 