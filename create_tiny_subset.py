#!/usr/bin/env python3
"""
create_tiny_subset.py

Creates a 0.5% subset of the precomputed RGB and Flow data while maintaining
the original directory structure. This is useful for quick testing and development
with minimal memory requirements.
"""
import os
import sys
import shutil
import random
import numpy as np
from tqdm import tqdm

def get_all_files(base_dir):
    """Get all RGB and Flow files while maintaining directory structure."""
    rgb_files = []
    flow_files = []
    
    print(f"Scanning directory: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_rgb.npz'):
                rgb_path = os.path.join(root, file)
                flow_path = rgb_path.replace('_rgb.npz', '_flow.npz')
                if os.path.exists(flow_path):
                    rgb_files.append(rgb_path)
                    flow_files.append(flow_path)
    
    print(f"Found {len(rgb_files)} pairs of RGB and Flow files")
    return rgb_files, flow_files

def create_subset(rgb_files, flow_files, output_base_dir, subset_fraction=0.005, seed=42):
    """Create a subset of the data while maintaining directory structure."""
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Calculate number of files to select
    num_files = len(rgb_files)
    num_subset = max(int(num_files * subset_fraction), 1)  # Ensure at least 1 file
    
    # Randomly select indices
    indices = random.sample(range(num_files), num_subset)
    
    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\nCreating {subset_fraction*100}% subset ({num_subset} files)")
    print(f"Output directory: {output_base_dir}")
    
    # Copy selected files
    for idx in tqdm(indices, desc="Copying files"):
        rgb_src = rgb_files[idx]
        flow_src = flow_files[idx]
        
        # Get relative paths
        rgb_rel_path = os.path.relpath(rgb_src, os.path.dirname(os.path.dirname(rgb_src)))
        flow_rel_path = os.path.relpath(flow_src, os.path.dirname(os.path.dirname(flow_src)))
        
        # Create destination paths
        rgb_dst = os.path.join(output_base_dir, rgb_rel_path)
        flow_dst = os.path.join(output_base_dir, flow_rel_path)
        
        # Create destination directories
        os.makedirs(os.path.dirname(rgb_dst), exist_ok=True)
        os.makedirs(os.path.dirname(flow_dst), exist_ok=True)
        
        # Copy files
        shutil.copy2(rgb_src, rgb_dst)
        shutil.copy2(flow_src, flow_dst)
    
    print("\nSubset creation completed!")
    print(f"Total files copied: {num_subset * 2} (RGB + Flow)")
    print(f"Output directory: {output_base_dir}")

def main():
    # Source directory containing precomputed data
    source_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\precomputed_data'
    
    # Output directory for tiny subset
    output_dir = r'C:\Users\anubh\OneDrive\Desktop\Thesis\precomputed_data_tiny'
    
    # Get all file pairs
    rgb_files, flow_files = get_all_files(source_dir)
    
    if not rgb_files:
        print("No RGB/Flow file pairs found!")
        return
    
    # Create subset with only 0.5% of the data
    create_subset(rgb_files, flow_files, output_dir, subset_fraction=0.005)
    
    # Verify the subset
    subset_rgb_files, subset_flow_files = get_all_files(output_dir)
    print(f"\nVerification:")
    print(f"Original dataset: {len(rgb_files)} pairs")
    print(f"Tiny subset dataset: {len(subset_rgb_files)} pairs")
    print(f"Subset percentage: {(len(subset_rgb_files) / len(rgb_files)) * 100:.2f}%")

if __name__ == "__main__":
    main() 