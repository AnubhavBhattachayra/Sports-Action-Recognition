#!/usr/bin/env python3
"""
check_data_pairs.py

Script to verify that every flow file has a corresponding RGB file and vice versa
in the precomputed data directory.
"""
import os
import argparse
from collections import defaultdict
from tqdm import tqdm

def check_data_pairs(precomputed_dir):
    """Check for matching RGB and flow files in the precomputed data directory."""
    # Initialize counters and storage
    rgb_files = defaultdict(list)
    flow_files = defaultdict(list)
    missing_pairs = []
    
    # Walk through the directory
    print("Scanning precomputed data directory...")
    for root, _, files in os.walk(precomputed_dir):
        for file in files:
            if file.endswith('_rgb.npz'):
                # Get the base name without _rgb.npz
                base_name = file[:-8]  # Remove '_rgb.npz'
                rgb_files[base_name].append(os.path.join(root, file))
            elif file.endswith('_flow.npz'):
                # Get the base name without _flow.npz
                base_name = file[:-9]  # Remove '_flow.npz'
                flow_files[base_name].append(os.path.join(root, file))
    
    # Check for matches
    print("\nChecking for matching pairs...")
    all_bases = set(list(rgb_files.keys()) + list(flow_files.keys()))
    
    for base_name in tqdm(all_bases):
        rgb_matches = rgb_files[base_name]
        flow_matches = flow_files[base_name]
        
        if not rgb_matches and flow_matches:
            missing_pairs.append((None, flow_matches[0], "Missing RGB file"))
        elif rgb_matches and not flow_matches:
            missing_pairs.append((rgb_matches[0], None, "Missing Flow file"))
    
    # Print results
    print("\n=== Data Pair Check Results ===")
    print(f"Total unique video bases found: {len(all_bases)}")
    print(f"RGB files found: {len(rgb_files)}")
    print(f"Flow files found: {len(flow_files)}")
    print(f"Missing pairs found: {len(missing_pairs)}")
    
    if missing_pairs:
        print("\nMissing pairs details:")
        for rgb_file, flow_file, reason in missing_pairs:
            if rgb_file:
                print(f"Missing flow for: {rgb_file}")
            if flow_file:
                print(f"Missing RGB for: {flow_file}")
    else:
        print("\nAll files have matching pairs! ✓")

def main():
    parser = argparse.ArgumentParser(description="Check for matching RGB and flow files in precomputed data")
    parser.add_argument('--precomputed_dir', type=str, required=True,
                      help='Path to the precomputed data directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.precomputed_dir):
        print(f"Error: Directory {args.precomputed_dir} does not exist!")
        return
    
    check_data_pairs(args.precomputed_dir)

if __name__ == "__main__":
    main() 