#!/usr/bin/env python3
"""
visualize_flow.py

Visualize RGB frames with optical flow overlaid using HSV color mapping.
Loads precomputed RGB and flow data, creates a visualization,
and saves it as a video.
"""
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def load_data(rgb_path, flow_path):
    """Load RGB and flow data from compressed npz files."""
    rgb_data = np.load(rgb_path)['data']
    flow_data = np.load(flow_path)['data']
    return rgb_data, flow_data

def visualize_flow_hsv(frame, flow):
    """Visualize optical flow using HSV color mapping.
    
    Args:
        frame: RGB frame
        flow: Optical flow field (2 channels: x and y)
    
    Returns:
        Visualization with flow overlaid
    """
    # Convert flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude to 0-1 range
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Create HSV image
    hsv = np.zeros_like(frame)
    
    # Set Hue based on flow direction (0-180 for OpenCV)
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Set Saturation based on flow magnitude
    hsv[..., 1] = magnitude * 255
    
    # Set Value to maximum for clear visualization
    hsv[..., 2] = 255
    
    # Convert HSV to BGR
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Create a smaller visualization
    h, w = frame.shape[:2]
    vis_size = min(h, w) // 4  # Make visualization 1/4 of the smaller dimension
    flow_small = cv2.resize(flow_rgb, (vis_size, vis_size))
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    alpha = 0.3  # More transparent
    cv2.addWeighted(flow_small, alpha, 
                   overlay[0:vis_size, 0:vis_size], 1-alpha, 0,
                   overlay[0:vis_size, 0:vis_size])
    
    # Add a border around the visualization
    cv2.rectangle(overlay, (0, 0), (vis_size, vis_size), (255, 255, 255), 2)
    
    # Create a small color wheel legend
    wheel_size = vis_size // 2
    wheel = np.zeros((wheel_size, wheel_size, 3), dtype=np.uint8)
    center = wheel_size // 2
    
    # Create color wheel showing direction and magnitude
    for y in range(wheel_size):
        for x in range(wheel_size):
            dx = x - center
            dy = y - center
            angle = np.arctan2(dy, dx)
            magnitude = np.sqrt(dx*dx + dy*dy)
            
            if magnitude <= center:
                # Convert angle to hue (0-180 for OpenCV)
                hue = (angle * 180 / np.pi / 2) % 180
                # Set saturation based on distance from center
                saturation = min(255, int(magnitude * 255 / center))
                # Set value to maximum
                value = 255
                
                # Convert HSV to BGR
                hsv_pixel = np.uint8([[[hue, saturation, value]]])
                bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
                wheel[y, x] = bgr_pixel[0, 0]
    
    # Add wheel to the bottom-right of the visualization
    wheel_pos = (vis_size - wheel_size, vis_size - wheel_size)
    overlay[wheel_pos[1]:wheel_pos[1]+wheel_size, 
           wheel_pos[0]:wheel_pos[0]+wheel_size] = wheel
    
    # Add small text labels
    font_scale = 0.4
    thickness = 1
    cv2.putText(overlay, "Flow", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness)
    cv2.putText(overlay, "Dir", (wheel_pos[0] + 5, wheel_pos[1] + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return overlay

def create_flow_video(rgb_sequence, flow_sequence, output_path, fps=30):
    """Create a video with flow visualization."""
    # Get dimensions
    h, w = rgb_sequence[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Process each frame
    for rgb_frame, flow_frame in tqdm(zip(rgb_sequence, flow_sequence), 
                                    total=len(rgb_sequence),
                                    desc="Creating video"):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Create flow visualization
        vis_frame = visualize_flow_hsv(frame_bgr, flow_frame)
        
        # Write frame
        out.write(vis_frame)
    
    # Release video writer
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize RGB frames with optical flow")
    parser.add_argument('--rgb_path', type=str, required=True,
                        help='Path to the RGB npz file')
    parser.add_argument('--flow_path', type=str, required=True,
                        help='Path to the flow npz file')
    parser.add_argument('--output_path', type=str, default='output.mp4',
                        help='Path to save the output video')
    args = parser.parse_args()
    
    # Load data
    print("Loading RGB and flow data...")
    rgb_sequence, flow_sequence = load_data(args.rgb_path, args.flow_path)
    
    # Create video
    print("Creating visualization video...")
    create_flow_video(rgb_sequence, flow_sequence, args.output_path)
    
    print("Done!")

if __name__ == "__main__":
    main() 