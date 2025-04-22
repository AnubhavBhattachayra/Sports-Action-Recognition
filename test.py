#!/usr/bin/env python3
"""
test.py

Quick test for Basketball-51 recognition pipeline on a single video.
"""
import sys
import os
import numpy as np
import logging
import datetime
import tensorflow as tf

# Reduce TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from basketball51_action_recognition import (
    DATASET_PATH, SEQ_LENGTH, IMG_SIZE,
    get_video_paths_and_labels,
    extract_frames,
    compute_optical_flow_sequence,
    build_model
)

# Set up logging
log_filename = f"basketball51_test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        logging.info("Starting Basketball-51 recognition test")
        print("Starting Basketball-51 recognition test")
        
        # Check dataset path
        if not os.path.exists(DATASET_PATH):
            error_msg = f"Dataset path does not exist: {DATASET_PATH}"
            logging.error(error_msg)
            print(error_msg)
            return
            
        logging.info(f"Dataset path exists: {DATASET_PATH}")
        print(f"Dataset path exists: {DATASET_PATH}")
        
        # determine which video to test
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            logging.info(f"Using video from command-line: {video_path}")
            print(f"Using video from command-line: {video_path}")
            true_label = None
        else:
            video_paths, labels, label2id = get_video_paths_and_labels(DATASET_PATH)
            if not video_paths:
                error_msg = f"No videos found in dataset at: {DATASET_PATH}"
                logging.error(error_msg)
                print(error_msg)
                return
                
            video_path = video_paths[0]
            true_label = labels[0]
            id2label = {idx: lbl for lbl, idx in label2id.items()}
            logging.info(f"Selected sample video: {video_path}\nTrue label: {id2label[true_label]}")
            print(f"Selected sample video: {video_path}\nTrue label: {id2label[true_label]}")

        # extract frames and optical flow
        logging.info("Extracting frames...")
        print("Extracting frames...")
        frames = extract_frames(video_path, SEQ_LENGTH, IMG_SIZE)
        
        logging.info("Computing optical flow...")
        print("Computing optical flow...")
        flow = compute_optical_flow_sequence(frames)

        frame_shape_info = f"Frames shape: {frames.shape}"
        flow_shape_info = f"Flow shape:   {flow.shape}"
        logging.info(frame_shape_info)
        logging.info(flow_shape_info)
        print(frame_shape_info)
        print(flow_shape_info)

        # fuse modalities and add batch dim
        X = np.concatenate([frames, flow], axis=-1)
        X = X[np.newaxis, ...]

        # build model and run prediction
        logging.info("Building model...")
        print("Building model...")
        _, _, label2id = get_video_paths_and_labels(DATASET_PATH)
        num_classes = len(label2id)
        input_shape = X.shape[1:]
        model = build_model(input_shape, num_classes, epochs=1)  # Set to 1 epoch

        logging.info("Running prediction...")
        print("Running prediction...")
        preds = model.predict(X, verbose=1)
        pred_idx = int(np.argmax(preds[0]))
        id2label = {idx: lbl for lbl, idx in label2id.items()}
        prediction_info = f"Predicted: {id2label[pred_idx]} (score={preds[0][pred_idx]:.4f})"
        logging.info(prediction_info)
        print(prediction_info)
        
        logging.info("Test completed successfully")
        print("Test completed successfully")
        
    except Exception as e:
        error_msg = f"Error during test: {str(e)}"
        logging.error(error_msg)
        print(error_msg)

if __name__ == '__main__':
    main() 