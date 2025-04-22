#!/usr/bin/env python3
"""
basketball51_action_recognition.py

Spatio-temporal action recognition on the Basketball-51 dataset using RGB frames and Farneback optical flow.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configuration
# Hard-coded local dataset path (modify as needed for Kaggle environment)
DATASET_PATH = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
SEQ_LENGTH    = 30            # number of frames per clip
IMG_SIZE      = (112, 112)    # width, height
BATCH_SIZE    = 8
EPOCHS        = 20


def find_dataset_path(name):
    """Search current directory tree for a folder named `name` and return its path."""
    for root, dirs, _ in os.walk('.'):
        if name in dirs:
            return os.path.join(root, name)
    raise FileNotFoundError(f"Dataset folder '{name}' not found")


def get_video_paths_and_labels(dataset_path):
    """Collect all .mp4 video paths and numeric labels based on subfolder names."""
    label_dirs = sorted([d for d in os.listdir(dataset_path)
                         if os.path.isdir(os.path.join(dataset_path, d))])
    label2id = {label: idx for idx, label in enumerate(label_dirs)}
    video_paths, labels = [], []
    for label in label_dirs:
        folder = os.path.join(dataset_path, label)
        for fname in os.listdir(folder):
            if fname.lower().endswith('.mp4'):
                video_paths.append(os.path.join(folder, fname))
                labels.append(label2id[label])
    return video_paths, labels, label2id


def extract_frames(video_path, seq_length, img_size):
    """Read a video, sample or pad to `seq_length`, resize frames, and normalize."""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    cap.release()
    if len(all_frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")
    # sample or pad
    if len(all_frames) >= seq_length:
        idxs = np.linspace(0, len(all_frames) - 1, seq_length).astype(int)
        frames = [all_frames[i] for i in idxs]
    else:
        frames = all_frames + [all_frames[-1]] * (seq_length - len(all_frames))
    frames = np.array(frames, dtype='float32') / 255.0
    return frames


def compute_optical_flow_sequence(frames):
    """Given an array of RGB frames, compute Farneback optical flow for each consecutive pair."""
    flows = []
    prev_gray = cv2.cvtColor((frames[0] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor((frames[i] * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev_gray = curr_gray
    # duplicate first flow to match sequence length
    flows.insert(0, flows[0])
    flows = np.stack(flows, axis=0)
    flows = flows.astype('float32')
    return flows


def prepare_data(paths, labels, seq_length, img_size, num_classes):
    """Extract RGB and flow sequences for each video, return fused X and one-hot y."""
    X_rgb, X_flow, y = [], [], []
    for vp, lbl in tqdm(zip(paths, labels), total=len(paths), desc='Preparing data'):
        frames = extract_frames(vp, seq_length, img_size)
        flow   = compute_optical_flow_sequence(frames)
        X_rgb.append(frames)
        X_flow.append(flow)
        y.append(lbl)
    X_rgb = np.array(X_rgb)
    X_flow = np.array(X_flow)
    # fuse modalities along channel axis: (B, T, H, W, 3+2)
    X = np.concatenate([X_rgb, X_flow], axis=-1)
    y = to_categorical(y, num_classes)
    return X, y


def build_model(input_shape, num_classes, epochs=EPOCHS):
    """Define a simple CNN-LSTM model for spatio-temporal classification."""
    inp = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same'))(inp)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(128)(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # use hard-coded dataset path
    data_dir = DATASET_PATH
    print(f"Found dataset at: {data_dir}")

    # collect paths and labels
    video_paths, labels, label2id = get_video_paths_and_labels(data_dir)
    num_classes = len(label2id)
    print(f"Found {len(video_paths)} videos across {num_classes} classes: {label2id}")

    # train/test split
    train_p, test_p, train_l, test_l = train_test_split(
        video_paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # prepare data
    X_train, y_train = prepare_data(train_p, train_l, SEQ_LENGTH, IMG_SIZE, num_classes)
    X_test,  y_test  = prepare_data(test_p, test_l, SEQ_LENGTH, IMG_SIZE, num_classes)

    # build & train model
    input_shape = (SEQ_LENGTH, IMG_SIZE[1], IMG_SIZE[0], 5)
    model = build_model(input_shape, num_classes)
    print(model.summary())
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(f"Test  Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")


def test_single_video(video_path=None, epochs=EPOCHS):
    """Test pipeline on a single video file (or first one found in dataset if None)."""
    # locate dataset (uses hardcoded path)
    data_dir = DATASET_PATH
    print(f"Dataset path: {data_dir}")

    # if no video specified, use first one found
    if video_path is None:
        video_paths, labels, label2id = get_video_paths_and_labels(data_dir)
        if not video_paths:
            print("No videos found in dataset at:", data_dir)
            return
        video_path = video_paths[0]
        true_label = labels[0]
        id2label = {idx: lbl for lbl, idx in label2id.items()}
        print(f"Selected video: {video_path}\nTrue label: {id2label[true_label]}")
    else:
        print(f"Using video: {video_path}")
        true_label = None

    # extract frames and flow
    frames = extract_frames(video_path, SEQ_LENGTH, IMG_SIZE)
    flow = compute_optical_flow_sequence(frames)
    
    print("Frames shape:", frames.shape)
    print("Flow shape:  ", flow.shape)
    
    # combine and add batch dimension
    X = np.concatenate([frames, flow], axis=-1)
    X = X[np.newaxis, ...]
    
    # build model
    _, _, label2id = get_video_paths_and_labels(data_dir)
    num_classes = len(label2id)
    input_shape = X.shape[1:]
    model = build_model(input_shape, num_classes, epochs=epochs)
    
    # get prediction
    preds = model.predict(X)
    pred_idx = int(np.argmax(preds[0]))
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    print(f"Predicted: {id2label[pred_idx]} (score={preds[0][pred_idx]:.4f})")
    return True


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_single_video()
    else:
        main() 