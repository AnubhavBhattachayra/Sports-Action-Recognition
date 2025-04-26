# Two-Stream ACA-Net for Basketball Action Recognition

This project implements a Two-Stream Attentive Context Aware Network (ACA-Net) for basketball action recognition on the Basketball-51 dataset. The model uses both RGB frames and optical flow as input to recognize 8 different basketball actions.

## Model Architecture

The model architecture consists of:

1. **Two-Stream Architecture**:
   - **RGB Stream**: Modified ResNet50 with ACA-Blocks, LSTA, and TSCI modules
   - **Flow Stream**: Same architecture as RGB stream but with different weights
   - **Fusion**: Concatenation of features (4096-dim) followed by fully connected layers

2. **Special Modules**:
   - **LSTA Module**: Long Short-Term Attention for temporal attention
   - **TSCI Module**: Temporal-Spatial-Channel Interaction for feature enhancement
   - **ACA Block**: Attentive Context Aware block built on top of ResNet blocks

3. **Classification Head**:
   - Feature concatenation (4096-dim)
   - Dense layer (4096 → 1024)
   - Dropout (0.5)
   - Final classification (1024 → 8 classes)

## Dataset

The Basketball-51 dataset consists of 8 action classes:
- 2p0: 2-point shot miss
- 2p1: 2-point shot make
- 3p0: 3-point shot miss
- 3p1: 3-point shot make
- fp0: Free throw miss
- fp1: Free throw make
- layup0: Layup miss
- layup1: Layup make

## Requirements

```
numpy>=1.19.5
opencv-python>=4.5.1
tensorflow>=2.4.0
tensorflow-addons>=0.13.0
tqdm>=4.56.0
scikit-learn>=0.24.1
matplotlib>=3.3.4
seaborn>=0.11.1
pandas>=1.2.0
scipy>=1.6.0
```

Install the requirements using:
```
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python run_two_stream_model.py --mode train --data_path /path/to/Basketball-51 --epochs 30
```

### Testing

To test the model on the test set:

```bash
python run_two_stream_model.py --mode test --data_path /path/to/Basketball-51 --model_path two_stream_aca_net_model.h5
```

To test on a specific video:

```bash
python run_two_stream_model.py --mode test --test_video /path/to/video.mp4 --model_path two_stream_aca_net_model.h5
```

### Visualization

To visualize feature embeddings using t-SNE:

```bash
python run_two_stream_model.py --mode visualize --visualize_type tsne --model_path two_stream_aca_net_model.h5
```

To visualize the confusion matrix:

```bash
python run_two_stream_model.py --mode visualize --visualize_type confusion --model_path two_stream_aca_net_model.h5
```

## Files

- `two_stream_aca_net.py`: Implementation of the Two-Stream ACA-Net model
- `run_two_stream_model.py`: Script to train, test, and visualize the model
- `basketball51_action_recognition.py`: Original implementation (used for compatibility)

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- AUC (Area Under the ROC Curve)
- Confusion Matrix

## Implementation Details

1. **Data Preprocessing**:
   - Extract 50 consecutive frames from each video
   - Compute optical flow using TV-L1 algorithm (fallback to Farneback)
   - Resize to 120x160 resolution
   - Normalize RGB to [0,1] and flow appropriately

2. **Training Process**:
   - Adam optimizer with learning rate 0.0001
   - Batch size 16
   - Early stopping with patience 5
   - Learning rate reduction on plateau
   - Cross-entropy loss

3. **Visualization**:
   - t-SNE for feature space visualization
   - Confusion matrix for error analysis
   - Training curves for monitoring progress

## References

This implementation is based on the ACA-Net architecture combining elements from:

1. ResNet50 for feature extraction
2. Long Short-Term Attention (LSTA) for temporal information
3. Temporal-Spatial-Channel Interaction (TSCI) for comprehensive feature modeling

## License

This project is licensed under the terms of the included LICENSE file.