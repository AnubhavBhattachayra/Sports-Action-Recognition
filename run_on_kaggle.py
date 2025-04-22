"""
Run this script in a Kaggle notebook to test and run the basketball recognition pipeline.
Make sure to add basketball51_action_recognition.py to your Kaggle notebook.
"""
import os
import sys
import glob
import numpy as np

# Setup for Kaggle environment
KAGGLE_DATASET_PATH = '/kaggle/input/basketball-51'

# Check if we're running on Kaggle
def is_kaggle():
    return os.path.exists('/kaggle/input')

def setup_environment():
    if is_kaggle():
        # Import the module and modify its dataset path
        import basketball51_action_recognition as bball
        # Update the path for Kaggle
        bball.DATASET_PATH = KAGGLE_DATASET_PATH
        return bball
    else:
        print("Not running on Kaggle. Using local path.")
        import basketball51_action_recognition as bball
        return bball

def run_test(bball_module):
    """Run a test on a single video"""
    print("Testing on a single video...")
    try:
        bball_module.test_single_video()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()

def run_training_for_one_epoch(bball_module):
    """Run training for just one epoch and log results"""
    print("Running training for one epoch...")
    
    # Monkey-patch the main() function to only run for one epoch
    original_main = bball_module.main
    
    def main_one_epoch():
        # Use the same code structure as in the original main
        data_dir = bball_module.DATASET_PATH
        print(f"Dataset path: {data_dir}")
        
        # Get video paths and labels
        video_paths, labels, label2id = bball_module.get_video_paths_and_labels(data_dir)
        num_classes = len(label2id)
        print(f"Found {len(video_paths)} videos across {num_classes} classes: {label2id}")
        
        # Take a small subset for quick testing
        MAX_VIDEOS = 50  # Small enough for quick test
        if len(video_paths) > MAX_VIDEOS:
            print(f"Using only {MAX_VIDEOS} videos for quick test")
            from sklearn.model_selection import train_test_split
            train_p, _, train_l, _ = train_test_split(
                video_paths, labels,
                train_size=MAX_VIDEOS,
                stratify=labels,
                random_state=42
            )
        else:
            train_p, train_l = video_paths, labels
        
        # We'll use the same small subset for both train and test for quick verification
        test_p, test_l = train_p[:10], train_l[:10]
        
        # Prepare data
        X_train, y_train = bball_module.prepare_data(train_p, train_l, bball_module.SEQ_LENGTH, bball_module.IMG_SIZE, num_classes)
        X_test, y_test = bball_module.prepare_data(test_p, test_l, bball_module.SEQ_LENGTH, bball_module.IMG_SIZE, num_classes)
        
        # Build & train model for just ONE epoch
        input_shape = (bball_module.SEQ_LENGTH, bball_module.IMG_SIZE[1], bball_module.IMG_SIZE[0], 5)
        model = bball_module.build_model(input_shape, num_classes)
        print(model.summary())
        
        # Set up a metrics logger
        class MetricsLogger:
            def on_epoch_end(self, epoch, logs=None):
                with open('training_log.txt', 'a') as f:
                    metrics = ' - '.join(f"{k}: {v:.4f}" for k, v in logs.items())
                    f.write(f"Epoch {epoch+1}: {metrics}\n")
        
        # Train for just one epoch
        from tensorflow.keras.callbacks import Callback
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=1,  # Just one epoch
            batch_size=bball_module.BATCH_SIZE,
            callbacks=[MetricsLogger()]
        )
        
        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, batch_size=bball_module.BATCH_SIZE)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        
        # Write final results to log
        with open('training_log.txt', 'a') as f:
            f.write(f"\nFinal Test Loss: {loss:.4f}\n")
            f.write(f"Final Test Accuracy: {acc:.4f}\n")
            
        return model, history
    
    # Replace main with our one-epoch version
    bball_module.main = main_one_epoch
    
    try:
        # Create log file
        with open('training_log.txt', 'w') as f:
            f.write("Basketball-51 Action Recognition Training Log\n")
            f.write("==========================================\n\n")
        
        # Run the modified main
        model, history = bball_module.main()
        print("One-epoch training completed successfully!")
        return model, history
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original main
        bball_module.main = original_main

if __name__ == "__main__":
    # Setup environment and get the basketball module
    bball = setup_environment()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            run_test(bball)
        elif sys.argv[1] == 'train':
            run_training_for_one_epoch(bball)
    else:
        print("Choose an option:")
        print("1. Test on a single video")
        print("2. Run training for one epoch")
        
        choice = input("Enter option (1 or 2): ")
        if choice == '1':
            run_test(bball)
        elif choice == '2':
            run_training_for_one_epoch(bball)
        else:
            print("Invalid choice. Exiting.") 