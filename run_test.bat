@echo off
echo Running Basketball-51 multiple sample training script
cd test
python test_training_multiple.py --data_dir "C:\Users\anubh\OneDrive\Desktop\Thesis\precomputed_data" --output_dir "..\test_output" --num_samples 10
echo Training complete
pause 