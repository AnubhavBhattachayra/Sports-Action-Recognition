@echo off
echo Starting precomputation of RGB and Flow data...

:: Set paths - Modify these according to your system
set DATASET_PATH=C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51
set OUTPUT_PATH=C:\Users\anubh\OneDrive\Desktop\Thesis\Sports-Action-Recognition\precomputed_data

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_PATH%" mkdir "%OUTPUT_PATH%"

:: Run the preprocessing script
python precompute_combined.py --dataset_path "%DATASET_PATH%" --output_path "%OUTPUT_PATH%" --workers 4

echo.
echo Processing complete!
echo Check the output in: %OUTPUT_PATH%
pause 