# Set paths - Modify these according to your system
$DATASET_PATH = "C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51"
$OUTPUT_PATH = "C:\Users\anubh\OneDrive\Desktop\Thesis\Sports-Action-Recognition\precomputed_data"

# Create output directory if it doesn't exist
if (-not (Test-Path $OUTPUT_PATH)) {
    New-Item -ItemType Directory -Path $OUTPUT_PATH | Out-Null
}

Write-Host "Starting precomputation of RGB and Flow data..."
Write-Host "Dataset path: $DATASET_PATH"
Write-Host "Output path: $OUTPUT_PATH"

# Run the preprocessing script
python precompute_combined.py --dataset_path $DATASET_PATH --output_path $OUTPUT_PATH --workers 4

Write-Host "`nProcessing complete!"
Write-Host "Check the output in: $OUTPUT_PATH"
Read-Host "Press Enter to continue..." 