print("Python is working!")
print("Checking the Basketball-51 dataset path...")
import os

dataset_path = r'C:\Users\anubh\OneDrive\Desktop\Thesis\Basketball-51\Basketball-51'
if os.path.exists(dataset_path):
    print(f"Dataset path exists: {dataset_path}")
    print("Contents:")
    for item in os.listdir(dataset_path):
        print(f"  - {item}")
else:
    print(f"Dataset path does not exist: {dataset_path}")

print("Script completed successfully.") 