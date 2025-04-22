# PowerShell script to check Python installation
Write-Host "Checking for Python installation..."

$pythonPaths = @(
    "python",
    "py",
    "python3",
    "C:\Python39\python.exe",
    "C:\Python310\python.exe",
    "C:\Python311\python.exe",
    "C:\Users\anubh\AppData\Local\Programs\Python\Python39\python.exe",
    "C:\Users\anubh\AppData\Local\Programs\Python\Python310\python.exe",
    "C:\Users\anubh\AppData\Local\Programs\Python\Python311\python.exe"
)

$pythonFound = $false

foreach ($path in $pythonPaths) {
    Write-Host "Trying $path..."
    try {
        $version = & $path -V 2>&1
        if ($?) {
            Write-Host "Found Python: $version at $path"
            $pythonFound = $true
            # Test importing tensorflow
            $tfTest = & $path -c "try: import tensorflow; print('TensorFlow', tensorflow.__version__); except Exception as e: print('Error:', e)"
            Write-Host "TensorFlow check: $tfTest"
            # Test importing OpenCV
            $cvTest = & $path -c "try: import cv2; print('OpenCV', cv2.__version__); except Exception as e: print('Error:', e)"
            Write-Host "OpenCV check: $cvTest"
            break
        }
    } catch {
        Write-Host "Failed with: $_"
    }
}

if (-not $pythonFound) {
    Write-Host "Python not found in common locations. Please check your Python installation."
} 