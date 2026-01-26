# Script chạy training Custom ResNet + MixUp trên Windows
Write-Host "Starting Custom ResNet Training (MixUp Mode)..."
Write-Host "Model: ResNet18 Custom (No MaxPool, 3x3 Conv)"
Write-Host "MixUp Alpha: 0.4"
Write-Host "Target: >74% Accuracy"
Write-Host "---------------------------------------------------------"

# Chuyển vào thư mục chứa script python
Set-Location -Path "face_classification/src"

# Chạy lệnh training
python train_resnet_mixup.py

Write-Host "---------------------------------------------------------"
Write-Host "Training finished."
Pause
