# This script runs the MobileNetV2 training on GPU using WSL.
# Make sure you have WSL installed with Python and TensorFlow.

Write-Host "Starting MobileNetV2 training on GPU via WSL..."
Write-Host "You can stop the process by pressing Ctrl+C in this window."
Write-Host "---------------------------------------------------------"

$wslCommand = @"
    # Set the path to your Python site-packages in WSL
    export SITE_PACKAGES='/home/vigilante/.local/lib/python3.10/site-packages'
    
    # Prepend NVIDIA libraries to the LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=\$SITE_PACKAGES/nvidia/cudnn/lib:\$SITE_PACKAGES/nvidia/cublas/lib:\$SITE_PACKAGES/nvidia/cufft/lib:\$SITE_PACKAGES/nvidia/curand/lib:\$SITE_PACKAGES/nvidia/cusolver/lib:\$SITE_PACKAGES/nvidia/cusparse/lib:\$SITE_PACKAGES/nvidia/nccl/lib:\$SITE_PACKAGES/nvidia/nvjitlink/lib:\$LD_LIBRARY_PATH
    
    echo '--- Starting Training ---'
    
    # Change to the source directory within WSL
    cd /mnt/d/fer2013/face_classification/src
    
    # Run the training script
    python3 train_emotion_classifier_mobilenet.py
"@

wsl bash -c $wslCommand

Write-Host "---------------------------------------------------------"
Write-Host "Training process finished or was interrupted."
