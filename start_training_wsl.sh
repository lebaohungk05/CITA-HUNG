#!/bin/bash
echo "Setting up environment for GPU training in WSL..."

export SITE_PACKAGES="/home/vigilante/.local/lib/python3.10/site-packages"

# Add all nvidia library paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/nccl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/nvjitlink/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cuda_runtime/lib

echo "Checking GPU availability..."
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')));"

echo "Starting training in background..."
cd src
# Run in background, redirect output
nohup python3 train_emotion_classifier_mobilenet.py > ../wsl_training.log 2>&1 &
echo "Training process started with PID $!"