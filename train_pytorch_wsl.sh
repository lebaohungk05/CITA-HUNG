#!/bin/bash
export SITE_PACKAGES='/home/vigilante/.local/lib/python3.10/site-packages'
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

echo "--- Starting PyTorch Training (MobileNetV2) ---"
cd /mnt/d/fer2013/face_classification
python3 src/train_emotion_classifier_mobilenet_pytorch.py
