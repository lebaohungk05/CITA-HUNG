import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def resnet18_custom(num_classes=7, pretrained=True):
    """
    ResNet-18 Customized for Small Images (48x48)
    Changes:
    - Conv1: 7x7 stride 2 -> 3x3 stride 1 (Preserve spatial dim)
    - MaxPool: Removed (Preserve spatial dim)
    - SE Blocks: Added
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # --- 1. ARCHITECTURE MODIFICATION FOR 48x48 IMAGES ---
    # ResNet chuẩn: Conv 7x7 stride 2 + MaxPool stride 2 -> Downsample 4 lần ngay đầu
    # Custom: Conv 3x3 stride 1 + No MaxPool -> Giữ nguyên kích thước 48x48 vào các lớp sau
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # Loại bỏ MaxPool hoàn toàn

    # --- 2. INTEGRATE SE BLOCKS & DROPOUT ---
    p_dropout = 0.2 
    
    model.layer1 = nn.Sequential(model.layer1, SEBlock(64), nn.Dropout2d(p=p_dropout))
    model.layer2 = nn.Sequential(model.layer2, SEBlock(128), nn.Dropout2d(p=p_dropout))
    model.layer3 = nn.Sequential(model.layer3, SEBlock(256), nn.Dropout2d(p=p_dropout))
    model.layer4 = nn.Sequential(model.layer4, SEBlock(512), nn.Dropout2d(p=p_dropout))

    # --- 3. CLASSIFIER ---
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    return model
