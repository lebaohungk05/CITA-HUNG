import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Giúp model tập trung vào các channel quan trọng (Attention).
    Tăng Accuracy mà chỉ tốn rất ít tham số.
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

def resnet18_pytorch(num_classes=7, pretrained=True):
    """
    ResNet-18 + SE Attention + Dropout
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # 1. Sửa Conv1 (3 kênh -> 1 kênh)
    original_conv = model.conv1
    new_conv = nn.Conv2d(1, original_conv.out_channels, 
                         kernel_size=original_conv.kernel_size, 
                         stride=original_conv.stride, 
                         padding=original_conv.padding, 
                         bias=False)
    
    if pretrained:
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
    model.conv1 = new_conv

    # 2. Tích hợp SE Block (Attention) + Dropout nhẹ
    # SE Block đặt sau mỗi stage để tinh chỉnh đặc trưng
    p_dropout = 0.05 # Giảm dropout xuống thấp vì đã có SE Block làm regularization tốt
    
    model.layer1 = nn.Sequential(model.layer1, SEBlock(64), nn.Dropout2d(p=p_dropout))
    model.layer2 = nn.Sequential(model.layer2, SEBlock(128), nn.Dropout2d(p=p_dropout))
    model.layer3 = nn.Sequential(model.layer3, SEBlock(256), nn.Dropout2d(p=p_dropout))
    model.layer4 = nn.Sequential(model.layer4, SEBlock(512), nn.Dropout2d(p=p_dropout))

    # 3. Fully Connected Layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), # Giảm nhẹ FC dropout
        nn.Linear(num_ftrs, num_classes)
    )

    return model
