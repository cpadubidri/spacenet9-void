import torch
import torch.nn as nn
import torchvision.models as models




class resnet18(nn.Module):
    def __init__(self, embedding_dim=128):
        super(resnet18, self).__init__()

        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # ❗ keep spatial info

        self.embedding_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, embedding_dim, kernel_size=1)  # final: (B, 128, H', W')
        )

    def forward(self, x):
        x = self.feature_extractor(x)     # (B, 512, H', W') ← e.g., 32x32 if input is 512x512
        x = self.embedding_head(x)        # (B, 128, H', W')
        return x

class ResNet50Spatial(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet50Spatial, self).__init__()

        #pretrained resnet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        #stride and dilation to reduce downsampling
        self.backbone = nn.Sequential(
            resnet.conv1,   # -> (B, 64, 256, 256)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # -> (B, 64, 128, 128)

            resnet.layer1,  # -> (B, 256, 128, 128)
            resnet.layer2,  # -> (B, 512, 64, 64)
            resnet.layer3,  # -> (B, 1024, 32, 32)

            # Modify layer4 to keep output at 32×32
            self._dilated_layer4(resnet.layer4)  # -> (B, 2048, 32, 32)
        )

        #embedding head: reduce channels to desired embedding dim
        self.embedding_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embedding_dim, kernel_size=1)  # -> (B, 128, 32, 32)
        )

    def _dilated_layer4(self, layer4):
        for n, m in layer4.named_modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)
        return layer4

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding_head(x)
        return x


