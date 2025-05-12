import torch
import torch.nn as nn
import torchvision.models as models




class resnet18(nn.Module):
    def __init__(self, embedding_dim=128):
        super(resnet18,self).__init__()

        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.feature_extractor(x) #1024>>>512
        x = self.embedding_head(x) #512>>>128
        return x
    

