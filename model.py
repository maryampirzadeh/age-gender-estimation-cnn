import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class AgeGenderResNet50(nn.Module):
    def __init__(self, pretrained=True, dropout=0.4):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features).squeeze(1)
        gender = self.gender_head(features)
        return age, gender