import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet18 import Resnet18


class MNISTModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = Resnet18(in_channel=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.backbone(x)[2]
        x = self.pool(x)
        x = self.fc(x)
        out = F.softmax(x)

        return out
