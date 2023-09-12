import torch
import torch.nn as nn

from model.darknet import get_cbl, get_Bottleneck, ResBlock


class Resnet18(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.stage0 = nn.Sequential(
            get_cbl(in_channel=in_channel, out_channel=32, kernel_size=3, stride=1, padding=1),
            get_cbl(in_channel=32, out_channel=64, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=1, in_channel=64)
        )

        self.stage1 = nn.Sequential(
            get_cbl(in_channel=64, out_channel=128, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=2, in_channel=128)
        )

        self.stage2 = nn.Sequential(
            get_cbl(in_channel=128, out_channel=256, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=2, in_channel=256)
        )

        self.stage3 = nn.Sequential(
            get_cbl(in_channel=256, out_channel=512, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=2, in_channel=512)
        )

        self.stage4 = nn.Sequential(
            get_cbl(in_channel=512, out_channel=1024, kernel_size=3, stride=2, padding=1),
            ResBlock(num_block=2, in_channel=1024)
        )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        out0 = self.stage2(x)
        out1 = self.stage3(out0)
        out2 = self.stage4(out1)

        return out0, out1, out2
